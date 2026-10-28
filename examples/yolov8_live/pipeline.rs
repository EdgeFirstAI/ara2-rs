// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Threaded inference pipeline.
//!
//! The stages of a frame use different hardware — GPU letterbox, NPU
//! inference, CPU decode, then GPU drawing — so running them on their own
//! threads lets consecutive frames overlap:
//!
//! ```text
//!  main (camera, display)      pre (GPU)           npu (ARA-2)        decode (CPU)
//!  frame N+3 in      ──────▶   letterbox N+2  ──▶  infer N+1    ──▶   decode N   ──▶  main: draw N
//!                              into a free slot     into a pooled
//!                                                   output set
//! ```
//!
//! One model is loaded on the ARA-2; each inference names the input and
//! output buffers it uses (`Model::submit_with_io_set`). A *slot* is an
//! input set the GPU letterboxes into, so with two slots the GPU prepares the
//! next frame while the ARA-2 reads the current one.
//!
//! Work moves between threads by ownership: a slot's input buffer travels
//! with its frame and returns once the ARA-2 has read it, and output sets
//! come from a small pool the decode thread refills. With DMA-BUF sets the
//! decoder reads an output set in place, so no stage copies model outputs.
//! Every channel is bounded, so a full pipeline pushes back to the camera,
//! which then drops frames rather than queue them.

use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::time::{Duration, Instant};

use ara2::{DEFAULT_TIMEOUT_MS, Model};
use edgefirst_decoder::{Decoder, DetectBox, ProtoData, Segmentation};
use edgefirst_image::{ConvertOptions, ImageProcessor, ImageProcessorTrait as _};
use edgefirst_tensor::TensorDyn;

use crate::error::Result;
use crate::{ModelInput, OutputBuffers};

/// A camera frame on its way through the pipeline.
#[derive(Debug)]
pub struct Frame {
    /// libcamera buffer index (the request cookie).
    pub index: usize,
    /// The buffer imported as a HAL image.
    pub image: TensorDyn,
    /// When the frame entered the pipeline.
    pub entered: Instant,
}

/// Time spent on a frame in each pipeline thread.
#[derive(Debug, Default, Clone, Copy)]
pub struct StageTimes {
    pub preprocess: Duration,
    /// Submission to completion, as the npu thread waits for it.
    pub infer: Duration,
    /// The ARA-2's own execution time, as the runtime reports it.
    pub npu: Duration,
    pub decode: Duration,
}

/// A slot's input buffer, passed between the pre and npu threads as the
/// right to use that slot.
pub struct Slot {
    input: ModelInput,
}

impl Slot {
    pub fn new(input: ModelInput) -> Self {
        Self { input }
    }
}

/// A letterboxed frame waiting for the NPU.
pub struct Prepared {
    slot: Slot,
    frame: Frame,
    times: StageTimes,
}

/// A frame whose outputs the ARA-2 has written.
pub struct Inferred {
    frame: Frame,
    outputs: OutputBuffers,
    times: StageTimes,
}

/// A decoded frame, ready to draw.
#[derive(Debug)]
pub struct Decoded {
    pub frame: Frame,
    pub detections: Vec<DetectBox>,
    pub masks: Vec<Segmentation>,
    pub protos: Option<ProtoData>,
    pub times: StageTimes,
}

/// What the main thread waits on: camera frames and decoded results.
#[derive(Debug)]
pub enum Event {
    Frame(libcamera::request::Request),
    Decoded(Box<Decoded>),
}

/// Pre thread: letterbox each frame into a free slot's input buffer.
pub fn preprocess(
    mut processor: ImageProcessor,
    letterbox: ConvertOptions,
    frames: &Receiver<Frame>,
    free: &Receiver<Slot>,
    prepared: &SyncSender<Prepared>,
) -> Result<()> {
    while let Ok(frame) = frames.recv() {
        let Ok(mut slot) = free.recv() else {
            break;
        };
        let start = Instant::now();
        processor.convert(&frame.image, slot.input.image(), Some(letterbox))?;
        slot.input.stage()?;
        let times = StageTimes {
            preprocess: start.elapsed(),
            ..StageTimes::default()
        };
        if prepared.send(Prepared { slot, frame, times }).is_err() {
            break;
        }
    }
    Ok(())
}

/// Npu thread: run each prepared slot into a pooled output set, then hand
/// the slot back to the pre thread.
pub fn infer(
    mut model: Model,
    prepared: &Receiver<Prepared>,
    free: &Sender<Slot>,
    pool: &Receiver<OutputBuffers>,
    inferred: &SyncSender<Inferred>,
) -> Result<()> {
    while let Ok(Prepared {
        slot,
        frame,
        mut times,
    }) = prepared.recv()
    {
        let Ok(outputs) = pool.recv() else {
            break;
        };
        let start = Instant::now();
        let stats = model
            .submit_with_io_set(slot.input.set(), outputs.set())?
            .wait(DEFAULT_TIMEOUT_MS)?;
        outputs.unstage()?;
        times.infer = start.elapsed();
        times.npu = stats.timing.run_time;
        // The pre thread may already be blocked waiting for this slot.
        if free.send(slot).is_err() {
            break;
        }
        if inferred
            .send(Inferred {
                frame,
                outputs,
                times,
            })
            .is_err()
        {
            break;
        }
    }
    Ok(())
}

/// Decode thread: turn model outputs into detections for the main thread.
///
/// `decode_proto` returns `None` both for a detection model, whose boxes it
/// has already decoded, and for a segmentation model it cannot give
/// prototypes for; only the second needs `decode`.
pub fn decode(
    decoder: &Decoder,
    segmentation: bool,
    inferred: &Receiver<Inferred>,
    pool: &Sender<OutputBuffers>,
    events: &Sender<Event>,
) -> Result<()> {
    while let Ok(Inferred {
        frame,
        outputs,
        mut times,
    }) = inferred.recv()
    {
        let start = Instant::now();
        let mut detections = Vec::new();
        let mut masks = Vec::new();
        let protos = decoder.decode_proto(&outputs.refs(), &mut detections)?;
        if protos.is_none() && segmentation {
            decoder.decode(&outputs.refs(), &mut detections, &mut masks)?;
        }
        times.decode = start.elapsed();
        // Decoding is done with the buffers; the npu thread may be waiting.
        let _ = pool.send(outputs);
        let result = Decoded {
            frame,
            detections,
            masks,
            protos,
            times,
        };
        if events.send(Event::Decoded(Box::new(result))).is_err() {
            break;
        }
    }
    Ok(())
}
