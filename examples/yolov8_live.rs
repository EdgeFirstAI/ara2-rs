// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # YOLOv8 Live Camera Inference on ARA-2 NPU
//!
//! Minimal serial example: captures NV12 frames from a camera via libcamera,
//! runs YOLOv8 detection + instance segmentation on the ARA-2 NPU, and
//! displays results in a native Wayland/EGL window.
//!
//! The entire path from camera sensor to display uses zero-copy DMA-BUF
//! buffers.  The pipeline is intentionally single-threaded and synchronous.
//!
//! ## Pipeline
//!
//! ```text
//! libcamera (NV12 DMA-BUF)
//!   -> HAL import (PlaneDescriptor, cached by buffer index)
//!   -> HAL convert (NV12 -> PlanarRGB letterbox)
//!   -> ARA-2 NPU inference
//!   -> HAL draw_masks (decode + composite -> RGBA canvas)
//!   -> EGL display (DMA-BUF -> EGLImage -> GL texture)
//! ```
//!
//! ## Usage
//!
//! ```text
//! yolov8_live <model.dvm> [--width 1920] [--height 1080]
//!     [--camera-name NAME] [--threshold 0.50] [--iou 0.45]
//!     [--socket /var/run/ara2.sock]
//! ```
//!
//! ## Requirements
//!
//! - libcamera (camera capture)
//! - libwlegl_display.so (Wayland/EGL display, compiled from `wlegl_display.c`)
//! - ARA-2 proxy service running
//! - Wayland compositor (Weston)

use ara2::{Session, dvm_metadata};
use edgefirst_hal::{
    decoder::{
        DecoderBuilder,
        configs::{self, DecoderType, QuantTuple},
    },
    image::{
        ColorMode, Crop, Flip, ImageProcessor, ImageProcessorTrait as _, MaskOverlay, Rect,
        Rotation,
    },
    tensor::{DType, PixelFormat, PlaneDescriptor, TensorDyn, TensorMemory, TensorTrait as _},
};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    request::ReuseFlag,
    stream::StreamRole,
};
use std::ffi::{c_char, c_int, c_void, CString};
extern crate libloading;
use std::os::fd::{AsFd as _, AsRawFd, BorrowedFd};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

// ── Display (runtime-loaded from libwlegl_display.so) ───────────────────────

type FnInit = unsafe extern "C" fn(c_int, c_int, *const c_char) -> *mut c_void;
type FnRenderDmabuf = unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int) -> c_int;
type FnIsOpen = unsafe extern "C" fn(*mut c_void) -> c_int;
type FnDestroy = unsafe extern "C" fn(*mut c_void);

struct EglDisplay {
    ptr: *mut c_void,
    width: c_int,
    height: c_int,
    _lib: libloading::Library,
    render_dmabuf: FnRenderDmabuf,
    is_open_fn: FnIsOpen,
    destroy_fn: FnDestroy,
}

impl EglDisplay {
    fn new(width: usize, height: usize, title: &str) -> Result<Self, String> {
        let lib = unsafe { libloading::Library::new("libwlegl_display.so") }
            .map_err(|e| format!("Cannot load libwlegl_display.so: {e}"))?;

        let init: FnInit = *unsafe { lib.get(b"display_init\0") }
            .map_err(|e| format!("display_init: {e}"))?;
        let render_dmabuf: FnRenderDmabuf = *unsafe { lib.get(b"display_render_dmabuf\0") }
            .map_err(|e| format!("display_render_dmabuf: {e}"))?;
        let is_open_fn: FnIsOpen = *unsafe { lib.get(b"display_is_open\0") }
            .map_err(|e| format!("display_is_open: {e}"))?;
        let destroy_fn: FnDestroy = *unsafe { lib.get(b"display_destroy\0") }
            .map_err(|e| format!("display_destroy: {e}"))?;

        let c_title = CString::new(title).unwrap();
        let ptr = unsafe { init(width as c_int, height as c_int, c_title.as_ptr()) };
        if ptr.is_null() {
            return Err(
                "Failed to create EGL display. Is a Wayland compositor running?".into(),
            );
        }
        Ok(Self {
            ptr,
            width: width as c_int,
            height: height as c_int,
            _lib: lib,
            render_dmabuf,
            is_open_fn,
            destroy_fn,
        })
    }

    fn render_dmabuf(&self, fd: i32) -> bool {
        unsafe { (self.render_dmabuf)(self.ptr, fd, self.width, self.height) == 0 }
    }

    fn is_open(&self) -> bool {
        unsafe { (self.is_open_fn)(self.ptr) != 0 }
    }
}

impl Drop for EglDisplay {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { (self.destroy_fn)(self.ptr) };
        }
    }
}

// ── DMA-BUF tensor cache ────────────────────────────────────────────────────

/// Cache HAL tensors by libcamera buffer index.
///
/// Unlike GStreamer (which recycles fds), libcamera's `FrameBufferAllocator`
/// pre-allocates stable buffers.  We cache by the request cookie (buffer
/// index) so `import_image` is called only once per slot.
struct FrameCache {
    entries: Vec<Option<TensorDyn>>,
}

impl FrameCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: (0..capacity).map(|_| None).collect(),
        }
    }

    fn get_or_import(
        &mut self,
        index: usize,
        processor: &ImageProcessor,
        framebuffer: &FrameBuffer,
        width: usize,
        height: usize,
    ) -> Result<&TensorDyn, Box<dyn std::error::Error>> {
        if self.entries[index].is_none() {
            let planes = framebuffer.planes();
            let luma_fd = planes.get(0).expect("NV12 requires at least one plane").fd();
            // SAFETY: fds come from libcamera FrameBuffer, valid while request is alive.
            // PlaneDescriptor::new() dups the fd so the cached tensor is independent.
            let y_desc =
                PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(luma_fd) })?;

            let uv_desc = if planes.len() >= 2 {
                let chroma = planes.get(1).unwrap();
                let mut p =
                    PlaneDescriptor::new(unsafe { BorrowedFd::borrow_raw(chroma.fd()) })?;
                if let Some(off) = chroma.offset() {
                    if off > 0 {
                        p = p.with_offset(off);
                    }
                }
                Some(p)
            } else {
                None
            };

            let tensor =
                processor.import_image(y_desc, uv_desc, width, height, PixelFormat::Nv12, DType::U8)?;
            self.entries[index] = Some(tensor);
        }
        Ok(self.entries[index].as_ref().unwrap())
    }
}

// ── Arguments ───────────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    threshold: f32,
    iou: f32,
    width: usize,
    height: usize,
    camera_name: Option<String>,
    socket: String,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: {} <model.dvm> [--threshold N] [--iou N] \
             [--width N] [--height N] [--camera-name NAME] [--socket PATH]",
            args[0]
        );
        std::process::exit(1);
    }
    let mut threshold = 0.50;
    let mut iou = 0.45;
    let mut width = 1920;
    let mut height = 1080;
    let mut camera_name = None;
    let mut socket = ara2::DEFAULT_SOCKET.to_string();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            flag @ ("--threshold" | "--iou" | "--width" | "--height" | "--camera-name"
            | "--socket") => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Error: {flag} requires a value");
                    std::process::exit(1);
                }
                match flag {
                    "--threshold" => {
                        threshold = args[i].parse().expect("invalid --threshold value")
                    }
                    "--iou" => iou = args[i].parse().expect("invalid --iou value"),
                    "--width" => width = args[i].parse().expect("invalid --width value"),
                    "--height" => height = args[i].parse().expect("invalid --height value"),
                    "--camera-name" => camera_name = Some(args[i].clone()),
                    "--socket" => socket = args[i].clone(),
                    _ => unreachable!(),
                }
            }
            other => eprintln!("Unknown argument: {other}"),
        }
        i += 1;
    }
    Args {
        model: args[1].clone().into(),
        threshold,
        iou,
        width,
        height,
        camera_name,
        socket,
    }
}

// ── COCO labels (fallback) ──────────────────────────────────────────────────

const COCO: &[&str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

// ── Output identification ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Task {
    Detect,
    Segment,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Normalize an ARA-2 output shape for the HAL decoder.
///
/// ARA-2 reports 3D `[nch, height, width]`.  Strip trailing 1s, prepend batch=1.
fn normalize_shape(raw: [usize; 3]) -> Vec<usize> {
    let mut shape: Vec<usize> = raw.to_vec();
    while shape.len() > 1 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape.insert(0, 1);
    shape
}

fn identify_det_outputs(shapes: &[Vec<usize>]) -> Result<(usize, usize), String> {
    if shapes.len() < 2 {
        return Err(format!("detection needs >= 2 outputs, got {}", shapes.len()));
    }
    let (mut boxes, mut scores) = (None, None);
    for (i, s) in shapes.iter().enumerate() {
        if s.len() == 3 && s[1] == 4 {
            boxes = Some(i);
        } else if s.len() == 3 && scores.is_none() {
            scores = Some(i);
        }
    }
    Ok((
        boxes.ok_or("cannot identify boxes output (shape[1] == 4)")?,
        scores.ok_or("cannot identify scores output")?,
    ))
}

fn identify_seg_outputs(shapes: &[Vec<usize>]) -> Result<(usize, usize, usize, usize), String> {
    if shapes.len() < 4 {
        return Err(format!("segmentation needs 4 outputs, got {}", shapes.len()));
    }
    let (mut scores, mut boxes, mut masks, mut protos) = (None, None, None, None);
    for (i, s) in shapes.iter().enumerate() {
        if s.len() == 4 {
            protos = Some(i);
        } else if s.len() == 3 && s[1] == 4 {
            boxes = Some(i);
        } else if s.len() == 3
            && masks.is_none()
            && protos.map_or(false, |p| shapes[p].get(1) == s.get(1))
        {
            masks = Some(i);
        } else if scores.is_none() {
            scores = Some(i);
        }
    }
    // Retry mask detection when proto tensor appears after mask-coeff tensor.
    if masks.is_none() {
        if let Some(pi) = protos {
            for (i, s) in shapes.iter().enumerate() {
                if s.len() == 3
                    && s[1] != 4
                    && Some(i) != scores
                    && Some(i) != boxes
                    && shapes[pi].get(1) == s.get(1)
                {
                    masks = Some(i);
                    break;
                }
            }
        }
    }
    Ok((
        boxes.ok_or("cannot identify boxes")?,
        scores.ok_or("cannot identify scores")?,
        masks.ok_or("cannot identify mask_coeff")?,
        protos.ok_or("cannot identify protos")?,
    ))
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn compute_letterbox(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Crop {
    let scale = (dst_w as f32 / src_w as f32).min(dst_h as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale) as usize;
    let new_h = (src_h as f32 * scale) as usize;
    Crop::new()
        .with_dst_rect(Some(Rect::new((dst_w - new_w) / 2, (dst_h - new_h) / 2, new_w, new_h)))
        .with_dst_color(Some([114, 114, 114, 255]))
}

// ── NV12 pixel format ───────────────────────────────────────────────────────

const PIXEL_FORMAT_NV12: libcamera::pixel_format::PixelFormat =
    libcamera::pixel_format::PixelFormat::new(u32::from_le_bytes([b'N', b'V', b'1', b'2']), 0);

// ── Main ────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cam_w = args.width;
    let cam_h = args.height;

    // ── 1. Read model metadata ──────────────────────────────────────────
    let dvm_data = std::fs::read(&args.model)?;
    let metadata = dvm_metadata::read_metadata(&dvm_data)?;
    let dvm_labels = dvm_metadata::read_labels(&dvm_data)?;
    let labels: Vec<&str> = if dvm_labels.is_empty() {
        COCO.to_vec()
    } else {
        dvm_labels.iter().map(|s| s.as_str()).collect()
    };

    let task = metadata
        .as_ref()
        .and_then(|m| m.task())
        .map(|t| if t == "segment" { Task::Segment } else { Task::Detect })
        .unwrap_or(Task::Detect);

    if let Some(ref m) = metadata {
        println!("Model: {}", args.model.display());
        println!("Task: {task:?}, Classes: {}", labels.len());
        if let Some(ref comp) = m.compilation {
            if let Some(ref ppa) = comp.ppa {
                println!(
                    "Target: {:?}, IPS: {:.0}, Power: {:.0} mW",
                    comp.target,
                    ppa.ips.unwrap_or(0.0),
                    ppa.power_mw.unwrap_or(0.0)
                );
            }
        }
    }

    // ── 2. Connect to ARA-2 and load model ──────────────────────────────
    let session = Session::create_via_unix_socket(&args.socket)?;
    let endpoints = session.list_endpoints()?;
    if endpoints.is_empty() {
        eprintln!("No ARA-2 endpoints found.  Is ara2-proxy running?");
        std::process::exit(1);
    }
    let endpoint = &endpoints[0];
    let stats = endpoint.dram_statistics()?;
    println!(
        "Endpoint: {:?}, DRAM: {:.0} / {:.0} MB free",
        endpoint.check_status()?,
        stats.free_size as f64 / 1048576.0,
        stats.dram_size as f64 / 1048576.0,
    );

    let mut model = endpoint.load_model_from_file(&args.model)?;
    model.allocate_tensors(Some(TensorMemory::Dma))?;

    let input_shape = model.input_shape(0);
    let (in_c, in_h, in_w) = (input_shape[0], input_shape[1], input_shape[2]);
    let input_dim = in_w.max(in_h) as f32;
    println!("Input: {in_c}x{in_h}x{in_w} (CHW)");

    // ── 3. Build decoder ────────────────────────────────────────────────
    let n_outputs = model.n_outputs();
    let mut shapes = Vec::with_capacity(n_outputs);
    let mut quants = Vec::with_capacity(n_outputs);

    for i in 0..n_outputs {
        let shape = normalize_shape(model.output_shape(i));
        let info = model.output_info(i)?;
        let is_box = shape.len() == 3 && shape[1] == 4;
        let scale = if is_box && input_dim > 1.0 {
            info.quant.qn / input_dim
        } else {
            info.quant.qn
        };
        quants.push((scale, info.quant.offset, info.bpp, info.quant.is_signed));
        shapes.push(shape);
    }

    let task = if task == Task::Detect && shapes.iter().any(|s| s.len() == 4) {
        Task::Segment
    } else {
        task
    };

    let decoder = match task {
        Task::Detect => {
            let (bi, si) = identify_det_outputs(&shapes)?;
            DecoderBuilder::new()
                .with_config_yolo_split_det(
                    configs::Boxes {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[bi].0, quants[bi].1)),
                        shape: shapes[bi].clone(),
                        normalized: Some(true),
                        ..Default::default()
                    },
                    configs::Scores {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[si].0, quants[si].1)),
                        shape: shapes[si].clone(),
                        ..Default::default()
                    },
                )
                .with_score_threshold(args.threshold)
                .with_iou_threshold(args.iou)
                .build()?
        }
        Task::Segment => {
            let (bi, si, mi, pi) = identify_seg_outputs(&shapes)?;
            DecoderBuilder::new()
                .with_config_yolo_split_segdet(
                    configs::Boxes {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[bi].0, quants[bi].1)),
                        shape: shapes[bi].clone(),
                        normalized: Some(true),
                        ..Default::default()
                    },
                    configs::Scores {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[si].0, quants[si].1)),
                        shape: shapes[si].clone(),
                        ..Default::default()
                    },
                    configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[mi].0, quants[mi].1)),
                        shape: shapes[mi].clone(),
                        ..Default::default()
                    },
                    configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        quantization: Some(QuantTuple(quants[pi].0, quants[pi].1)),
                        shape: shapes[pi].clone(),
                        ..Default::default()
                    },
                )
                .with_score_threshold(args.threshold)
                .with_iou_threshold(args.iou)
                .build()?
        }
    };

    // ── 4. Setup HAL processor and model I/O tensors ────────────────────
    let mut processor = ImageProcessor::new()?;

    let input_quant = model.input_quants(0);
    let input_dtype = if input_quant.is_signed { DType::I8 } else { DType::U8 };
    let input_fd = model.input_tensor(0).clone_fd()?;
    let plane = PlaneDescriptor::new(input_fd.as_fd())?;
    let mut model_input =
        processor.import_image(plane, None, in_w, in_h, PixelFormat::PlanarRgb, input_dtype)?;

    let letterbox = compute_letterbox(cam_w, cam_h, in_w, in_h);
    let _letterbox_norm = MaskOverlay::default()
        .with_letterbox_crop(&letterbox, in_w, in_h)
        .letterbox;

    let output_tensors: Vec<TensorDyn> = (0..n_outputs)
        .map(|i| {
            let dtype = match (quants[i].2, quants[i].3) {
                (1, false) => DType::U8,
                (1, true) => DType::I8,
                (2, false) => DType::U16,
                (2, true) => DType::I16,
                _ => panic!("unsupported bpp={} signed={}", quants[i].2, quants[i].3),
            };
            let fd = model.output_tensor(i).clone_fd().expect("clone output fd");
            TensorDyn::from_fd(fd, &shapes[i], dtype, None).expect("wrap output tensor")
        })
        .collect();
    let output_refs: Vec<&TensorDyn> = output_tensors.iter().collect();

    // ── 5. Output canvas ────────────────────────────────────────────────
    let mut canvas =
        processor.create_image(cam_w, cam_h, PixelFormat::Rgba, DType::U8, None)?;
    let canvas_fd = canvas.clone_fd()?;
    let canvas_raw_fd = canvas_fd.as_raw_fd();

    // ── 6. Setup libcamera ──────────────────────────────────────────────
    let mgr = CameraManager::new()?;
    let cameras = mgr.cameras();
    if cameras.is_empty() {
        eprintln!("No cameras found");
        std::process::exit(1);
    }

    let cam_index = if let Some(ref name) = args.camera_name {
        cameras
            .iter()
            .position(|c| c.id() == name)
            .unwrap_or_else(|| {
                eprintln!("Camera '{name}' not found. Available:");
                for c in cameras.iter() {
                    eprintln!("  {}", c.id());
                }
                std::process::exit(1);
            })
    } else {
        0
    };

    let cam = cameras.get(cam_index).unwrap();
    println!("Camera: {}", cam.id());
    let mut cam = cam.acquire().expect("Failed to acquire camera");

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::VideoRecording])
        .expect("Failed to generate camera configuration");
    {
        let mut cfg = cfgs.get_mut(0).unwrap();
        cfg.set_pixel_format(PIXEL_FORMAT_NV12);
        cfg.set_size(libcamera::geometry::Size {
            width: cam_w as u32,
            height: cam_h as u32,
        });
    }
    match cfgs.validate() {
        CameraConfigurationStatus::Valid => {}
        CameraConfigurationStatus::Adjusted => {
            let cfg = cfgs.get(0).unwrap();
            let size = cfg.get_size();
            eprintln!(
                "Camera configuration adjusted: {}x{} {:?}",
                size.width,
                size.height,
                cfg.get_pixel_format()
            );
        }
        CameraConfigurationStatus::Invalid => {
            eprintln!("Invalid camera configuration");
            std::process::exit(1);
        }
    }
    cam.configure(&mut cfgs)
        .expect("Failed to configure camera");

    let stream = cfgs.get(0).unwrap().stream().unwrap();

    let mut alloc = FrameBufferAllocator::new(&cam);
    let buffers = alloc
        .alloc(&stream)
        .expect("Failed to allocate camera buffers");
    let n_buffers = buffers.len();
    println!("Allocated {n_buffers} camera buffers");

    let reqs: Vec<_> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect();

    let (tx, rx) = mpsc::channel();
    cam.on_request_completed(move |req| {
        let _ = tx.send(req);
    });

    // ── 7. Setup display ────────────────────────────────────────────────
    let display = EglDisplay::new(cam_w, cam_h, "ARA-2 YOLOv8 Live")?;

    // ── 8. Start camera and warmup ──────────────────────────────────────
    cam.start(None).expect("Failed to start camera");
    for req in reqs {
        cam.queue_request(req).map_err(|(_, e)| e)?;
    }

    let mut frame_cache = FrameCache::new(n_buffers);

    // Warmup: process first frame end-to-end
    let mut req = rx
        .recv_timeout(Duration::from_secs(5))
        .expect("No frames from camera");
    {
        let idx = req.cookie() as usize;
        let buf: &FrameBuffer = req.buffer(&stream).unwrap();
        let src = frame_cache.get_or_import(idx, &processor, buf, cam_w, cam_h)?;
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e)?;

        processor.convert(src, &mut model_input, Rotation::None, Flip::None, letterbox.clone())?;
        model.run()?;

        let overlay = MaskOverlay::default()
            .with_background(src)
            .with_letterbox_crop(&letterbox, in_w, in_h)
            .with_color_mode(ColorMode::Instance);
        processor.draw_masks(&decoder, &output_refs, &mut canvas, overlay)?;

        if !display.render_dmabuf(canvas_raw_fd) {
            return Err(
                "DMA-BUF display failed. This example requires EGL_EXT_image_dma_buf_import.".into(),
            );
        }
    }

    println!("\nWarmup complete");
    println!("Capturing {cam_w}x{cam_h} -- press Ctrl+C to stop\n");

    // ── 9. Live inference loop ──────────────────────────────────────────
    let mut frame_count: u64 = 0;
    let t_start = Instant::now();

    let mut t_pull = 0.0_f64;
    let mut t_import = 0.0_f64;
    let mut t_convert = 0.0_f64;
    let mut t_npu = 0.0_f64;
    let mut t_draw = 0.0_f64;
    let mut t_display = 0.0_f64;
    let mut total_dropped: u64 = 0;

    while display.is_open() {
        let t0 = Instant::now();

        // Pull the latest frame, dropping stale queued frames.
        let mut req = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("Camera timeout");
                break;
            }
        };
        let mut dropped = 0u64;
        while let Ok(newer) = rx.try_recv() {
            req.reuse(ReuseFlag::REUSE_BUFFERS);
            cam.queue_request(req).map_err(|(_, e)| e)?;
            req = newer;
            dropped += 1;
        }
        let t1 = Instant::now();

        let idx = req.cookie() as usize;
        let buf: &FrameBuffer = req.buffer(&stream).unwrap();
        let src = frame_cache.get_or_import(idx, &processor, buf, cam_w, cam_h)?;

        // Requeue immediately — the cached tensor owns a dup'd fd,
        // so the libcamera buffer can be reused by the camera while
        // we process.  This prevents buffer starvation during ISP
        // adjustments (AWB/AE) that hold buffers internally.
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e)?;
        let t2 = Instant::now();

        processor.convert(src, &mut model_input, Rotation::None, Flip::None, letterbox.clone())?;
        let t3 = Instant::now();

        model.run()?;
        let t4 = Instant::now();

        let overlay = MaskOverlay::default()
            .with_background(src)
            .with_letterbox_crop(&letterbox, in_w, in_h)
            .with_color_mode(ColorMode::Instance);
        let detections = processor.draw_masks(&decoder, &output_refs, &mut canvas, overlay)?;
        let t5 = Instant::now();

        display.render_dmabuf(canvas_raw_fd);
        let t6 = Instant::now();

        t_pull += (t1 - t0).as_secs_f64();
        t_import += (t2 - t1).as_secs_f64();
        t_convert += (t3 - t2).as_secs_f64();
        t_npu += (t4 - t3).as_secs_f64();
        t_draw += (t5 - t4).as_secs_f64();
        t_display += (t6 - t5).as_secs_f64();
        total_dropped += dropped;

        frame_count += 1;
        if frame_count % 30 == 0 {
            let elapsed = (Instant::now() - t_start).as_secs_f64();
            let fps = frame_count as f64 / elapsed;
            let n = 30.0;
            eprint!(
                "\r  FPS: {:5.1}  \
                 pull:{:5.1} imp:{:4.1} cvt:{:4.1} npu:{:5.1} \
                 draw:{:5.1} disp:{:4.1} \
                 tot:{:5.1}ms drop:{} det:{} f:{}",
                fps,
                t_pull / n * 1000.0,
                t_import / n * 1000.0,
                t_convert / n * 1000.0,
                t_npu / n * 1000.0,
                t_draw / n * 1000.0,
                t_display / n * 1000.0,
                (t_pull + t_import + t_convert + t_npu + t_draw + t_display) / n * 1000.0,
                total_dropped,
                detections.len(),
                frame_count,
            );
            t_pull = 0.0;
            t_import = 0.0;
            t_convert = 0.0;
            t_npu = 0.0;
            t_draw = 0.0;
            t_display = 0.0;
        }
    }

    // ── 10. Shutdown ────────────────────────────────────────────────────
    eprintln!();
    cam.stop().ok();
    if frame_count > 0 {
        let elapsed = (Instant::now() - t_start).as_secs_f64();
        println!(
            "Processed {} frames in {:.1}s ({:.1} FPS average)",
            frame_count,
            elapsed,
            frame_count as f64 / elapsed,
        );
    }

    Ok(())
}
