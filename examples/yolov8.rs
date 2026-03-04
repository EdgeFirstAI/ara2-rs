// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # YOLOv8 Inference on ARA-2 NPU
//!
//! Demonstrates detection and segmentation inference using ARA-2 with EdgeFirst
//! HAL for preprocessing, decoding, and overlay rendering.
//!
//! ## Usage
//!
//! ```text
//! yolov8 <model.dvm> <image.jpg> [--save] [--threshold 0.25] [--iou 0.45]
//! ```
//!
//! ## Features
//!
//! - Automatic detection vs segmentation from DVM metadata or output shapes
//! - DMA-backed tensors for zero-copy preprocessing
//! - Complete timing breakdown (preprocess, inference, postprocess, render)
//! - Overlay rendering with HAL (bounding boxes + segmentation masks)
//! - COCO class labels as fallback when DVM metadata has no labels

use ara2::{Session, dvm_metadata};
use edgefirst_hal::{
    decoder::{self, DetectBox, Nms, XYWH},
    image::{
        Crop, Flip, ImageProcessor, ImageProcessorTrait as _, PLANAR_RGB, RGBA, Rotation,
        TensorImage, TensorImageRef,
    },
    tensor::{TensorMapTrait as _, TensorMemory, TensorTrait as _},
};
use ndarray::{Array2, Array3};
use std::{path::PathBuf, time::Instant};

// ── Arguments ────────────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    image: PathBuf,
    save: bool,
    threshold: f32,
    iou: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model.dvm> <image.jpg> [--save] [--threshold N] [--iou N]",
            args[0]
        );
        std::process::exit(1);
    }
    let mut threshold = 0.25;
    let mut iou = 0.45;
    let mut save = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--save" => save = true,
            "--threshold" => {
                i += 1;
                threshold = args[i].parse().expect("invalid --threshold value");
            }
            "--iou" => {
                i += 1;
                iou = args[i].parse().expect("invalid --iou value");
            }
            other => eprintln!("Unknown argument: {other}"),
        }
        i += 1;
    }
    Args {
        model: args[1].clone().into(),
        image: args[2].clone().into(),
        save,
        threshold,
        iou,
    }
}

// ── COCO labels (fallback) ───────────────────────────────────────────────────

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

// ── Output identification ────────────────────────────────────────────────────

/// Whether this model performs detection only or detection + segmentation.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Task {
    Detect,
    Segment,
}

/// Identified output tensor indices for split-output YOLOv8 DVM models.
///
/// DVM models compiled with `dvconvert` split YOLO outputs into separate
/// tensors. Detection models have 2 outputs; segmentation models have 4.
#[derive(Debug)]
struct Outputs {
    scores: usize,
    boxes: usize,
    masks: Option<usize>,
    protos: Option<usize>,
    num_classes: usize,
    num_boxes: usize,
}

impl Outputs {
    /// Identify output tensors from model shapes. Detection: 2 outputs (scores
    /// + boxes). Segmentation: 4 outputs (scores + boxes + mask_coeff +
    ///   protos).
    fn identify(model: &ara2::Model, task: Task) -> Result<Self, String> {
        let shapes: Vec<Vec<usize>> = (0..model.n_outputs())
            .map(|i| model.output_shape(i).to_vec())
            .collect();

        for (i, s) in shapes.iter().enumerate() {
            println!("  output[{i}] shape={s:?}");
        }

        match task {
            Task::Detect => Self::detect_heuristic(&shapes),
            Task::Segment => Self::segment_heuristic(&shapes),
        }
    }

    fn detect_heuristic(shapes: &[Vec<usize>]) -> Result<Self, String> {
        if shapes.len() < 2 {
            return Err(format!(
                "detection needs >= 2 outputs, got {}",
                shapes.len()
            ));
        }
        let (mut scores, mut boxes) = (None, None);
        for (i, s) in shapes.iter().enumerate() {
            if s[0] == 4 {
                boxes = Some(i);
            } else if scores.is_none() {
                scores = Some(i);
            }
        }
        let si = scores.ok_or("cannot identify scores output")?;
        let bi = boxes.ok_or("cannot identify boxes output (shape[0]==4)")?;
        Ok(Self {
            scores: si,
            boxes: bi,
            masks: None,
            protos: None,
            num_classes: shapes[si][0],
            num_boxes: shapes[si][1],
        })
    }

    fn segment_heuristic(shapes: &[Vec<usize>]) -> Result<Self, String> {
        if shapes.len() < 4 {
            return Err(format!(
                "segmentation needs 4 outputs, got {}",
                shapes.len()
            ));
        }
        let (mut scores, mut boxes, mut masks, mut protos) = (None, None, None, None);
        for (i, s) in shapes.iter().enumerate() {
            if s.len() == 3 && s[0] == 32 && s[2] != 1 {
                protos = Some(i); // [32, H, W] spatial
            } else if s[0] == 4 {
                boxes = Some(i);
            } else if s[0] == 32 {
                masks = Some(i); // [32, num_boxes]
            } else {
                scores = Some(i);
            }
        }
        let si = scores.ok_or("cannot identify scores")?;
        Ok(Self {
            scores: si,
            boxes: boxes.ok_or("cannot identify boxes")?,
            masks: Some(masks.ok_or("cannot identify mask_coeff")?),
            protos: Some(protos.ok_or("cannot identify protos")?),
            num_classes: shapes[si][0],
            num_boxes: shapes[si][1],
        })
    }
}

// ── Dequantization helpers ───────────────────────────────────────────────────

/// Dequantize an output tensor to f32 using the model's quantization params.
/// Handles both 1-byte (i8/u8) and 2-byte (i16) output tensor types.
fn dequantize_output(model: &ara2::Model, idx: usize) -> Vec<f32> {
    let map = model
        .output_tensor(idx)
        .map()
        .expect("failed to map output");
    let quant = model
        .output_quants(idx)
        .expect("failed to get output quants");
    let bpp = model.output_bpp(idx);
    let slice = map.as_slice();

    if bpp == 2 {
        let i16_slice: &[i16] =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const i16, slice.len() / 2) };
        i16_slice
            .iter()
            .map(|&v| (v as f32 - quant.offset as f32) * quant.qn)
            .collect()
    } else if quant.is_signed {
        let i8_slice: &[i8] =
            unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const i8, slice.len()) };
        i8_slice
            .iter()
            .map(|&v| (v as f32 - quant.offset as f32) * quant.qn)
            .collect()
    } else {
        slice
            .iter()
            .map(|&v| (v as f32 - quant.offset as f32) * quant.qn)
            .collect()
    }
}

/// Dequantize boxes with per-axis normalization to [0,1] range.
/// XYWH rows: x,w are normalized by width; y,h by height.
fn dequantize_boxes(model: &ara2::Model, idx: usize, input_w: f32, input_h: f32) -> Vec<f32> {
    let map = model.output_tensor(idx).map().expect("failed to map boxes");
    let quant = model
        .output_quants(idx)
        .expect("failed to get output quants");
    let bpp = model.output_bpp(idx);
    let slice = map.as_slice();
    let shape = model.output_shape(idx);
    let num_boxes = shape[1];

    let mut out = Vec::with_capacity(4 * num_boxes);
    for row in 0..4 {
        let scale = if row == 0 || row == 2 {
            quant.qn / input_w
        } else {
            quant.qn / input_h
        };
        for col in 0..num_boxes {
            let raw = if bpp == 2 {
                let p = unsafe { *(slice.as_ptr().add((row * num_boxes + col) * 2) as *const i16) };
                p as f32
            } else if quant.is_signed {
                let p = slice[row * num_boxes + col] as i8;
                p as f32
            } else {
                slice[row * num_boxes + col] as f32
            };
            out.push(raw * scale);
        }
    }
    out
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // ── 1. Read DVM metadata ─────────────────────────────────────────────
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
        .map(|t| {
            if t == "segment" {
                Task::Segment
            } else {
                Task::Detect
            }
        })
        .unwrap_or(Task::Detect);

    println!("Task: {task:?}, classes: {}", labels.len());

    // ── 2. Connect to ARA-2 and load model ───────────────────────────────
    let session = Session::create_via_unix_socket("/var/run/ara2.sock")?;
    let endpoint = &session.list_endpoints()?[0];
    println!(
        "Endpoint: {:?}, DRAM free: {:.1} MB",
        endpoint.check_status()?,
        endpoint.dram_statistics()?.free_size as f64 / 1048576.0
    );

    let t0 = Instant::now();
    let mut model = endpoint.load_model_from_file(&args.model)?;
    model.allocate_tensors(Some(TensorMemory::Dma))?;
    println!("Model loaded in {:?}", t0.elapsed());

    let input_shape = model.input_shape(0);
    let (in_c, in_h, in_w) = (input_shape[0], input_shape[1], input_shape[2]);
    println!("Input: {in_c}x{in_h}x{in_w} (CHW)");

    let outputs = Outputs::identify(&model, task)?;
    println!(
        "Identified: scores={}, boxes={}, classes={}, anchors={}",
        outputs.scores, outputs.boxes, outputs.num_classes, outputs.num_boxes
    );

    // ── 3. Load and preprocess image ─────────────────────────────────────
    let image_bytes = std::fs::read(&args.image)?;
    let src = TensorImage::load(&image_bytes, Some(RGBA), Some(TensorMemory::Dma))?;
    let (img_w, img_h) = (src.width(), src.height());
    println!("Image: {img_w}x{img_h}");

    let mut processor = ImageProcessor::new()?;

    let t_pre = Instant::now();
    {
        let mut dst = TensorImageRef::from_borrowed_tensor(model.input_tensor(0), PLANAR_RGB)?;
        processor.convert_ref(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
    }

    // Apply quantization shift if the model expects signed input
    let input_quant = model.input_quants(0);
    if input_quant.is_signed {
        let mut map = model.input_tensor(0).map()?;
        let slice = map.as_mut_slice();
        let signed: &mut [i8] =
            unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut i8, slice.len()) };
        for v in signed.iter_mut() {
            *v = (*v as u8 as i8).wrapping_sub(-128);
        }
    }
    let preprocess_time = t_pre.elapsed();

    // ── 4. Inference ─────────────────────────────────────────────────────
    let t_inf = Instant::now();
    let timing = model.run()?;
    let inference_time = t_inf.elapsed();

    // ── 5. Post-process ──────────────────────────────────────────────────
    let t_post = Instant::now();
    let mut detections: Vec<DetectBox> = Vec::with_capacity(100);

    // Dequantize boxes with per-axis normalization to [0,1]
    let boxes_f32 = dequantize_boxes(&model, outputs.boxes, in_w as f32, in_h as f32);
    let boxes_arr = Array2::from_shape_vec((4, outputs.num_boxes), boxes_f32)?;

    // Dequantize scores
    let scores_f32 = dequantize_output(&model, outputs.scores);
    let scores_arr = Array2::from_shape_vec(
        (outputs.num_classes, outputs.num_boxes),
        scores_f32[..outputs.num_classes * outputs.num_boxes].to_vec(),
    )?;

    let proto_data = match task {
        Task::Detect => {
            decoder::yolo::decode_yolo_split_det_float(
                boxes_arr.view(),
                scores_arr.view(),
                args.threshold,
                args.iou,
                Some(Nms::ClassAgnostic),
                &mut detections,
            );
            None
        }
        Task::Segment => {
            let mi = outputs.masks.unwrap();
            let pi = outputs.protos.unwrap();

            // Dequantize mask coefficients
            let mask_f32 = dequantize_output(&model, mi);
            let mask_shape = model.output_shape(mi);
            let mask_arr = Array2::from_shape_vec(
                (mask_shape[0], outputs.num_boxes),
                mask_f32[..mask_shape[0] * outputs.num_boxes].to_vec(),
            )?;

            // Dequantize protos and permute from (C,H,W) to (H,W,C)
            let proto_f32 = dequantize_output(&model, pi);
            let proto_shape = model.output_shape(pi);
            let protos_chw = Array3::from_shape_vec(
                (proto_shape[0], proto_shape[1], proto_shape[2]),
                proto_f32[..proto_shape[0] * proto_shape[1] * proto_shape[2]].to_vec(),
            )?;
            let protos_hwc = protos_chw.permuted_axes([1, 2, 0]);

            // Decode with proto extraction (returns ProtoData for HAL rendering)
            let pd = decoder::yolo::impl_yolo_split_segdet_float_proto::<XYWH, _, _, _, _>(
                boxes_arr.view(),
                scores_arr.view(),
                mask_arr.view(),
                protos_hwc.view(),
                args.threshold,
                args.iou,
                Some(Nms::ClassAgnostic),
                &mut detections,
            );
            Some(pd)
        }
    };
    let postprocess_time = t_post.elapsed();

    // ── 6. Print results ─────────────────────────────────────────────────
    println!("\n--- Timing ---");
    println!("  Preprocess:  {:?}", preprocess_time);
    println!(
        "  Inference:   {:?} (npu: {:?}, in: {:?}, out: {:?})",
        inference_time, timing.run_time, timing.input_time, timing.output_time
    );
    println!("  Postprocess: {:?}", postprocess_time);
    println!(
        "  Total:       {:?}",
        preprocess_time + inference_time + postprocess_time
    );

    println!("\n--- Detections ({}) ---", detections.len());
    for det in &detections {
        let name = labels.get(det.label).unwrap_or(&"?");
        // Scale normalized [0,1] model-input coords to original image pixels
        let x1 = (det.bbox.xmin * img_w as f32).max(0.0);
        let y1 = (det.bbox.ymin * img_h as f32).max(0.0);
        let x2 = (det.bbox.xmax * img_w as f32).min(img_w as f32);
        let y2 = (det.bbox.ymax * img_h as f32).min(img_h as f32);
        println!(
            "  {name:>12} ({:2}): {:5.1}%  [{:.0}, {:.0}, {:.0}, {:.0}]",
            det.label,
            det.score * 100.0,
            x1,
            y1,
            x2,
            y2
        );
    }

    // ── 7. Save overlay ──────────────────────────────────────────────────
    if args.save {
        let t_render = Instant::now();

        // Load original image as RGBA for overlay rendering
        let mut overlay = TensorImage::load(&image_bytes, Some(RGBA), None)?;

        match proto_data {
            Some(pd) => processor.render_from_protos(&mut overlay, &detections, &pd)?,
            None => processor.render_to_image(&mut overlay, &detections, &[])?,
        }

        let stem = args.image.file_stem().unwrap_or_default().to_string_lossy();
        let out_path = args.image.with_file_name(format!("{stem}_overlay.jpg"));
        overlay.save_jpeg(out_path.to_str().unwrap(), 95)?;

        println!("\n  Render:      {:?}", t_render.elapsed());
        println!("  Saved:       {}", out_path.display());
    }

    Ok(())
}
