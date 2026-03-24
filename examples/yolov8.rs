// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # YOLOv8 Inference on ARA-2 NPU
//!
//! Demonstrates detection and segmentation inference using ARA-2 with EdgeFirst
//! HAL for preprocessing, decoding, and overlay rendering.
//!
//! Uses the HAL `Decoder` API which handles dequantization and NMS internally,
//! matching the pattern used by the Maivin `model` crate.
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
//! - HAL Decoder with automatic dequantization of quantized outputs
//! - Complete timing breakdown (preprocess, inference, postprocess, render)
//! - Overlay rendering with HAL (bounding boxes + segmentation masks)
//! - COCO class labels as fallback when DVM metadata has no labels

use ara2::{Session, dvm_metadata};
use edgefirst_hal::{
    decoder::{
        ArrayViewDQuantized, DecoderBuilder, DetectBox, ProtoData,
        configs::{self, DecoderType, QuantTuple},
    },
    image::{
        Crop, Flip, ImageProcessor, ImageProcessorTrait as _, Rotation, load_image, save_jpeg,
    },
    tensor::{DType, PixelFormat, TensorMapTrait as _, TensorMemory, TensorTrait as _},
};
use std::os::fd::AsFd as _;
use ndarray::IxDyn;
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

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Normalize an ARA-2 output shape for the HAL decoder.
///
/// ARA-2's FFI struct always reports 3 dimensions `[nch, height, width]`.
/// For logically 2D outputs (e.g. scores `[80, 8400]`), the struct pads
/// width=1, producing `[80, 8400, 1]`. We strip that trailing 1 to recover
/// the true rank.
///
/// The HAL decoder expects an ONNX-style leading batch dimension (it slices
/// `[0, .., ..]` internally to remove it), so we prepend batch=1.
///
/// Examples:
///   `[80, 8400, 1]` → `[1, 80, 8400]`  (scores)
///   `[4, 8400, 1]`  → `[1, 4, 8400]`   (boxes)
///   `[32, 160, 160]` → `[1, 32, 160, 160]` (protos — no trailing 1 to strip)
fn normalize_shape(raw: [usize; 3]) -> Vec<usize> {
    let mut shape: Vec<usize> = raw.to_vec();
    while shape.len() > 1 && shape.last() == Some(&1) {
        shape.pop();
    }
    shape.insert(0, 1);
    shape
}

/// Build an `ArrayViewDQuantized` from a raw output tensor, using bpp and
/// signedness to select the correct integer type.
fn output_to_quantized_view<'a>(
    bytes: &'a [u8],
    shape: &[usize],
    bpp: usize,
    is_signed: bool,
) -> Result<ArrayViewDQuantized<'a>, Box<dyn std::error::Error>> {
    let ix = IxDyn(shape);
    match (bpp, is_signed) {
        (1, true) => {
            let data: &[i8] =
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
            Ok(ndarray::ArrayView::from_shape(ix, data)?.into())
        }
        (1, false) => Ok(ndarray::ArrayView::from_shape(ix, bytes)?.into()),
        (2, true) => {
            let data: &[i16] = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const i16,
                    bytes.len() / 2,
                )
            };
            Ok(ndarray::ArrayView::from_shape(ix, data)?.into())
        }
        (2, false) => {
            let data: &[u16] = unsafe {
                std::slice::from_raw_parts(
                    bytes.as_ptr() as *const u16,
                    bytes.len() / 2,
                )
            };
            Ok(ndarray::ArrayView::from_shape(ix, data)?.into())
        }
        _ => Err(format!("unsupported bpp={bpp} signed={is_signed}").into()),
    }
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

    // ── 3. Build HAL Decoder from output tensor metadata ─────────────────
    //
    // Collect normalized shapes and quantization for each output. For box
    // outputs (shape contains 4), divide the quantization scale by input_dim
    // so the decoder produces normalized [0,1] coordinates.
    let input_dim = in_w.max(in_h) as f32;
    let n_outputs = model.n_outputs();

    let mut shapes = Vec::with_capacity(n_outputs);
    let mut quants = Vec::with_capacity(n_outputs);

    for i in 0..n_outputs {
        let raw_shape = model.output_shape(i);
        let shape = normalize_shape(raw_shape);
        let info = model.output_info(i)?;
        let is_box_output = shape.contains(&4);
        let scale = if is_box_output && input_dim > 1.0 {
            info.quant.qn / input_dim
        } else {
            info.quant.qn
        };
        println!(
            "  output[{i}] raw_shape={raw_shape:?} shape={shape:?} bpp={} signed={} qn={} offset={} (adj_scale={scale})",
            info.bpp, info.quant.is_signed, info.quant.qn, info.quant.offset
        );
        quants.push((scale, info.quant.offset, info.bpp, info.quant.is_signed));
        shapes.push(shape);
    }

    // ── 3a. Load and preprocess image (before decoder build for diagnostics) ──
    let image_bytes = std::fs::read(&args.image)?;
    let src = load_image(&image_bytes, Some(PixelFormat::Rgba), Some(TensorMemory::Dma))?;
    let (img_w, img_h) = (src.width().unwrap(), src.height().unwrap());
    println!("Image: {img_w}x{img_h}");

    let mut processor = ImageProcessor::new()?;

    let t_pre = Instant::now();
    {
        let input = model.input_tensor(0);
        let input_fd = input.clone_fd()?;
        let mut dst = processor.create_image_from_fd(
            input_fd.as_fd(),
            in_w,
            in_h,
            PixelFormat::PlanarRgb,
            DType::U8,
        )?;
        processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::default())?;
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

    // ── 3b. Inference (run before decoder build so we can analyze raw buffers) ──
    let t_inf = Instant::now();
    let timing = model.run()?;
    let inference_time = t_inf.elapsed();

    // ── 3c. Post-inference buffer analysis ──
    {
        let maps_post: Vec<_> = (0..n_outputs)
            .map(|i| model.output_tensor(i).map().expect("failed to map output"))
            .collect();
        println!("\n--- Post-Inference Buffer Analysis ---");
        for i in 0..n_outputs {
            let bytes = maps_post[i].as_slice();
            let (bpp, is_signed) = (quants[i].2, quants[i].3);
            if bpp == 1 && is_signed {
                let data: &[i8] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, bytes.len()) };
                let mut hist = std::collections::HashMap::new();
                for &v in data.iter() { *hist.entry(v).or_insert(0u64) += 1; }
                println!("  output[{i}]: shape={:?} unique_values={} total={}", shapes[i], hist.len(), data.len());
                if hist.len() <= 10 {
                    let mut sorted: Vec<_> = hist.iter().collect();
                    sorted.sort_by_key(|&(&v, _)| v);
                    for &(&v, &c) in &sorted { println!("    q={v}: {c} ({:.1}%)", 100.0 * c as f64 / data.len() as f64); }
                }
            } else if bpp == 1 && !is_signed {
                let mut hist = std::collections::HashMap::new();
                for &v in bytes.iter() { *hist.entry(v).or_insert(0u64) += 1; }
                println!("  output[{i}]: shape={:?} unique_values={} total={}", shapes[i], hist.len(), bytes.len());
            } else if bpp == 2 && is_signed {
                let data: &[i16] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i16, bytes.len()/2) };
                let mut hist = std::collections::HashMap::new();
                for &v in data.iter() { *hist.entry(v).or_insert(0u64) += 1; }
                println!("  output[{i}]: shape={:?} unique_values={} total={}", shapes[i], hist.len(), data.len());
            }
        }
    }

    // ── 3d. Build HAL Decoder from output tensor metadata ────────────────
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

    println!("Decoder: {:?}", decoder.model_type());

    // ── 6. Post-process via HAL Decoder ──────────────────────────────────
    let t_post = Instant::now();

    // Map all output tensors and build quantized array views
    let maps: Vec<_> = (0..n_outputs)
        .map(|i| model.output_tensor(i).map().expect("failed to map output"))
        .collect();

    let views: Vec<ArrayViewDQuantized> = (0..n_outputs)
        .map(|i| {
            output_to_quantized_view(
                maps[i].as_slice(),
                &shapes[i],
                quants[i].2,  // bpp
                quants[i].3,  // is_signed
            )
            .expect("failed to create quantized view")
        })
        .collect();

    // Decode with both paths for comparison
    let mut detections: Vec<DetectBox> = Vec::with_capacity(100);
    let mut masks: Vec<edgefirst_hal::decoder::Segmentation> = Vec::with_capacity(100);
    decoder.decode_quantized(&views, &mut detections, &mut masks)?;
    let postprocess_time = t_post.elapsed();

    // Also get ProtoData for draw_masks_proto path
    let mut det2: Vec<DetectBox> = Vec::with_capacity(100);
    let proto_data: Option<ProtoData> = decoder.decode_quantized_proto(&views, &mut det2)?;

    // Debug: inspect quantization parameters for mask/proto outputs
    if task == Task::Segment {
    println!("\n--- Quantization Debug ---");
    {
        let (_, _, mi, pi) = identify_seg_outputs(&shapes).unwrap();
        println!("  mask_coeff output[{mi}]: scale={}, offset={}, bpp={}, signed={}",
            quants[mi].0, quants[mi].1, quants[mi].2, quants[mi].3);
        println!("  protos output[{pi}]: scale={}, offset={}, bpp={}, signed={}",
            quants[pi].0, quants[pi].1, quants[pi].2, quants[pi].3);
        let combined_scale = quants[mi].0 * quants[pi].0;
        println!("  combined_scale = {} * {} = {}", quants[mi].0, quants[pi].0, combined_scale);

        // Sample raw data from mask coefficient and proto tensors
        let mask_bytes = maps[mi].as_slice();
        let proto_bytes = maps[pi].as_slice();
        print!("  mask_coeff first 32 values: [");
        if quants[mi].3 { // signed
            let data: &[i8] = unsafe { std::slice::from_raw_parts(mask_bytes.as_ptr() as *const i8, mask_bytes.len()) };
            for v in data.iter().take(32) { print!("{v}, "); }
        } else {
            for v in mask_bytes.iter().take(32) { print!("{v}, "); }
        }
        println!("]");

        // Comprehensive proto buffer analysis
        println!("  proto buffer: {} bytes, bpp={}", proto_bytes.len(), quants[pi].2);
        if quants[pi].3 { // signed
            let data: &[i8] = unsafe { std::slice::from_raw_parts(proto_bytes.as_ptr() as *const i8, proto_bytes.len()) };
            // Unique value histogram
            let mut hist = std::collections::HashMap::new();
            for &v in data.iter() { *hist.entry(v).or_insert(0u64) += 1; }
            let mut sorted_vals: Vec<_> = hist.iter().collect();
            sorted_vals.sort_by_key(|&(&v, _)| v);
            println!("  proto unique values: {} (total elements: {})", hist.len(), data.len());
            for &(&val, &count) in sorted_vals.iter().take(20) {
                let pct = 100.0 * count as f64 / data.len() as f64;
                let dequant = (val as f64 - quants[pi].1 as f64) * quants[pi].0 as f64;
                println!("    q={val:>5}: {count:>8} ({pct:>5.1}%)  dequant={dequant:.4}");
            }
            if sorted_vals.len() > 20 { println!("    ... and {} more unique values", sorted_vals.len() - 20); }
            // Sample from different regions
            let n = data.len();
            let proto_shape = &shapes[pi]; // [1, 32, 160, 160]
            let ch = proto_shape.get(1).copied().unwrap_or(32);
            let hw = proto_shape.get(2).copied().unwrap_or(160) * proto_shape.get(3).copied().unwrap_or(160);
            println!("  proto layout: channels={ch} spatial={hw}");
            // Print channel 0 row 0, channel 0 middle row, channel 16 row 0
            let offsets = [
                ("ch0 row0", 0usize),
                ("ch0 mid", hw / 2),
                ("ch16 row0", 16 * hw),
                ("ch31 last", n.saturating_sub(160)),
            ];
            for (label, off) in offsets {
                if off + 16 <= n {
                    print!("  {label} @{off}: [");
                    for v in data[off..off+16].iter() { print!("{v}, "); }
                    println!("]");
                }
            }
        } else {
            println!("  (unsigned proto analysis - not expected)");
        }

        // Simulate one dot product for first detection's mask coeff
        // to see what the matmul result looks like
        if !detections.is_empty() {
            // The mask coeff tensor shape is [1, 32, 8400] in our normalized shape
            // After find_outputs_with_shape_quantized and swap_axes, it becomes [32, 8400]
            // Then reversed_axes makes it [8400, 32]. masks.row(box_idx) gives [32].
            println!("  (Manual matmul simulation would require box indices - see decoder internals)");
        }
    }

    // Debug: inspect masks
    println!("\n--- Masks ---");
    println!("  detections: {}, masks: {}", detections.len(), masks.len());
    for (i, mask) in masks.iter().enumerate() {
        let seg = &mask.segmentation;
        let nonzero = seg.iter().filter(|&&v| v > 0).count();
        // Also show value distribution
        let min = seg.iter().min().copied().unwrap_or(0);
        let max = seg.iter().max().copied().unwrap_or(0);
        let sum: u64 = seg.iter().map(|&v| v as u64).sum();
        let mean = if !seg.is_empty() { sum as f64 / seg.len() as f64 } else { 0.0 };
        println!(
            "  mask[{i}]: shape={:?} bbox=[{:.3},{:.3},{:.3},{:.3}] nonzero={}/{} min={} max={} mean={:.1}",
            seg.shape(), mask.xmin, mask.ymin, mask.xmax, mask.ymax, nonzero, seg.len(), min, max, mean
        );
    }
    if let Some(ref pd) = proto_data {
        let n_coeffs = pd.mask_coefficients.len();
        let protos_info = match &pd.protos {
            edgefirst_hal::decoder::ProtoTensor::Quantized { protos, quantization } =>
                format!("Quantized {:?} scale={} zp={}", protos.shape(), quantization.scale, quantization.zero_point),
            edgefirst_hal::decoder::ProtoTensor::Float(arr) =>
                format!("Float {:?}", arr.shape()),
        };
        println!("  proto: {protos_info}, coefficients: {n_coeffs}");
        // Print first detection's mask coefficients (dequantized)
        if let Some(coeffs) = pd.mask_coefficients.first() {
            print!("  det[0] coeffs (dequant): [");
            for v in coeffs.iter().take(8) { print!("{v:.4}, "); }
            println!("...]");
        }
    }
    } // task == Task::Segment

    // ── 7. Print results ─────────────────────────────────────────────────
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

    // ── 8. Save overlays ─────────────────────────────────────────────────
    if args.save {
        let stem = args.image.file_stem().unwrap_or_default().to_string_lossy();

        // Save with materialized masks (decode_quantized path)
        {
            let t_render = Instant::now();
            let mut overlay = load_image(&image_bytes, Some(PixelFormat::Rgba), None)?;
            processor.draw_masks(&mut overlay, &detections, &masks)?;
            let out_path = args.image.with_file_name(format!("{stem}_masks.jpg"));
            save_jpeg(&overlay, out_path.to_str().unwrap(), 95)?;
            println!("\n  Render (masks):  {:?}", t_render.elapsed());
            println!("  Saved:           {}", out_path.display());
        }

        // Save with proto path (decode_quantized_proto path)
        if let Some(pd) = proto_data {
            let t_render = Instant::now();
            let mut overlay = load_image(&image_bytes, Some(PixelFormat::Rgba), None)?;
            processor.draw_masks_proto(&mut overlay, &detections, &pd)?;
            let out_path = args.image.with_file_name(format!("{stem}_proto.jpg"));
            save_jpeg(&overlay, out_path.to_str().unwrap(), 95)?;
            println!("  Render (proto):  {:?}", t_render.elapsed());
            println!("  Saved:           {}", out_path.display());
        }
    }

    Ok(())
}

// ── Output identification helpers ────────────────────────────────────────────

/// Identify detection output indices: boxes [1,4,N] and scores [1,C,N].
fn identify_det_outputs(shapes: &[Vec<usize>]) -> Result<(usize, usize), String> {
    if shapes.len() < 2 {
        return Err(format!(
            "detection needs >= 2 outputs, got {}",
            shapes.len()
        ));
    }
    let (mut boxes, mut scores) = (None, None);
    for (i, s) in shapes.iter().enumerate() {
        if s.contains(&4) {
            boxes = Some(i);
        } else if scores.is_none() {
            scores = Some(i);
        }
    }
    Ok((
        boxes.ok_or("cannot identify boxes output (shape contains 4)")?,
        scores.ok_or("cannot identify scores output")?,
    ))
}

/// Identify segmentation output indices: boxes, scores, mask_coeff, protos.
fn identify_seg_outputs(shapes: &[Vec<usize>]) -> Result<(usize, usize, usize, usize), String> {
    if shapes.len() < 4 {
        return Err(format!(
            "segmentation needs 4 outputs, got {}",
            shapes.len()
        ));
    }
    let (mut scores, mut boxes, mut masks, mut protos) = (None, None, None, None);
    for (i, s) in shapes.iter().enumerate() {
        if s.len() == 4 && s.contains(&32) {
            protos = Some(i); // [1, 32, H, W] or [1, H, W, 32]
        } else if s.contains(&4) {
            boxes = Some(i);
        } else if s.contains(&32) {
            masks = Some(i); // [1, 32, num_boxes]
        } else {
            scores = Some(i);
        }
    }
    Ok((
        boxes.ok_or("cannot identify boxes")?,
        scores.ok_or("cannot identify scores")?,
        masks.ok_or("cannot identify mask_coeff")?,
        protos.ok_or("cannot identify protos")?,
    ))
}
