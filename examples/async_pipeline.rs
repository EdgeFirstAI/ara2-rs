// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # Async Pipeline Example
//!
//! Demonstrates pipelined inference with a circular buffer of DMA-BUF
//! tensor sets. Each "slot" in the ring is a separate model instance with
//! its own registered DMA-BUF tensors. While the NPU executes slot N, the
//! CPU fills slot N+1 (and reads results from slot N-1), achieving full
//! CPU/NPU overlap.
//!
//! ## Pipeline Stages
//!
//! ```text
//!   ┌──────────┐   ┌──────────┐   ┌──────────┐
//!   │ Fill (CPU)│──▶│Infer(NPU)│──▶│Read (CPU)│
//!   └──────────┘   └──────────┘   └──────────┘
//!        slot[i]      slot[i-1]      slot[i-2]
//! ```
//!
//! ## Usage
//!
//! ```text
//! async_pipeline /path/to/model.dvm [iterations] [depth]
//! ```
//!
//! - `iterations`: total inferences to run (default: 100)
//! - `depth`: ring buffer depth / number of model slots (default: 2)

use ara2::{DEFAULT_TIMEOUT_MS, InferRequest, Model, Session};
use edgefirst_hal::tensor::TensorTrait;
use std::{env, path::PathBuf, process, time::Instant};

/// Simulate CPU preprocessing by writing a pattern into all input tensors.
fn fill_inputs(model: &mut Model, frame: usize) {
    let val = (frame % 256) as u8;
    for i in 0..model.n_inputs() {
        let tensor = model.input_tensor(i);
        let mut map = tensor.map().expect("Failed to map input tensor");
        map.fill(val);
    }
}

/// Simulate CPU postprocessing by reading the first byte of each output.
fn read_outputs(model: &Model) -> u8 {
    let mut checksum: u8 = 0;
    for i in 0..model.n_outputs() {
        let tensor = model.output_tensor(i);
        let map = tensor.map().expect("Failed to map output tensor");
        checksum = checksum.wrapping_add(map[0]);
    }
    checksum
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.dvm> [iterations] [depth]", args[0]);
        process::exit(1);
    }
    let model_path = PathBuf::from(&args[1]);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    if iterations == 0 {
        eprintln!("Error: iterations must be > 0");
        process::exit(1);
    }
    let depth: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    if depth == 0 {
        eprintln!("Error: depth must be > 0");
        process::exit(1);
    }

    // ── Connect ─────────────────────────────────────────────────────
    let session = Session::create_via_unix_socket(ara2::DEFAULT_SOCKET)
        .expect("Failed to connect to ARA-2 proxy");
    println!("Connected to ARA-2 proxy");

    let endpoints = session.list_endpoints().expect("Failed to list endpoints");
    assert!(!endpoints.is_empty(), "No endpoints available");
    let endpoint = &endpoints[0];

    // ── Create circular buffer of model slots ───────────────────────
    // Each slot is a separate Model instance with its own DMA-BUF
    // tensors. This allows the NPU to execute on one set of buffers
    // while the CPU fills/reads another.
    let mut slots: Vec<Model> = Vec::with_capacity(depth);
    for _ in 0..depth {
        let mut model = endpoint
            .load_model_from_file(&model_path)
            .expect("Failed to load model");
        model
            .allocate_tensors(None)
            .expect("Failed to allocate tensors");
        slots.push(model);
    }

    println!(
        "Pipeline: {} slots x ({} inputs, {} outputs) with DMA-BUF tensors",
        depth,
        slots[0].n_inputs(),
        slots[0].n_outputs()
    );

    // Warm-up each slot
    for slot in &mut slots {
        slot.run().expect("Warm-up failed");
    }
    println!("Warm-up complete\n");

    // ── Baseline: synchronous ───────────────────────────────────────
    println!("=== Synchronous baseline (fill → run → read) x {iterations} ===");
    let start = Instant::now();
    for frame in 0..iterations {
        let slot = &mut slots[0];
        fill_inputs(slot, frame);
        slot.run().expect("Sync inference failed");
        let _ = read_outputs(slot);
    }
    let sync_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter, Throughput: {:.1} fps",
        sync_elapsed.as_secs_f64() * 1000.0,
        sync_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations as f64 / sync_elapsed.as_secs_f64(),
    );

    // ── Pipelined: circular buffer ──────────────────────────────────
    println!("\n=== Pipelined async (depth={depth}) x {iterations} ===");
    let start = Instant::now();

    // In-flight request ring: one Option<InferRequest> per slot.
    let mut inflight: Vec<Option<InferRequest>> = (0..depth).map(|_| None).collect();

    for frame in 0..iterations + depth {
        let slot_idx = frame % depth;

        // Wait for the previous request on this slot (if any).
        if let Some(req) = inflight[slot_idx].take() {
            req.wait(DEFAULT_TIMEOUT_MS).expect("Wait failed");
            // Read outputs from the slot that just completed.
            let _ = read_outputs(&slots[slot_idx]);
        }

        // If there are still frames to submit, fill and submit.
        if frame < iterations {
            fill_inputs(&mut slots[slot_idx], frame);
            let req = slots[slot_idx].submit().expect("Submit failed");
            inflight[slot_idx] = Some(req);
        }
    }

    let pipeline_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter, Throughput: {:.1} fps",
        pipeline_elapsed.as_secs_f64() * 1000.0,
        pipeline_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
        iterations as f64 / pipeline_elapsed.as_secs_f64(),
    );

    // ── Summary ─────────────────────────────────────────────────────
    let speedup = sync_elapsed.as_secs_f64() / pipeline_elapsed.as_secs_f64();
    println!(
        "\nSpeedup: {:.2}x (sync {:.2}ms vs pipeline {:.2}ms)",
        speedup,
        sync_elapsed.as_secs_f64() * 1000.0,
        pipeline_elapsed.as_secs_f64() * 1000.0,
    );

    let count = session
        .inflight_count()
        .expect("Failed to get inflight count");
    println!("In-flight requests remaining: {count}");
}
