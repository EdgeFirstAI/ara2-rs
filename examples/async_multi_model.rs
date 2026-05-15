// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # Async Multi-Model Pipeline Example
//!
//! Demonstrates pipelined inference across two different models loaded
//! on the same ARA-2 endpoint. Two scheduling patterns are benchmarked:
//!
//! ## Dual-Model: Same Image → Both Models
//!
//! Every frame is processed by both model A and model B. The async
//! variant pipelines the current iteration's NPU work with the
//! previous iteration's output reads and next iteration's input fills.
//!
//! ```text
//!   Iter N:   fill A+B ──▶ submit A ──▶ submit B
//!   Iter N+1: wait A, read A; wait B, read B; fill A+B, submit A+B
//! ```
//!
//! ## A/B Alternating: Even Frames → A, Odd Frames → B
//!
//! Frames alternate between models, giving each model a full frame
//! interval of NPU time while the CPU fills the other model's inputs.
//!
//! ```text
//!   Frame 0 (A): fill A, submit A
//!   Frame 1 (B): fill B, submit B; wait A, read A
//!   Frame 2 (A): fill A, submit A; wait B, read B
//! ```
//!
//! ## Usage
//!
//! ```text
//! async_multi_model <model_a.dvm> <model_b.dvm> [iterations]
//! ```

use ara2::{DEFAULT_TIMEOUT_MS, InferRequest, Model, Session};
use edgefirst_hal::tensor::TensorTrait;
use std::{env, path::PathBuf, process, time::Instant};

/// Simulate CPU preprocessing by writing a frame-dependent pattern.
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

/// Extract the model name from a file path for display.
fn model_name(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Print throughput statistics for a benchmark run.
fn print_stats(_label: &str, elapsed: std::time::Duration, iterations: usize) {
    let ms = elapsed.as_secs_f64() * 1000.0;
    let fps = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "  Total: {ms:.2}ms, Avg: {:.2}ms/iter, Throughput: {fps:.1} fps",
        ms / iterations as f64
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "Usage: {} <model_a.dvm> <model_b.dvm> [iterations]",
            args[0]
        );
        process::exit(1);
    }
    let path_a = PathBuf::from(&args[1]);
    let path_b = PathBuf::from(&args[2]);
    let iterations: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    if iterations == 0 {
        eprintln!("Error: iterations must be > 0");
        process::exit(1);
    }

    let name_a = model_name(&path_a);
    let name_b = model_name(&path_b);

    // ── Connect ─────────────────────────────────────────────────────
    let session = Session::create_via_unix_socket(ara2::DEFAULT_SOCKET)
        .expect("Failed to connect to ARA-2 proxy");
    println!("Connected to ARA-2 proxy");

    let endpoints = session.list_endpoints().expect("Failed to list endpoints");
    assert!(!endpoints.is_empty(), "No endpoints available");
    let endpoint = &endpoints[0];

    // ── Load both models ────────────────────────────────────────────
    let mut model_a = endpoint
        .load_model_from_file(&path_a)
        .expect("Failed to load model A");
    model_a
        .allocate_tensors(None)
        .expect("Failed to allocate tensors for A");

    let mut model_b = endpoint
        .load_model_from_file(&path_b)
        .expect("Failed to load model B");
    model_b
        .allocate_tensors(None)
        .expect("Failed to allocate tensors for B");

    println!(
        "Model A: {name_a} ({} in, {} out)",
        model_a.n_inputs(),
        model_a.n_outputs()
    );
    println!(
        "Model B: {name_b} ({} in, {} out)",
        model_b.n_inputs(),
        model_b.n_outputs()
    );

    // Warm-up
    model_a.run().expect("Warm-up A failed");
    model_b.run().expect("Warm-up B failed");
    println!("Warm-up complete\n");

    // ════════════════════════════════════════════════════════════════
    // Benchmark 1: Dual-model — same image through both models
    // ════════════════════════════════════════════════════════════════

    // ── 1a: Synchronous dual-model ──────────────────────────────────
    println!("=== Dual-model sync (fill+run+read A, then B) x {iterations} ===");
    let start = Instant::now();
    for frame in 0..iterations {
        fill_inputs(&mut model_a, frame);
        model_a.run().expect("Sync A failed");
        let _ = read_outputs(&model_a);

        fill_inputs(&mut model_b, frame);
        model_b.run().expect("Sync B failed");
        let _ = read_outputs(&model_b);
    }
    let dual_sync = start.elapsed();
    print_stats("dual-sync", dual_sync, iterations);

    // ── 1b: Async dual-model (submit both, wait both) ───────────────
    println!("\n=== Dual-model async pipelined x {iterations} ===");
    let start = Instant::now();

    let mut inflight_a: Option<InferRequest> = None;
    let mut inflight_b: Option<InferRequest> = None;

    for frame in 0..iterations + 1 {
        // Wait for previous iteration's requests
        if let Some(req) = inflight_a.take() {
            req.wait(DEFAULT_TIMEOUT_MS).expect("Wait A failed");
            let _ = read_outputs(&model_a);
        }
        if let Some(req) = inflight_b.take() {
            req.wait(DEFAULT_TIMEOUT_MS).expect("Wait B failed");
            let _ = read_outputs(&model_b);
        }

        // Submit next iteration (same data to both models)
        if frame < iterations {
            fill_inputs(&mut model_a, frame);
            fill_inputs(&mut model_b, frame);
            inflight_a = Some(model_a.submit().expect("Submit A failed"));
            inflight_b = Some(model_b.submit().expect("Submit B failed"));
        }
    }
    let dual_async = start.elapsed();
    print_stats("dual-async", dual_async, iterations);

    let speedup = dual_sync.as_secs_f64() / dual_async.as_secs_f64();
    println!("  Speedup: {speedup:.2}x over sync");

    // ════════════════════════════════════════════════════════════════
    // Benchmark 2: A/B Alternating — even frames → A, odd → B
    // ════════════════════════════════════════════════════════════════

    // ── 2a: Synchronous alternating ─────────────────────────────────
    println!("\n=== A/B alternating sync (even→A, odd→B) x {iterations} ===");
    let start = Instant::now();
    for frame in 0..iterations {
        if frame % 2 == 0 {
            fill_inputs(&mut model_a, frame);
            model_a.run().expect("Sync A failed");
            let _ = read_outputs(&model_a);
        } else {
            fill_inputs(&mut model_b, frame);
            model_b.run().expect("Sync B failed");
            let _ = read_outputs(&model_b);
        }
    }
    let alt_sync = start.elapsed();
    print_stats("alt-sync", alt_sync, iterations);

    // ── 2b: Async A/B alternating (pipelined) ───────────────────────
    //
    // Uses 2 models as a natural double-buffer: while model A is on
    // the NPU for an even frame, model B's inputs are filled for the
    // next odd frame and vice versa.
    println!("\n=== A/B alternating async pipelined x {iterations} ===");
    let start = Instant::now();

    let models = [&mut model_a, &mut model_b];
    let mut inflight: [Option<InferRequest>; 2] = [None, None];

    for frame in 0..iterations + 2 {
        let slot = frame % 2;

        // Wait for previous request on this model (if any)
        if let Some(req) = inflight[slot].take() {
            req.wait(DEFAULT_TIMEOUT_MS).expect("Wait failed");
            let _ = read_outputs(models[slot]);
        }

        // Submit next frame on this model if within range
        if frame < iterations {
            fill_inputs(models[slot], frame);
            inflight[slot] = Some(models[slot].submit().expect("Submit failed"));
        }
    }
    let alt_async = start.elapsed();
    print_stats("alt-async", alt_async, iterations);

    let speedup = alt_sync.as_secs_f64() / alt_async.as_secs_f64();
    println!("  Speedup: {speedup:.2}x over sync");

    // ── Summary ─────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║                    Summary                          ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!(
        "║ Dual sync:       {:7.1} fps  ({:.2}ms/iter)          ║",
        iterations as f64 / dual_sync.as_secs_f64(),
        dual_sync.as_secs_f64() * 1000.0 / iterations as f64,
    );
    println!(
        "║ Dual async:      {:7.1} fps  ({:.2}ms/iter)          ║",
        iterations as f64 / dual_async.as_secs_f64(),
        dual_async.as_secs_f64() * 1000.0 / iterations as f64,
    );
    println!(
        "║ A/B alt sync:    {:7.1} fps  ({:.2}ms/iter)          ║",
        iterations as f64 / alt_sync.as_secs_f64(),
        alt_sync.as_secs_f64() * 1000.0 / iterations as f64,
    );
    println!(
        "║ A/B alt async:   {:7.1} fps  ({:.2}ms/iter)          ║",
        iterations as f64 / alt_async.as_secs_f64(),
        alt_async.as_secs_f64() * 1000.0 / iterations as f64,
    );
    println!("╚══════════════════════════════════════════════════════╝");

    let count = session
        .inflight_count()
        .expect("Failed to get inflight count");
    println!("In-flight requests remaining: {count}");
}
