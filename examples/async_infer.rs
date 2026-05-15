// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # Async Inference Example
//!
//! Demonstrates the submit/wait async inference API on the ARA-2 NPU.
//!
//! The example loads a DVM model, runs a warm-up inference synchronously,
//! then compares synchronous `run()` against asynchronous `submit()` +
//! `wait()` over multiple iterations.
//!
//! ## Usage
//!
//! ```text
//! async_infer /path/to/model.dvm [iterations]
//! ```

use ara2::{DEFAULT_TIMEOUT_MS, Session};
use std::{env, hint::black_box, path::PathBuf, process, time::Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.dvm> [iterations]", args[0]);
        process::exit(1);
    }
    let model_path = PathBuf::from(&args[1]);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    if iterations == 0 {
        eprintln!("Error: iterations must be > 0");
        process::exit(1);
    }

    // ── Connect ─────────────────────────────────────────────────────
    let session = Session::create_via_unix_socket(ara2::DEFAULT_SOCKET)
        .expect("Failed to connect to ARA-2 proxy");
    println!("Connected to ARA-2 proxy");

    let endpoints = session.list_endpoints().expect("Failed to list endpoints");
    assert!(!endpoints.is_empty(), "No endpoints available");
    let endpoint = &endpoints[0];

    // ── Load model ──────────────────────────────────────────────────
    let mut model = endpoint
        .load_model_from_file(&model_path)
        .expect("Failed to load model");
    model
        .allocate_tensors(None)
        .expect("Failed to allocate tensors");

    println!(
        "Model loaded: {} inputs, {} outputs",
        model.n_inputs(),
        model.n_outputs()
    );

    // ── Warm-up ─────────────────────────────────────────────────────
    let timing = model.run().expect("Warm-up inference failed");
    println!(
        "Warm-up: inference={:.2}ms, input_xfer={:.2}ms, output_xfer={:.2}ms",
        timing.run_time.as_secs_f64() * 1000.0,
        timing.input_time.as_secs_f64() * 1000.0,
        timing.output_time.as_secs_f64() * 1000.0,
    );
    println!();

    // ── Synchronous benchmark ───────────────────────────────────────
    println!("Synchronous run() x {iterations}:");
    let start = Instant::now();
    for _ in 0..iterations {
        model.run().expect("Sync inference failed");
    }
    let sync_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter",
        sync_elapsed.as_secs_f64() * 1000.0,
        sync_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
    );

    // ── Async submit/wait benchmark ─────────────────────────────────
    println!("Async submit() + wait() x {iterations}:");
    let start = Instant::now();
    for _ in 0..iterations {
        let request = model.submit().expect("Submit failed");
        // In a real pipeline, you would do preprocessing of the next
        // frame here while the NPU is busy.
        let _timing = request.wait(DEFAULT_TIMEOUT_MS).expect("Wait failed");
    }
    let async_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter",
        async_elapsed.as_secs_f64() * 1000.0,
        async_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
    );

    // ── Async with simulated overlap ────────────────────────────────
    println!("Async with simulated CPU work overlap x {iterations}:");
    let start = Instant::now();
    for i in 0..iterations {
        let request = model.submit().expect("Submit failed");

        // Simulate preprocessing work that overlaps with NPU inference.
        let cpu_start = Instant::now();
        let mut dummy: u64 = 0;
        for j in 0..100_000u64 {
            dummy = dummy.wrapping_add(j);
        }
        black_box(dummy);
        let cpu_time = cpu_start.elapsed();

        let timing = request.wait(DEFAULT_TIMEOUT_MS).expect("Wait failed");
        if i == 0 {
            println!(
                "  (first iter: cpu_work={:.2}ms, npu={:.2}ms)",
                cpu_time.as_secs_f64() * 1000.0,
                timing.run_time.as_secs_f64() * 1000.0,
            );
        }
    }
    let overlap_elapsed = start.elapsed();
    println!(
        "  Total: {:.2}ms, Avg: {:.2}ms/iter",
        overlap_elapsed.as_secs_f64() * 1000.0,
        overlap_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
    );

    // ── In-flight count ─────────────────────────────────────────────
    let count = session
        .inflight_count()
        .expect("Failed to get inflight count");
    println!("\nIn-flight requests after all done: {count}");
}
