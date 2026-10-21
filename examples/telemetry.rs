// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! # Telemetry Example
//!
//! Prints everything the ARA-2 runtime reports about the endpoint, a loaded
//! model and one inference, then measures what sampling endpoint statistics
//! costs while inference runs.
//!
//! Fields the loaded DVAPI version does not provide print as `None` in
//! the debug output or as `n/a`, and calls it lacks print a message saying so.
//!
//! ## Usage
//!
//! ```text
//! telemetry /path/to/model.dvm [iterations] [sample_ms]
//! ```

use std::{
    env, process,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use ara2::{Error, Model, Session};

fn main() -> Result<(), Error> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.dvm> [iterations] [sample_ms]", args[0]);
        process::exit(1);
    }
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let sample_ms: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    if iterations == 0 {
        eprintln!("Error: iterations must be > 0");
        process::exit(1);
    }

    let session = Session::connect()?;
    println!(
        "DVAPI {} via {:?}, fd passing: {}",
        session.abi(),
        session.socket_type(),
        session.supports_fd_passing()
    );
    for v in session.versions()? {
        println!("  running {:<14} {}", v.component.to_string(), v.version);
    }

    let endpoint = session
        .list_endpoints()?
        .into_iter()
        .next()
        .expect("no endpoints available");
    println!("\n{:#?}", endpoint.info()?);
    println!("\n{:#?}", endpoint.statistics()?);
    match endpoint.is_busy() {
        Ok(busy) => println!("\nbusy: {busy}"),
        Err(Error::Unsupported { feature, abi }) => {
            println!("\nbusy: n/a ({feature} is not provided by DVAPI {abi})")
        }
        Err(e) => return Err(e),
    }

    let mut model = endpoint.load_model_from_file(model_path(&args).as_ref())?;
    model.allocate_tensors(None)?;
    println!(
        "\nmodel {:?} type {:?} version {}",
        model.name(),
        model.model_type(),
        model.version()
    );

    let stats = model.run()?;
    println!("\n{stats:#?}");
    if stats.inference_start.is_none() {
        println!("(timestamps are n/a: this runtime does not stamp on CLOCK_MONOTONIC)");
    }

    let baseline = run_for(&mut model, iterations)?;
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let (endpoint, stop) = (endpoint.clone(), Arc::clone(&stop));
        thread::spawn(move || {
            let mut calls = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                let t = Instant::now();
                let ok = endpoint.statistics().is_ok();
                calls.push(t.elapsed());
                assert!(ok, "statistics() failed while sampling");
                thread::sleep(Duration::from_millis(sample_ms));
            }
            calls
        })
    };
    let sampled = run_for(&mut model, iterations)?;
    stop.store(true, Ordering::Relaxed);
    let calls = sampler.join().expect("sampler thread panicked");
    let mean = calls.iter().sum::<Duration>() / calls.len().max(1) as u32;
    println!(
        "\ninference: {baseline:.1} fps alone, {sampled:.1} fps while sampling every {sample_ms} ms"
    );
    println!(
        "statistics(): {} calls, mean {mean:?}, max {:?}",
        calls.len(),
        calls.iter().max()
    );
    Ok(())
}

fn model_path(args: &[String]) -> &std::path::Path {
    std::path::Path::new(&args[1])
}

fn run_for(model: &mut Model, iterations: usize) -> Result<f64, Error> {
    let start = Instant::now();
    for _ in 0..iterations {
        model.run()?;
    }
    Ok(iterations as f64 / start.elapsed().as_secs_f64())
}
