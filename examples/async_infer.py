#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
Async inference example for ARA-2 NPU.

Demonstrates the submit/wait asynchronous inference API. Loads a DVM
model, runs a warm-up, then compares synchronous ``run()`` against
asynchronous ``submit()`` + ``wait()`` over multiple iterations.

Usage::

    python async_infer.py /path/to/model.dvm [iterations]

Requirements:
    edgefirst-ara2
    ARA-2 proxy service running (dvproxy — systemd unit: ara2.service or dvproxy.service)
"""

from __future__ import annotations

import argparse
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="ARA-2 async inference example")
    parser.add_argument("model", help="Path to a compiled .dvm model file")
    parser.add_argument(
        "iterations", nargs="?", type=int, default=10, help="Number of iterations (> 0)"
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be > 0")

    import edgefirst_ara2 as ara2

    print(f"edgefirst-ara2 v{ara2.__version__}")
    print()

    # ── Connect ──────────────────────────────────────────────────────
    session = ara2.Session.create_via_unix_socket(ara2.DEFAULT_SOCKET)
    print(f"Connected via {session.socket_type} socket")

    endpoints = session.list_endpoints()
    assert len(endpoints) > 0, "No endpoints available"
    endpoint = endpoints[0]

    # ── Load model ───────────────────────────────────────────────────
    model = endpoint.load_model(args.model)
    model.allocate_tensors()
    print(f"Model loaded: {model.n_inputs} inputs, {model.n_outputs} outputs")

    # ── Warm-up ──────────────────────────────────────────────────────
    timing = model.run()
    print(
        f"Warm-up: inference={timing.run_time_us / 1000:.2f}ms, "
        f"input_xfer={timing.input_time_us / 1000:.2f}ms, "
        f"output_xfer={timing.output_time_us / 1000:.2f}ms"
    )
    print()

    n = args.iterations

    # ── Synchronous benchmark ────────────────────────────────────────
    print(f"Synchronous run() x {n}:")
    start = time.perf_counter()
    for _ in range(n):
        model.run()
    sync_elapsed = (time.perf_counter() - start) * 1000
    print(f"  Total: {sync_elapsed:.2f}ms, Avg: {sync_elapsed / n:.2f}ms/iter")

    # ── Async submit/wait benchmark ──────────────────────────────────
    print(f"Async submit() + wait() x {n}:")
    start = time.perf_counter()
    for _ in range(n):
        request = model.submit()
        # In a real pipeline you would preprocess the next frame here.
        request.wait()
    async_elapsed = (time.perf_counter() - start) * 1000
    print(f"  Total: {async_elapsed:.2f}ms, Avg: {async_elapsed / n:.2f}ms/iter")

    # ── Async with simulated CPU overlap ─────────────────────────────
    print(f"Async with simulated CPU work overlap x {n}:")
    start = time.perf_counter()
    for i in range(n):
        request = model.submit()

        # Simulate preprocessing work overlapping with NPU inference.
        cpu_start = time.perf_counter()
        total = sum(range(100_000))  # noqa: F841
        cpu_ms = (time.perf_counter() - cpu_start) * 1000

        timing = request.wait()
        if i == 0:
            print(
                f"  (first iter: cpu_work={cpu_ms:.2f}ms, "
                f"npu={timing.run_time_us / 1000:.2f}ms)"
            )
    overlap_elapsed = (time.perf_counter() - start) * 1000
    print(f"  Total: {overlap_elapsed:.2f}ms, Avg: {overlap_elapsed / n:.2f}ms/iter")

    print("\nDone.")


if __name__ == "__main__":
    main()
