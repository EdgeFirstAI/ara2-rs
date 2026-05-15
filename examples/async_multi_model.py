#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
Multi-model async pipeline example for ARA-2 NPU.

Demonstrates pipelined inference across two different models loaded on
the same endpoint. Two scheduling patterns are benchmarked:

Dual-model
    Every frame is processed by both model A and model B. The async
    variant submits both requests then waits, allowing the NPU to queue
    and pipeline the work.

A/B alternating
    Even frames run on model A, odd frames on model B. Each model acts
    as a natural double-buffer — while one model runs on the NPU, the
    other model's inputs are prepared on the CPU.

Usage::

    python async_multi_model.py <model_a.dvm> <model_b.dvm> [iterations]

Requirements:
    edgefirst-ara2
    ARA-2 proxy service running (dvproxy — systemd unit: ara2.service or dvproxy.service)
"""

from __future__ import annotations

import argparse
import os
import time


def fill_inputs(model, frame: int) -> None:
    """Simulate CPU preprocessing by writing a pattern into input tensors."""
    import numpy as np

    val = frame % 256
    for i in range(model.n_inputs):
        shape = model.input_shape(i)
        data = np.full((shape[0], shape[1], shape[2]), val, dtype=np.uint8)
        model.set_input_tensor(i, data)


def read_outputs(model) -> int:
    """Simulate CPU postprocessing by reading the first byte of each output."""
    checksum = 0
    for i in range(model.n_outputs):
        data = model.get_output_tensor(i)
        checksum = (checksum + int(data.flat[0])) & 0xFF
    return checksum


def model_name(path: str) -> str:
    """Extract model name from file path for display."""
    return os.path.splitext(os.path.basename(path))[0]


def print_stats(elapsed_ms: float, n: int) -> None:
    """Print throughput statistics."""
    fps = n / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
    print(f"  Total: {elapsed_ms:.2f}ms, Avg: {elapsed_ms / n:.2f}ms/iter, Throughput: {fps:.1f} fps")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARA-2 multi-model async pipeline")
    parser.add_argument("model_a", help="Path to first .dvm model file")
    parser.add_argument("model_b", help="Path to second .dvm model file")
    parser.add_argument(
        "iterations", nargs="?", type=int, default=100, help="Number of iterations (> 0)"
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be > 0")

    import edgefirst_ara2 as ara2

    n = args.iterations
    name_a = model_name(args.model_a)
    name_b = model_name(args.model_b)

    print(f"edgefirst-ara2 v{ara2.__version__}")
    print()

    # ── Connect ──────────────────────────────────────────────────────
    session = ara2.Session.create_via_unix_socket(ara2.DEFAULT_SOCKET)
    print(f"Connected via {session.socket_type} socket")

    endpoints = session.list_endpoints()
    assert len(endpoints) > 0, "No endpoints available"
    endpoint = endpoints[0]

    # ── Load both models ─────────────────────────────────────────────
    model_a = endpoint.load_model(args.model_a)
    model_a.allocate_tensors()
    model_b = endpoint.load_model(args.model_b)
    model_b.allocate_tensors()

    print(f"Model A: {name_a} ({model_a.n_inputs} in, {model_a.n_outputs} out)")
    print(f"Model B: {name_b} ({model_b.n_inputs} in, {model_b.n_outputs} out)")

    # Warm-up
    model_a.run()
    model_b.run()
    print("Warm-up complete\n")

    # ════════════════════════════════════════════════════════════════
    # Benchmark 1: Dual-model — same image through both models
    # ════════════════════════════════════════════════════════════════

    # ── 1a: Synchronous ──────────────────────────────────────────────
    print(f"=== Dual-model sync (fill+run+read A, then B) x {n} ===")
    start = time.perf_counter()
    for frame in range(n):
        fill_inputs(model_a, frame)
        model_a.run()
        read_outputs(model_a)

        fill_inputs(model_b, frame)
        model_b.run()
        read_outputs(model_b)
    dual_sync_ms = (time.perf_counter() - start) * 1000

    print_stats(dual_sync_ms, n)

    # ── 1b: Async pipelined ──────────────────────────────────────────
    print(f"\n=== Dual-model async pipelined x {n} ===")
    start = time.perf_counter()

    inflight_a = None
    inflight_b = None

    for frame in range(n + 1):
        # Wait for previous iteration's requests
        if inflight_a is not None:
            inflight_a.wait()
            read_outputs(model_a)
            inflight_a = None
        if inflight_b is not None:
            inflight_b.wait()
            read_outputs(model_b)
            inflight_b = None

        # Submit next iteration (same data to both models)
        if frame < n:
            fill_inputs(model_a, frame)
            fill_inputs(model_b, frame)
            inflight_a = model_a.submit()
            inflight_b = model_b.submit()

    dual_async_ms = (time.perf_counter() - start) * 1000
    print_stats(dual_async_ms, n)

    speedup = dual_sync_ms / dual_async_ms if dual_async_ms > 0 else 0
    print(f"  Speedup: {speedup:.2f}x over sync")

    # ════════════════════════════════════════════════════════════════
    # Benchmark 2: A/B Alternating — even frames → A, odd → B
    # ════════════════════════════════════════════════════════════════

    # ── 2a: Synchronous ──────────────────────────────────────────────
    print(f"\n=== A/B alternating sync (even→A, odd→B) x {n} ===")
    start = time.perf_counter()
    for frame in range(n):
        if frame % 2 == 0:
            fill_inputs(model_a, frame)
            model_a.run()
            read_outputs(model_a)
        else:
            fill_inputs(model_b, frame)
            model_b.run()
            read_outputs(model_b)
    alt_sync_ms = (time.perf_counter() - start) * 1000
    print_stats(alt_sync_ms, n)

    # ── 2b: Async A/B alternating (pipelined) ────────────────────────
    print(f"\n=== A/B alternating async pipelined x {n} ===")
    start = time.perf_counter()

    models = [model_a, model_b]
    inflight = [None, None]

    for frame in range(n + 2):
        slot = frame % 2

        # Wait for previous request on this model (if any)
        if inflight[slot] is not None:
            inflight[slot].wait()
            read_outputs(models[slot])
            inflight[slot] = None

        # Submit next frame if within range
        if frame < n:
            fill_inputs(models[slot], frame)
            inflight[slot] = models[slot].submit()

    alt_async_ms = (time.perf_counter() - start) * 1000
    print_stats(alt_async_ms, n)

    speedup = alt_sync_ms / alt_async_ms if alt_async_ms > 0 else 0
    print(f"  Speedup: {speedup:.2f}x over sync")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║                    Summary                          ║")
    print("╠══════════════════════════════════════════════════════╣")
    dual_sync_fps = n / (dual_sync_ms / 1000) if dual_sync_ms > 0 else 0
    dual_async_fps = n / (dual_async_ms / 1000) if dual_async_ms > 0 else 0
    alt_sync_fps = n / (alt_sync_ms / 1000) if alt_sync_ms > 0 else 0
    alt_async_fps = n / (alt_async_ms / 1000) if alt_async_ms > 0 else 0
    print(f"║ Dual sync:     {dual_sync_fps:7.1f} fps  ({dual_sync_ms / n:.2f}ms/iter)    ║")
    print(f"║ Dual async:    {dual_async_fps:7.1f} fps  ({dual_async_ms / n:.2f}ms/iter)    ║")
    print(f"║ A/B alt sync:  {alt_sync_fps:7.1f} fps  ({alt_sync_ms / n:.2f}ms/iter)    ║")
    print(f"║ A/B alt async: {alt_async_fps:7.1f} fps  ({alt_async_ms / n:.2f}ms/iter)    ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"In-flight requests remaining: {session.inflight_count()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
