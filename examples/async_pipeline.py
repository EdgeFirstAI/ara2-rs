#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
Async pipeline example for ARA-2 NPU with circular buffer.

Demonstrates pipelined inference using a ring of model instances, each
with its own DMA-BUF tensors. While the NPU executes on one set of
buffers, the CPU fills the next set and reads results from the previous.

Pipeline stages::

    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Fill (CPU)│──▶│Infer(NPU)│──▶│Read (CPU)│
    └──────────┘   └──────────┘   └──────────┘
         slot[i]      slot[i-1]      slot[i-2]

Usage::

    python async_pipeline.py /path/to/model.dvm [iterations] [depth]

Requirements:
    edgefirst-ara2
    ARA-2 proxy service running (dvproxy — systemd unit: ara2.service or dvproxy.service)
"""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ARA-2 async pipeline example")
    parser.add_argument("model", help="Path to a compiled .dvm model file")
    parser.add_argument(
        "iterations", nargs="?", type=int, default=100, help="Number of iterations (> 0)"
    )
    parser.add_argument(
        "depth", nargs="?", type=int, default=2, help="Ring buffer depth (> 0)"
    )
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be > 0")
    if args.depth <= 0:
        parser.error("depth must be > 0")

    import edgefirst_ara2 as ara2

    print(f"edgefirst-ara2 v{ara2.__version__}")
    print()

    # ── Connect ──────────────────────────────────────────────────────
    session = ara2.Session.create_via_unix_socket(ara2.DEFAULT_SOCKET)
    print(f"Connected via {session.socket_type} socket")

    endpoints = session.list_endpoints()
    assert len(endpoints) > 0, "No endpoints available"
    endpoint = endpoints[0]

    # ── Create circular buffer of model slots ────────────────────────
    n = args.iterations
    depth = args.depth
    slots = []
    for _ in range(depth):
        model = endpoint.load_model(args.model)
        model.allocate_tensors()
        slots.append(model)

    print(
        f"Pipeline: {depth} slots x ({slots[0].n_inputs} inputs, "
        f"{slots[0].n_outputs} outputs) with DMA-BUF tensors"
    )

    # Warm-up each slot
    for slot in slots:
        slot.run()
    print("Warm-up complete\n")

    # ── Synchronous baseline ─────────────────────────────────────────
    print(f"=== Synchronous baseline (fill → run → read) x {n} ===")
    start = time.perf_counter()
    for frame in range(n):
        slot = slots[0]
        fill_inputs(slot, frame)
        slot.run()
        read_outputs(slot)
    sync_ms = (time.perf_counter() - start) * 1000
    sync_fps = n / (sync_ms / 1000)
    print(f"  Total: {sync_ms:.2f}ms, Avg: {sync_ms / n:.2f}ms/iter, Throughput: {sync_fps:.1f} fps")

    # ── Pipelined async ──────────────────────────────────────────────
    print(f"\n=== Pipelined async (depth={depth}) x {n} ===")
    start = time.perf_counter()

    inflight: list[object | None] = [None] * depth

    for frame in range(n + depth):
        slot_idx = frame % depth

        # Wait for previous request on this slot (if any).
        if inflight[slot_idx] is not None:
            inflight[slot_idx].wait()
            read_outputs(slots[slot_idx])
            inflight[slot_idx] = None

        # Fill and submit if there are still frames to process.
        if frame < n:
            fill_inputs(slots[slot_idx], frame)
            inflight[slot_idx] = slots[slot_idx].submit()

    pipeline_ms = (time.perf_counter() - start) * 1000
    pipeline_fps = n / (pipeline_ms / 1000)
    print(
        f"  Total: {pipeline_ms:.2f}ms, Avg: {pipeline_ms / n:.2f}ms/iter, "
        f"Throughput: {pipeline_fps:.1f} fps"
    )

    # ── Summary ──────────────────────────────────────────────────────
    speedup = sync_ms / pipeline_ms
    print(f"\nSpeedup: {speedup:.2f}x (sync {sync_ms:.2f}ms vs pipeline {pipeline_ms:.2f}ms)")
    print(f"In-flight requests remaining: {session.inflight_count()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
