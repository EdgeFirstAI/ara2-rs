#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
Telemetry example for ARA-2 NPU.

Prints everything the runtime reports about the endpoint, a loaded model and
one inference. Fields the loaded DVAPI version does not provide print as
``n/a``.

Usage::

    python telemetry.py /path/to/model.dvm

Requirements:
    edgefirst-ara2
    ARA-2 proxy service running (systemd unit: rt-sdk-ara2.service)
"""

from __future__ import annotations

import argparse


def na(value) -> str:
    return "n/a" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARA-2 telemetry example")
    parser.add_argument("model", help="Path to a compiled .dvm model file")
    args = parser.parse_args()

    import edgefirst_ara2 as ara2

    session = ara2.Session.connect()
    print(
        f"DVAPI {session.abi} via {session.socket_type}, "
        f"fd passing: {session.supports_fd_passing}"
    )
    for v in session.versions():
        print(f"  running {v.component:<14} {v.version}")

    endpoint = session.list_endpoints()[0]

    info = endpoint.info()
    print(f"\ndevice {info.device_id:#x} vendor {info.vendor_id:#x}")
    if info.chip is not None:
        chip = info.chip
        print(
            f"chip {na(chip.id)} rev {na(chip.revision)}: "
            f"{chip.control_processors} control, {chip.neural_processors} neural "
            f"processors, {chip.l2_bytes} B L2"
        )
    if info.dram is not None:
        print(f"dram vendor {na(info.dram.vendor_name)}")
    if info.interface is not None:
        i = info.interface
        print(
            f"interface {i.kind} bus {i.bus} device {i.device} port {i.port} "
            f"sysfs {na(i.sysfs_path)}"
        )

    es = endpoint.statistics()
    print(
        f"\nstate {es.state}, {es.temp_c:.1f} C, {es.core_voltage_v:.3f} V, "
        f"soft resets {es.soft_reset_count}"
    )
    c = es.clocks
    print(
        f"clocks: sys {c.sys_mhz} MHz, dram {c.dram_mhz} MHz, "
        f"nnp {na(c.nnp_mhz)} MHz, sbp {na(c.sbp_mhz)} MHz"
    )
    d = es.dram
    print(
        f"dram: {d.dram_size} B total, {d.dram_occupancy_size} B occupied, "
        f"{d.free_size} B free, largest contiguous free "
        f"{na(d.largest_contiguous_free)}"
    )

    try:
        print(f"\nbusy {endpoint.is_busy()}")
    except ara2.UnsupportedError as e:
        print(f"\nbusy: n/a ({e})")

    model = endpoint.load_model(args.model)
    model.allocate_tensors(None)
    print(f"\nmodel {na(model.name)} type {model.model_type} version {model.version}")

    run = model.run()
    t = run.timing
    print(
        f"\nrun {t.run_time * 1e3:.3f} ms, input {t.input_time * 1e3:.3f} ms, "
        f"output {t.output_time * 1e3:.3f} ms"
    )
    c = run.clocks
    print(
        f"clocks: sys {c.sys_mhz} MHz, dram {c.dram_mhz} MHz, "
        f"nnp {na(c.nnp_mhz)} MHz, sbp {na(c.sbp_mhz)} MHz"
    )
    print(f"cycles: {run.hw_cycles} hardware, {run.fp_cycles} floating point")
    print(
        f"timestamps (CLOCK_MONOTONIC s): input {na(run.input_start)}, "
        f"inference {na(run.inference_start)}, output {na(run.output_start)}"
    )
    print(
        f"retries: {run.retries_this_inference} this inference, "
        f"{run.retries_session_total} this session"
    )


if __name__ == "__main__":
    main()
