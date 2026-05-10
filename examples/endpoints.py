#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""
ARA-2 endpoint discovery and status.

Connects to the ARA-2 proxy service, retrieves software version
information for all components (proxy, firmware, driver), and lists
every available NPU endpoint with its operational state and DRAM
usage breakdown.

This example does not load a model or perform inference — it is a
diagnostic tool for verifying that the ARA-2 hardware and proxy
service are operational.

Usage::

    python endpoints.py

Requirements:
    edgefirst-ara2
    ARA-2 proxy service running (dvproxy — systemd unit: ara2.service or dvproxy.service)
"""

import edgefirst_ara2 as ara2


def main() -> None:
    """Connect to the ARA-2 proxy and print endpoint status.

    The function performs three steps:

    1. **Connect** — creates a ``Session`` via the default UNIX socket
       (``/var/run/ara2.sock``).
    2. **Versions** — queries the proxy for component version strings
       (proxy, firmware, driver, libaraclient).
    3. **Endpoints** — lists each NPU endpoint with its operational
       state (Idle, Active, Fault, ...) and DRAM statistics (total,
       free, model occupancy, tensor occupancy).

    Errors at each step are caught and printed rather than aborting,
    so partial information is still displayed if one query fails.
    """
    print(f"edgefirst-ara2 v{ara2.__version__}")
    print()

    # ── 1. Connect to the ARA-2 proxy via UNIX socket ────────────────
    try:
        session = ara2.Session.create_via_unix_socket(ara2.DEFAULT_SOCKET)
        print(f"Connected via {session.socket_type} socket")
    except ara2.ProxyError as e:
        print(f"Failed to connect: {e}")
        return

    # ── 2. Query component versions ──────────────────────────────────
    try:
        versions = session.versions()
        print("Component versions:")
        for component, version in sorted(versions.items()):
            print(f"  {component}: {version}")
        print()
    except ara2.Ara2Error as e:
        print(f"Failed to get versions: {e}")

    # ── 3. List NPU endpoints and their DRAM statistics ──────────────
    try:
        endpoints = session.list_endpoints()
    except ara2.Ara2Error as e:
        print(f"Failed to list endpoints: {e}")
        return

    print(f"Found {len(endpoints)} endpoint(s)")
    print()

    for i, endpoint in enumerate(endpoints):
        try:
            state = endpoint.check_status()
            stats = endpoint.dram_statistics()
            print(f"Endpoint {i}: {state}")
            print(f"  DRAM: {stats.free_size / 1048576:.1f} MB free "
                  f"/ {stats.dram_size / 1048576:.1f} MB total")
            print(f"  Model: {stats.model_occupancy_size / 1048576:.1f} MB, "
                  f"Tensor: {stats.tensor_occupancy_size / 1048576:.1f} MB")
        except ara2.HardwareError as e:
            print(f"Endpoint {i}: error - {e}")
        print()


if __name__ == "__main__":
    main()
