#!/usr/bin/env python3
"""
Example script demonstrating the edgefirst-ara2 Python library.

Connects to the ARA-2 proxy, retrieves version information,
and lists available endpoints with their status.
"""

import edgefirst_ara2 as ara2


def main():
    print(f"edgefirst-ara2 v{ara2.__version__}")
    print()

    # Connect to the ARA-2 proxy via UNIX socket
    try:
        session = ara2.Session.create_via_unix_socket(ara2.DEFAULT_SOCKET)
        print(f"Connected via {session.socket_type} socket")
    except ara2.ProxyError as e:
        print(f"Failed to connect: {e}")
        return

    # Get version information
    versions = session.versions()
    print("Component versions:")
    for component, version in sorted(versions.items()):
        print(f"  {component}: {version}")
    print()

    # List available endpoints
    endpoints = session.list_endpoints()
    print(f"Found {len(endpoints)} endpoint(s)")
    print()

    for i, endpoint in enumerate(endpoints):
        state = endpoint.check_status()
        stats = endpoint.dram_statistics()
        print(f"Endpoint {i}: {state}")
        print(f"  DRAM: {stats.free_size / 1048576:.1f} MB free "
              f"/ {stats.dram_size / 1048576:.1f} MB total")
        print(f"  Model: {stats.model_occupancy_size / 1048576:.1f} MB, "
              f"Tensor: {stats.tensor_occupancy_size / 1048576:.1f} MB")
        print()


if __name__ == "__main__":
    main()
