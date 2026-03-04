#!/usr/bin/env python3
"""
Example script demonstrating the ara2 Python library.

This script connects to an ARA-2 proxy, retrieves version information,
and lists available endpoints with their status.
"""

import ara2


def main():
    # Print library version
    print(f"ara2 Python library version: {ara2.version()}")
    print()

    # Connect to the ARA-2 proxy via UNIX socket
    # Change this path to match your ARA-2 proxy configuration
    socket_path = "/var/run/ara2.sock"
    
    try:
        session = ara2.Session.create_via_unix_socket(socket_path)
        print(f"Connected to ARA-2 proxy via {socket_path}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("\nTrying TCP connection as fallback...")
        
        # Fallback to TCP connection
        try:
            session = ara2.Session.create_via_tcp_ipv4_socket("127.0.0.1", 5555)
            print("Connected via TCP to 127.0.0.1:5555")
        except Exception as e:
            print(f"Failed to connect via TCP: {e}")
            return
    
    print()

    # Get version information
    try:
        versions = session.versions()
        print("Component versions:")
        for component, version in sorted(versions.items()):
            print(f"  {component}: {version}")
        print()
    except Exception as e:
        print(f"Failed to get versions: {e}")
    
    # List available endpoints
    try:
        endpoints = session.list_endpoints()
        print(f"Found {len(endpoints)} endpoint(s)")
        print()
        
        for i, endpoint in enumerate(endpoints):
            print(f"Endpoint {i}:")
            
            # Check endpoint status
            try:
                state = endpoint.check_status()
                print(f"  State: {state}")
            except Exception as e:
                print(f"  Failed to check status: {e}")
            
            # Get DRAM statistics
            try:
                stats = endpoint.dram_statistics()
                print(f"  DRAM Statistics:")
                print(f"    Total size: {stats.dram_size:,} bytes")
                print(f"    Used: {stats.dram_occupancy_size:,} bytes")
                print(f"    Free: {stats.free_size:,} bytes")
                print(f"    Reserved occupied: {stats.reserved_occupancy_size:,} bytes")
                print(f"    Reserved free: {stats.reserved_free_size:,} bytes")
            except Exception as e:
                print(f"  Failed to get DRAM statistics: {e}")
            
            print()
    
    except Exception as e:
        print(f"Failed to list endpoints: {e}")


if __name__ == "__main__":
    main()
