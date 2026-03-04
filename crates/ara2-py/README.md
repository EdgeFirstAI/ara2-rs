# Python Bindings for ARA-2

Python bindings for the ARA-2 neural network accelerator client library.

Published to PyPI as [`edgefirst-ara2`](https://pypi.org/project/edgefirst-ara2/).

## Installation

### From PyPI

```bash
pip install edgefirst-ara2
```

### Prerequisites for Development

- Python 3.11 or higher
- Rust stable toolchain (edition 2024)
- maturin (`pip install maturin`)
- ARA-2 client library (`libaraclient.so.1`)

### Development Install

```bash
cd crates/ara2-py
maturin develop --release --features pyo3/abi3-py311
```

## Quick Start

```python
import edgefirst_ara2

# Connect to ARA-2 proxy
session = edgefirst_ara2.Session.create_via_unix_socket("/var/run/ara2.sock")

# Get version information
versions = session.versions()
print(f"Proxy version: {versions['proxy']}")

# List endpoints
endpoints = session.list_endpoints()
print(f"Found {len(endpoints)} endpoints")

# Check endpoint status
for endpoint in endpoints:
    state = endpoint.check_status()
    stats = endpoint.dram_statistics()
    print(f"State: {state}")
    print(f"Free DRAM: {stats.free_size / stats.dram_size * 100:.1f}%")
```

See `examples/ara2_python_example.py` for a complete example.

## API Reference

### Session

Connection to the ARA-2 proxy service.

**Static Methods:**
- `create_via_unix_socket(socket_path: str) -> Session`
- `create_via_tcp_ipv4_socket(ip: str, port: int) -> Session`

**Methods:**
- `versions() -> dict[str, str]` - Get component versions
- `list_endpoints() -> list[Endpoint]` - List available endpoints

### Endpoint

Represents an ARA-2 accelerator device.

**Methods:**
- `check_status() -> State` - Get device state
- `dram_statistics() -> DramStatistics` - Get memory usage
- `load_model(model_path: str, output_quantization: OutputQuantization | None = None) -> Model`

### Model

Loaded neural network model.

**Methods:**
- `run() -> ModelTiming` - Execute inference
- `n_inputs() -> int` - Number of input tensors
- `n_outputs() -> int` - Number of output tensors

### Supporting Types

- **State** (enum): Init, Idle, Active, ActiveSlow, ActiveBoosted, ThermalInactive, ThermalUnknown, Inactive, Fault
- **OutputQuantization** (enum): None, U8, I8
- **DramStatistics**: Memory usage information (properties: dram_size, dram_occupancy_size, free_size, reserved_occupancy_size, reserved_free_size)
- **ModelTiming**: Performance metrics (properties: run_time_us, input_time_us, output_time_us)

## Building Wheels

```bash
cd crates/ara2-py
maturin build --release --features pyo3/abi3-py311
```

Wheels are created in `target/wheels/`.

## Stable ABI

The Python bindings use PyO3's stable ABI feature (`abi3-py311`), which means:

- A single wheel works across Python 3.11, 3.12, 3.13, and future versions
- No need to build separate wheels for each Python version
- Minimum supported Python version is 3.11

## Examples and Tests

- **Example**: `examples/ara2_python_example.py`

Run example:
```bash
python examples/ara2_python_example.py
```

Verify installation:
```bash
python -c "import edgefirst_ara2; print(edgefirst_ara2.version())"
```

## Troubleshooting

### "libaraclient.so.1 not found"

Add the library to your path:
```bash
export LD_LIBRARY_PATH=/path/to/ara2/lib:$LD_LIBRARY_PATH
```

### Build Errors

Clean and rebuild:
```bash
cargo clean
cd crates/ara2-py
maturin develop --release --features pyo3/abi3-py311
```

### Import Error

Verify installation:
```bash
pip list | grep edgefirst-ara2
```

Reinstall if needed:
```bash
cd crates/ara2-py
maturin develop --release --features pyo3/abi3-py311
```

## Architecture

The Python bindings follow design patterns from:
- [EdgeFirst Client](https://github.com/edgefirstai/client)
- [How I Design And Develop Real-World Python Extensions In Rust](https://medium.com/@kudryavtsev_ia/how-i-design-and-develop-real-world-python-extensions-in-rust-2abfe2377182)

**Key principles:**
- Separate crates: `ara2` (core) + `ara2-py` (bindings)
- Tuple struct wrappers for Python classes
- Automatic error conversion (Rust `Result` -> Python exceptions)
- Type stubs for IDE support

## License

Licensed under the Apache License 2.0.

Copyright 2025 Au-Zone Technologies. All Rights Reserved.
