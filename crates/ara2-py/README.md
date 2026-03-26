# Python Bindings for ARA-2

Python bindings for the ARA-2 neural network accelerator client library,
providing efficient NPU inference from Python via a proxy service running
on NXP i.MX platforms with Kinara ARA-2 hardware.

Published to PyPI as [`edgefirst-ara2`](https://pypi.org/project/edgefirst-ara2/).

## Architecture

```
Python Application ──(UNIX/TCP socket)──▶ ara2-proxy ──(PCIe)──▶ ARA-2 NPU
       │                                (system service)        (Kinara hardware)
       │
edgefirst-hal ──(DMA-BUF fd)──▶ GPU preprocessing (zero-copy)
```

Your Python code connects to the `ara2-proxy` system service (not directly
to the hardware). The proxy manages device access and must be running before
your application starts.

## Installation

### From PyPI

```bash
pip install edgefirst-ara2
```

For zero-copy preprocessing with edgefirst-hal:

```bash
pip install edgefirst-ara2[hal]
```

### Prerequisites for Development

- Python 3.11 or higher
- Rust stable toolchain (edition 2024)
- maturin (`pip install maturin`)
- ARA-2 client library (`libaraclient.so.1`)

### Development Install

```bash
cd crates/ara2-py
maturin develop --release --features abi3
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
    print(f"State: {state}, Free DRAM: {stats.free_size / stats.dram_size * 100:.1f}%")
```

## Inference with numpy

```python
import numpy as np
import edgefirst_ara2

session = edgefirst_ara2.Session.create_via_unix_socket("/var/run/ara2.sock")
endpoints = session.list_endpoints()
model = endpoints[0].load_model("model.dvm")

# Allocate tensors and run inference
model.allocate_tensors()
input_data = np.zeros(model.input_size(0), dtype=np.uint8)
model.set_input_tensor(0, input_data)
timing = model.run()

print(f"Inference: {timing.run_time_us} us")
output = model.get_output_tensor(0)
dequantized = model.dequantize(0)
```

## Zero-Copy DMA-BUF Pipeline

For maximum throughput, use DMA-BUF tensors with
[edgefirst-hal](https://pypi.org/project/edgefirst-hal/) for GPU-accelerated
preprocessing. This eliminates CPU memory copies between preprocessing and
inference:

| Path | CPU copies | Flow |
|------|-----------|------|
| Standard (numpy) | 2 | numpy → shared memory → NPU |
| DMA-BUF | 0 | GPU writes directly to NPU input buffer |

**How it works:** `allocate_tensors("dma")` allocates the model's input tensor
in a DMA-BUF — a Linux kernel buffer accessible by multiple hardware devices.
`input_tensor_fd(0)` returns a file descriptor to that buffer. You pass this
FD to `edgefirst_hal.import_image()`, which maps it as a GPU image surface.
The GPU writes the preprocessed frame directly into the NPU's input buffer —
no CPU copies involved.

```python
import os
import edgefirst_ara2 as ara2
import edgefirst_hal as hal

session = ara2.Session.create_via_unix_socket(ara2.DEFAULT_SOCKET)
endpoint = session.list_endpoints()[0]

with endpoint.load_model("yolov8s.dvm") as model:
    model.allocate_tensors("dma")  # Must use "dma" for tensor FD access

    # Get DMA-BUF FD for the model's input tensor
    input_fd = model.input_tensor_fd(0)
    c, h, w = model.input_shape(0)
    try:
        # Import as PlanarRgb (CHW layout) to match ARA-2 tensor format
        dst = hal.import_image(input_fd, w, h, hal.PixelFormat.PlanarRgb)
    finally:
        os.close(input_fd)  # FD duplicated by import_image; close original

    # GPU-accelerated convert: camera frame -> model input (zero CPU copies)
    processor = hal.ImageProcessor()
    src = hal.load_image("image.jpg", format=hal.PixelFormat.Rgba, mem=hal.TensorMemory.DMA)
    processor.convert(src, dst)

    # Run inference — NPU reads from the same DMA-BUF
    timing = model.run()
    print(f"Inference: {timing.run_time_us} us")
```

## Performance

Benchmarked on NXP i.MX 8M Plus + ARA-2 with YOLOv8n (640x640).
The Python API adds minimal overhead over native Rust thanks to DMA-BUF
zero-copy — GPU and NPU operate on the same physical memory buffers.

| Stage | Rust | Python | Overhead |
|-------|------|--------|----------|
| GPU preprocess (RGBA → CHW) | 6.35 ms | 6.37 ms | +0.02 ms |
| NPU inference (wall clock) | 8.95 ms | 9.13 ms | +0.18 ms |
| &nbsp;&nbsp;NPU execution | 3.33 ms | 3.33 ms | — |
| &nbsp;&nbsp;DMA input upload | 2.21 ms | 2.20 ms | — |
| &nbsp;&nbsp;DMA output download | 1.96 ms | 1.96 ms | — |
| Postprocess (decode + NMS) | 1.41 ms | 2.53 ms | +1.12 ms |
| **Total pipeline** | **16.71 ms** | **18.03 ms** | **+1.32 ms** |
| **Throughput** | **59.9 FPS** | **55.5 FPS** | |

> Steady-state mean over 20 iterations. Python overhead is in postprocessing
> (numpy array marshalling). GPU preprocessing and NPU inference are identical.

Run the benchmark yourself:

```bash
python examples/yolov8.py model.dvm image.jpg --benchmark 20
```

## DVM Metadata

Read model metadata without loading onto the NPU:

```python
import edgefirst_ara2

metadata = edgefirst_ara2.read_metadata("model.dvm")
if metadata:
    print(f"Task: {metadata.task}")
    print(f"Classes: {metadata.classes}")
    if metadata.compilation and metadata.compilation.ppa:
        print(f"IPS: {metadata.compilation.ppa.ips}")

labels = edgefirst_ara2.read_labels("model.dvm")
```

## API Reference

### Session

Connection to the ARA-2 proxy service.

**Static Methods:**
- `create_via_unix_socket(socket_path: str) -> Session`
- `create_via_tcp_ipv4_socket(ip: str, port: int) -> Session`

**Methods:**
- `versions() -> dict[str, str]` - Get component versions
- `list_endpoints() -> list[Endpoint]` - List available endpoints

**Properties:**
- `socket_type: str` - "unix" or "tcp"

### Endpoint

Represents an ARA-2 accelerator device.

**Methods:**
- `check_status() -> State` - Get device state
- `dram_statistics() -> DramStatistics` - Get memory usage
- `load_model(model_path: str) -> Model` - Load a .dvm model

### Model

Loaded neural network model.

**Lifecycle:**
- `allocate_tensors(memory: str | None = None)` - Allocate tensors ("dma", "shm", "mem", or None)
- `set_timeout_ms(timeout_ms: int)` - Set inference timeout
- `run() -> ModelTiming` - Execute inference

**Tensor I/O (numpy):**
- `set_input_tensor(index: int, data: np.ndarray)` - Copy data into input
- `get_output_tensor(index: int) -> np.ndarray` - Copy output data out
- `dequantize(index: int) -> np.ndarray` - Dequantize output to float32

**DMA-BUF Zero-Copy:**
- `input_tensor_fd(index: int) -> int` - Get input tensor FD
- `output_tensor_fd(index: int) -> int` - Get output tensor FD
- `input_tensor_memory(index: int) -> str` - Input memory type
- `output_tensor_memory(index: int) -> str` - Output memory type

**Introspection:**
- `n_inputs: int`, `n_outputs: int` - Tensor counts
- `input_shape(i) -> (C, H, W)`, `output_shape(i) -> (C, H, W)`
- `input_size(i) -> int`, `output_size(i) -> int` - Size in bytes
- `input_bpp(i) -> int`, `output_bpp(i) -> int` - Bytes per element
- `input_info(i) -> InputTensorInfo`, `output_info(i) -> OutputTensorInfo`
- `input_quants(i) -> InputQuantization`, `output_quants(i) -> OutputQuantization`

### Metadata Functions

- `read_metadata(path: str) -> DvmMetadata | None`
- `read_labels(path: str) -> list[str]`
- `has_metadata(path: str) -> bool`

### Supporting Types

- **State** (enum): Init, Idle, Active, ActiveSlow, ActiveBoosted, ThermalInactive, ThermalUnknown, Inactive, Fault
- **ModelOutputType** (enum): Classification, Detection, SemanticSegmentation, Raw
- **DramStatistics**: dram_size, free_size, model_occupancy_size, ...
- **ModelTiming**: run_time_us, input_time_us, output_time_us
- **InputQuantization**: qn, scale, mean, is_signed
- **OutputQuantization**: qn, scale, offset, is_signed

### Exceptions

```
Ara2Error (RuntimeError)
 +-- LibraryError       - libaraclient.so loading failures
 +-- HardwareError      - NPU faults, endpoint errors
 +-- ProxyError         - Proxy connection failures
 +-- ModelError         - Model load/inference failures
 +-- TensorError        - Tensor allocation, DMA-BUF errors
 +-- MetadataError      - DVM metadata parsing errors
```

## Building Wheels

```bash
cd crates/ara2-py
maturin build --release --features abi3
```

Wheels are created in `target/wheels/`.

## Stable ABI

The bindings use PyO3's stable ABI (`abi3-py311`):
- A single wheel works across Python 3.11, 3.12, 3.13, and future versions
- Minimum supported Python version is 3.11

## Troubleshooting

### "libaraclient.so.1 not found"

```bash
export LD_LIBRARY_PATH=/path/to/ara2/lib:$LD_LIBRARY_PATH
```

### Verify Installation

```bash
python -c "import edgefirst_ara2; print(edgefirst_ara2.__version__)"
```

## License

Licensed under the Apache License 2.0.

Copyright 2025 Au-Zone Technologies. All Rights Reserved.
