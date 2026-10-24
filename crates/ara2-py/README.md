# Python Bindings for ARA-2

Python bindings for the ARA-2 neural network accelerator client library,
providing efficient NPU inference from Python via a proxy service running
on NXP i.MX platforms with Kinara ARA-2 hardware.

Published to PyPI as [`edgefirst-ara2`](https://pypi.org/project/edgefirst-ara2/).

## Architecture

```
Python Application ──(UNIX/TCP socket)──▶ ARA-2 proxy ──(PCIe)──▶ ARA-2 NPU
       │                                (system service)         (Kinara hardware)
       │
edgefirst-image ──(DMA-BUF fd)──▶ GPU preprocessing (zero-copy)
```

Your Python code connects to the proxy system service (not directly to the
hardware). The proxy manages device access and must be running before your
application starts — it is `rt-sdk-ara2.service` on EdgeFirst Yocto images
shipping NXP's rt-sdk-ara2 integration:

```bash
systemctl status rt-sdk-ara2
```

## Installation

### From PyPI

```bash
pip install edgefirst-ara2
```

For zero-copy preprocessing with the EdgeFirst HAL:

```bash
pip install edgefirst-ara2[hal]
```

The `hal` extra pulls `edgefirst-image` (and, through it, `edgefirst-tensor`).
The HAL ships as one wheel per library rather than a single `edgefirst-hal`
wheel; add `edgefirst-codec` for JPEG/PNG decode and `edgefirst-decoder` for
YOLO post-processing when you need them.

### Prerequisites for Development

- Python 3.11 or higher
- Rust stable toolchain (edition 2024)
- maturin (`pip install maturin`)
- ARA-2 client library (`libaraclient`, see `ara2::LIBRARY_NAMES` for the names tried)

### Development Install

```bash
cd crates/ara2-py
maturin develop --release --features abi3
```

## Quick Start

```python
import edgefirst_ara2

# Connect to ARA-2 proxy
session = edgefirst_ara2.Session.connect()

# Get version information
for v in session.versions():
    print(f"{v.component}: {v.version}")

# List endpoints
endpoints = session.list_endpoints()
print(f"Found {len(endpoints)} endpoints")

# Check endpoint status
for endpoint in endpoints:
    state = endpoint.check_status()
    stats = endpoint.dram_statistics()
    print(f"State: {state}, Free DRAM: {stats.free_size / stats.dram_size * 100:.1f}%")
```

Over TCP file descriptors cannot be passed, so `session.supports_fd_passing` is `False` and tensors use heap memory instead of shared-memory or DMA-BUF. That applies to `create_via_tcp_ipv4_socket`, and also to `Session.connect()` when the only proxy it finds listens on TCP (a proxy configured for `IPV4`, as the r3.0 sample configuration is). Explicitly requesting `"shm"` or `"dma"` from `allocate_tensors()` over TCP raises `TensorError`. Heap tensors have no file descriptor, so `input_tensor_fd()` and `output_tensor_fd()` raise `TensorError` too: check `session.supports_fd_passing` and copy with `set_input_tensor()` and `get_output_tensor()` instead, as the `yolov8` examples do.

## Inference with numpy

```python
import numpy as np
import edgefirst_ara2

session = edgefirst_ara2.Session.connect()
endpoints = session.list_endpoints()
model = endpoints[0].load_model("model.dvm")

# Allocate tensors and run inference
model.allocate_tensors()
input_data = np.zeros(model.input_size(0), dtype=np.uint8)
model.set_input_tensor(0, input_data)
stats = model.run()

print(f"Inference: {stats.timing.run_time * 1e3:.2f} ms")
output = model.get_output_tensor(0)
dequantized = model.dequantize(0)
```

## Zero-Copy DMA-BUF Pipeline

For maximum throughput, use DMA-BUF tensors with
[edgefirst-image](https://pypi.org/project/edgefirst-image/) for GPU-accelerated
preprocessing. This eliminates CPU memory copies between preprocessing and
inference:

| Path | CPU copies | Flow |
|------|-----------|------|
| Standard (numpy) | 2 | numpy → shared memory → NPU |
| DMA-BUF | 0 | GPU writes directly to NPU input buffer |

**How it works:** `allocate_tensors("dma")` allocates the model's input tensor
in a DMA-BUF — a Linux kernel buffer accessible by multiple hardware devices.
`input_tensor_fd(0)` returns a file descriptor to that buffer. You pass this
FD to `edgefirst.image.ImageProcessor.import_image()`, which maps it as a GPU
image surface.
The GPU writes the preprocessed frame directly into the NPU's input buffer —
no CPU copies involved.

```python
import os

import edgefirst.codec as ef_codec
import edgefirst.image as ef_image
import edgefirst_ara2 as ara2

session = ara2.Session.connect()
endpoint = session.list_endpoints()[0]
processor = ef_image.ImageProcessor()

with endpoint.load_model("yolov8s.dvm") as model:
    model.allocate_tensors("dma")  # Must use "dma" for tensor FD access

    # Get DMA-BUF FD for the model's input tensor
    input_fd = model.input_tensor_fd(0)
    c, h, w = model.input_shape(0)
    try:
        # Import as PlanarRgb (CHW layout) to match ARA-2 tensor format
        dst = processor.import_image(input_fd, w, h, ef_image.PixelFormat.PlanarRgb)
    finally:
        os.close(input_fd)  # FD duplicated by import_image; close original

    # edgefirst-codec decodes into a buffer edgefirst-image allocated. The
    # decoder never converts, so a colour JPEG lands in its native NV12.
    info = ef_codec.Tensor.peek_image_info_file("image.jpg")
    src = processor.create_image(info.width, info.height, info.format)
    ef_codec.decode_file_into(src, "image.jpg")

    # GPU-accelerated convert: source frame -> model input (zero CPU copies).
    # `letterbox=` fits the source preserving aspect ratio and pads the rest.
    processor.convert(src, dst, letterbox=(114, 114, 114, 255))

    # Run inference — NPU reads from the same DMA-BUF
    stats = model.run()
    print(f"Inference: {stats.timing.run_time * 1e3:.2f} ms")
```

## Performance

Benchmarked on NXP FRDM i.MX 95 + ARA-2 with YOLOv8m-seg (640×640).
The Python API adds minimal overhead over native Rust thanks to DMA-BUF
zero-copy — GPU and NPU operate on the same physical memory buffers.

| Stage | Rust | Python | Overhead |
|-------|------|--------|----------|
| GPU preprocess (letterbox + RGBA→CHW) | 2.85 ms | 2.88 ms | +0.03 ms |
| NPU inference (wall clock) | 34.53 ms | 34.63 ms | +0.10 ms |
| &nbsp;&nbsp;NPU execution | 26.04 ms | 26.04 ms | — |
| &nbsp;&nbsp;DMA input upload | 2.02 ms | 2.05 ms | — |
| &nbsp;&nbsp;DMA output download | 3.68 ms | 3.68 ms | — |
| Decode (NMS + dequant) | 4.05 ms | 4.31 ms | +0.26 ms |
| Materialize (CPU coeff × proto → bitmaps) | 5.67 ms | 5.98 ms | +0.31 ms |
| Draw (GL mask overlay) | 5.54 ms | 5.71 ms | +0.17 ms |
| **Total pipeline** | **52.64 ms** | **53.52 ms** | **+0.88 ms** |
| **Throughput** | **19.0 FPS** | **18.7 FPS** | |

> Steady-state mean over 30 iterations after warmup. Python overhead is
> under 1 ms across the entire pipeline.

Run the benchmark yourself. Create a virtual environment on the target and
install the packages from PyPI:

```bash
python3 -m venv ~/venv
~/venv/bin/pip install edgefirst-ara2 'edgefirst-codec>=0.37' \
    'edgefirst-decoder>=0.37' 'edgefirst-image>=0.37'
~/venv/bin/python3 yolov8.py yolov8m-seg-int16.dvm zidane.jpg --benchmark 30 --save
```

## Models

Official pre-trained models are published in the EdgeFirst model zoo on
Hugging Face:

| Task | Repository |
| --- | --- |
| Detection | <https://huggingface.co/EdgeFirst/yolov8-det> |
| Segmentation | <https://huggingface.co/EdgeFirst/yolov8-seg> |

Each repository ships one directory per target — `tflite/`, `imx95/`,
`onnx/`, `hailo/`, `jetson/`, `qnn/`. The ARA-2 builds are the int16 `.dvm`
exports under `ara240/`, in `n`/`s`/`m` sizes:

```bash
curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/ara240/yolov8n-det-int16.dvm
curl -LO https://huggingface.co/EdgeFirst/yolov8-seg/resolve/main/ara240/yolov8n-seg-int16.dvm
```

Every export embeds an `edgefirst.json` schema and the class `labels.txt` in
a ZIP trailer appended to the `.dvm`; `read_metadata()` and `read_labels()`
below read them without loading the model onto the NPU.

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

### Model compatibility

A `.dvm` is compiled for one runtime generation: model format 2 (converters built on the DVAPI 1.x SDKs) loads only on DVAPI 1.x runtimes (verified on 1.1 and 1.3), and format 3 (DVAPI 3.0 SDK) only on DVAPI 3.x. `read_model_format()` reads the format without loading the model, and returns `None` when the file's structure is not recognised:

```python
fmt = edgefirst_ara2.read_model_format("model.dvm")  # 2, 3 or None
```

The client library reports a model of the other generation exactly as it reports a damaged file. `Endpoint.load_model()` reads the format first, so when the library cannot parse the file the exception (`ModelError`, `code` 520, `name` `"DV_MODEL_LOAD_FAILURE"`) says which it is: `compatibility` is `"mismatch"`, `"matching"` or `"unknown"`, `model_format` is the format read, and the message says whether to reconvert the model. `metadata.ara2.converter_version` and `sdk_version` give the converter provenance when the model carries it.

```python
try:
    model = endpoint.load_model("model.dvm")
except edgefirst_ara2.Ara2Error as e:
    if e.compatibility == "mismatch":
        print(f"model format {e.model_format} needs reconverting for {session.abi}")
    raise
```

## Async Inference

The `submit()` / `wait()` API enables overlapping CPU work with NPU
execution. This is the building block for pipeline parallelism — while
the NPU runs inference on one frame, the CPU can preprocess the next.

```python
import edgefirst_ara2 as ara2

session = ara2.Session.connect()
endpoint = session.list_endpoints()[0]
model = endpoint.load_model("model.dvm")
model.allocate_tensors()

# Submit — returns immediately while NPU works
request = model.submit()
print(f"Request #{request.request_id} submitted")

# CPU is free to do other work here...
# The GIL is NOT held during wait(), so other Python threads can run

stats = request.wait()  # blocks until NPU finishes
print(f"NPU inference: {stats.timing.run_time * 1e3:.2f} ms")

# Monitor pipeline depth
print(f"In-flight: {session.inflight_count()}")
```

> **Warning:** Do not call `model.allocate_tensors()` while an
> `InferRequest` is still pending — the NPU is actively reading/writing
> the tensor buffers.

## Telemetry and Control

`Model.run()` and `InferRequest.wait()` return an `InferenceStats` with the
timing, the clocks the inference ran at, hardware cycle counts and, on DVAPI
3.0, `CLOCK_MONOTONIC` timestamps comparable with `time.monotonic()`.
The timestamps are `None` on DVAPI 1.x, whose runtime stamps with the wall
clock.

```python
import time
import edgefirst_ara2 as ara2

session = ara2.Session.connect()
endpoint = session.list_endpoints()[0]

info = endpoint.info()
if info.chip is not None:
    print(info.chip.id, info.chip.neural_processors)
if info.interface is not None:
    print(info.interface.kind)

stats = endpoint.statistics()
print(
    f"{stats.temp_c:.1f} C, NNP {stats.clocks.nnp_mhz} MHz, "
    f"{stats.dram.free_size} B DRAM free"
)
print(endpoint.is_busy())

model = endpoint.load_model("model.dvm")
model.allocate_tensors()
before = time.monotonic()
run = model.run()
print(f"{run.timing.run_time * 1e3:.2f} ms, {run.hw_cycles} cycles, {model.model_type}")
if run.inference_start is not None:
    print(f"queued {run.inference_start - before:.6f} s after submit")
```

Per-model statistics, inference queue statistics, the endpoint power state and power-state control are not exposed, because the runtime does not yet report or apply them reliably.

The client library's own logging (DVAPI 1.3 and later) is set per session:

```python
session.set_log_level(ara2.LogLevel.Warn)  # raises UnsupportedError on DVAPI 1.1
```

## API Reference

### Session

Connection to the ARA-2 proxy service.

**Static Methods:**
- `connect() -> Session` - Connect to the endpoint `proxy_endpoint()` resolves: `ARA2_SOCKET` (a path, `unix:` or `tcp:`), then the known socket paths, then running proxies, preferring UNIX
- `create_via_unix_socket(socket_path: str) -> Session`
- `create_via_tcp_ipv4_socket(ip: str, port: int) -> Session`

**Methods:**
- `versions() -> list[ComponentVersion]` - Versions of the running components
- `supported_versions() -> list[ComponentVersion]` - Every component version the proxy supports
- `list_endpoints() -> list[Endpoint]` - List available endpoints
- `inflight_count() -> int` - Number of pending async inference requests
- `set_log_level(level: LogLevel)` - Client library log verbosity (DVAPI 1.3 and later)

**Properties:**
- `socket_type: str` - "unix" or "tcp"
- `supports_fd_passing: bool` - True over UNIX sockets (shared-memory and DMA-BUF tensors); False over TCP, where tensors are heap-backed

### Endpoint

Represents an ARA-2 accelerator device.

**Methods:**
- `check_status() -> State` - Get device state
- `dram_statistics() -> DramStatistics` - Get memory usage
- `statistics() -> EndpointStatistics` - State, clocks, voltage, temperature, soft resets and DRAM
- `info() -> EndpointInfo` - Chip, DRAM and host-interface details
- `is_busy() -> bool` - Whether an inference is running or queued; raises `UnsupportedError` where the runtime lacks the call
- `load_model(model_path: str) -> Model` - Load a .dvm model

### Model

Loaded neural network model.

**Lifecycle:**
- `allocate_tensors(memory: str | None = None)` - Allocate tensors ("dma", "shm", "mem", or None)
- `set_timeout(timeout: float)` - Set the `run()` timeout in seconds (default 1.0)
- `run() -> InferenceStats` - Execute inference synchronously
- `submit() -> InferRequest` - Submit inference asynchronously (returns immediately)

**Tensor I/O (numpy):**
- `set_input_tensor(index: int, data: np.ndarray)` - Copy data into input
- `get_output_tensor(index: int) -> np.ndarray` - Copy output data out
- `dequantize(index: int) -> np.ndarray` - Dequantize output to float32

**DMA-BUF Zero-Copy:**
- `input_tensor_fd(index: int) -> int` - Get input tensor FD (pass to `edgefirst.image.ImageProcessor.import_image`, which dups it — close after)
- `output_tensor_fd(index: int) -> int` - Get output tensor FD (pass to `edgefirst.tensor.Tensor.from_fd`, which takes ownership — do **not** close after)
- `input_tensor_memory(index: int) -> str` - Input memory type
- `output_tensor_memory(index: int) -> str` - Output memory type

**Introspection:**
- `n_inputs: int`, `n_outputs: int` - Tensor counts
- `version: str`, `name: str | None`, `model_type: ModelType`
- `input_shape(i) -> (C, H, W)`, `output_shape(i) -> (C, H, W)`
- `input_size(i) -> int`, `output_size(i) -> int` - Size in bytes
- `input_bpp(i) -> int`, `output_bpp(i) -> int` - Bytes per element
- `input_info(i) -> InputTensorInfo`, `output_info(i) -> OutputTensorInfo`
- `input_quants(i) -> InputQuantization`, `output_quants(i) -> OutputQuantization`

### InferRequest

Pending asynchronous inference request, created by `Model.submit()`.

**Methods:**
- `wait(timeout: float = 1.0) -> InferenceStats` - Block until complete, timeout in seconds (GIL released)

**Properties:**
- `request_id: int` - Proxy-assigned ID for log correlation

### Module Functions

- `proxy_endpoint() -> ProxyEndpoint` - Resolve the endpoint `Session.connect()` opens: `ARA2_SOCKET` if set, otherwise the first entry of `SOCKET_PATHS` that exists, otherwise a running proxy (UNIX before TCP)
- `discover() -> list[Proxy]` - Find every running ARA-2 proxy and where it listens

### Metadata Functions

- `read_metadata(path: str) -> DvmMetadata | None`
- `read_labels(path: str) -> list[str]`
- `has_metadata(path: str) -> bool`
- `read_model_format(path: str) -> int | None` - model format (2 for DVAPI 1.x runtimes, 3 for 3.x), `None` when not recognised

### Supporting Types

- **Proxy**: pid, endpoints, config, exe, `connect()` — a running proxy from `discover()`
- **ProxyEndpoint**: kind (`"unix"`/`"tcp"`), path, host, port, `connect()`
- **ComponentVersion**: component, version — one entry of `Session.versions()`
- **Abi** (enum): V1_1, V1_3, V3_0 — the DVAPI generation of the loaded `libaraclient`
- **State** (enum): Init, Idle, Active, ActiveSlow, ThermalUnknown, Inactive, Fault, Unknown, plus ActiveBoosted and ThermalInactive under DVAPI 1.1 or ThermalActiveSlow and FailSafe under DVAPI 1.3 and 3.0
- **ModelOutputType** (enum): Classification, Detection, SemanticSegmentation, Raw
- **DramStatistics**: dram_size, free_size, model_occupancy_size, ...
- **ModelTiming**: run_time, input_time, output_time (float seconds)
- **InferenceStats**: timing, clocks, hw_cycles, fp_cycles, input_start, inference_start, output_start, retries_session_total, retries_this_inference
- **Clocks**: sys_mhz, dram_mhz, nnp_mhz, sbp_mhz
- **EndpointStatistics**: state, clocks, core_voltage_v, temp_c, soft_reset_count, dram
- **EndpointInfo**: device_id, vendor_id, chip, dram, interface
- **ChipInfo**: id, revision, control_processors, neural_processors, l2_bytes
- **DramInfo**: vendor_name
- **Interface**: kind, bus, device, port, sysfs_path
- **LogLevel** (enum): Trace, Debug, Info, Warn, Error, Critical, Off
- **ModelType** (enum): Ara1Cnn, Ara2Cnn, Ara2Llm, Ara2LlmDynV2, McsLlm, Other
- **InterfaceKind** (enum): Pcie, Usb, Other
- **InputQuantization**: qn, scale, mean, is_signed
- **OutputQuantization**: qn, scale, offset, is_signed

### Exceptions

```
Ara2Error (RuntimeError)
 +-- LibraryError       - libaraclient loading failures
 +-- HardwareError      - NPU faults, endpoint errors
 +-- ProxyError         - Proxy connection failures
 +-- ModelError         - Model load/inference failures
 +-- TensorError        - Tensor allocation, DMA-BUF errors
 +-- MetadataError      - DVM metadata parsing errors
 +-- UnsupportedError   - Feature not provided by the loaded DVAPI version
```

Every exception has `code`, `name`, `sub_code`, `component`, `severity`, `model_format` and `compatibility` attributes, `None` where they do not apply. A model the client library cannot parse (code 520) raises `ModelError` with `model_format` and `compatibility` set.

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

### "libaraclient not found" (or a specific `libaraclient_*.so` variant)

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

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=67b03702-12df-456b-86f9-f246395421b5" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
