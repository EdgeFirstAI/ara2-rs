# ARA-2 Client Library

[![CI](https://github.com/EdgeFirstAI/ara2-rs/actions/workflows/test.yml/badge.svg)](https://github.com/EdgeFirstAI/ara2-rs/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![crates.io](https://img.shields.io/crates/v/ara2.svg)](https://crates.io/crates/ara2)

Rust client library for the [Kinara](https://kinara.ai) ARA-2 neural network
accelerator. Provides session management, model loading, and inference on
NXP i.MX platforms equipped with ARA-2 PCIe hardware.

## Supported Platforms

| Platform | SoC | Status |
|----------|-----|--------|
| [NXP FRDM i.MX 8M Plus](https://www.nxp.com/design/design-center/development-boards-and-designs/FRDM-IMX8MPLUS) | i.MX 8M Plus | Tested |
| [NXP FRDM i.MX 95](https://www.nxp.com/design/design-center/development-boards-and-designs/i-mx-evaluation-and-development-boards/freedom-development-platform-for-i-mx-95:FRDM-IMX95) | i.MX 95 | Tested |

Requires [EdgeFirst Yocto Images](https://github.com/EdgeFirstAI/yocto) with ARA-2 SDK support.

## Workspace

| Crate | Description |
|-------|-------------|
| [`ara2`](crates/ara2) | Core client library — session, endpoint, model, and DVM metadata APIs |
| [`ara2-sys`](crates/ara2-sys) | FFI bindings to `libaraclient` via `libloading` |

### Integration with the EdgeFirst HAL

The EdgeFirst HAL ships as one crate per library rather than a single
`edgefirst-hal` crate. The `ara2` crate depends on two of them:

| Crate | Used for |
|-------|----------|
| [`edgefirst-tensor`](https://crates.io/crates/edgefirst-tensor) | Tensor memory management — DMA-backed tensors for zero-copy NPU transfers |
| [`edgefirst-image`](https://crates.io/crates/edgefirst-image) | Image preprocessing — hardware-accelerated format conversion, scaling, and overlay rendering |

Post-processing and image loading are the application's concern, not the
client library's, so the examples take them as their own dependencies:

| Crate | Used for |
|-------|----------|
| [`edgefirst-decoder`](https://crates.io/crates/edgefirst-decoder) | Post-processing — YOLO decoding, NMS, segmentation masks |
| [`edgefirst-codec`](https://crates.io/crates/edgefirst-codec) | JPEG/PNG decode straight into a pre-allocated tensor |

Two off-by-default features opt back into the HAL types they contribute to
`ara2`'s own API: `decoder` adds `OutputSpec::dshape_typed()`, returning
`edgefirst_decoder::configs::DimName` pairs rather than the metadata's raw
axis names, and `codec` adds `Error::Codec` with its
`From<edgefirst_codec::CodecError>` conversion. Both are additive: a feature
may add to this crate's API, never reshape it.

### Python Bindings

Python bindings are available as a separate package via PyPI:

```bash
pip install edgefirst-ara2
```

See [`crates/ara2-py/README.md`](crates/ara2-py/README.md) for the Python API reference.

## Quick Start

```rust
use ara2::Session;
use edgefirst_tensor::{TensorMemory, TensorTrait as _};

// Connect to the ARA-2 proxy service
let session = Session::connect()?;

// Enumerate NPU endpoints and check status
let endpoints = session.list_endpoints()?;
let endpoint = &endpoints[0];
println!("Endpoint state: {:?}", endpoint.check_status()?);

// Load a compiled model (.dvm) and allocate DMA tensors
let mut model = endpoint.load_model_from_file("model.dvm".as_ref())?;
model.allocate_tensors(Some(TensorMemory::DmaBuf))?;

// Run inference
let timing = model.run()?;
println!("NPU inference: {:?}", timing.run_time);
# Ok::<(), ara2::Error>(())
```

## Async Inference

The `submit()` / `wait()` API enables overlapping CPU work with NPU
execution — the building block for pipeline parallelism:

```rust
use ara2::{Session, DEFAULT_TIMEOUT_MS};

let session = Session::connect()?;
let endpoints = session.list_endpoints()?;
let mut model = endpoints[0].load_model_from_file("model.dvm".as_ref())?;
model.allocate_tensors(None)?;

// Submit — returns immediately while the NPU works
let request = model.submit()?;

// CPU is free to do other work (preprocess next frame, etc.)

// Block until the NPU finishes
let timing = request.wait(DEFAULT_TIMEOUT_MS)?;
println!("NPU inference: {:?}", timing.run_time);

// Monitor pipeline depth
assert_eq!(session.inflight_count()?, 0);
# Ok::<(), ara2::Error>(())
```

The Python API mirrors this exactly:

```python
import edgefirst_ara2 as ara2

session = ara2.Session.connect()
endpoint = session.list_endpoints()[0]
model = endpoint.load_model("model.dvm")
model.allocate_tensors()

# Submit — returns immediately
request = model.submit()

# CPU work here... the GIL is NOT held during wait()
timing = request.wait()
print(f"NPU inference: {timing.run_time_us} µs")
```

See the [`async_infer`](examples/async_infer.rs) example for a complete
benchmark comparing synchronous vs. asynchronous inference, and
[`async_pipeline`](examples/async_pipeline.rs) for pipelined inference
with a circular buffer of DMA-BUF tensor sets (2x+ throughput improvement).

## Runtime Requirements

Requires NXP's rt-sdk-ara2 SDK 2.1.1 or later. The following must be present
on the target system:

- **`libaraclient`** — Kinara client library, from NXP's `imx-nxp-ara2`
  package (rt-sdk-ara2 SDK 2.1.1+). NXP's packaging has changed the shared
  library's name in every SDK drop so far, so [`open_library`](crates/ara2/src/lib.rs)
  tries the architecture-suffixed name for the target first
  (`libaraclient_aarch64.so`, `libaraclient_x86_64.so`), then
  `libaraclient.so` and `libaraclient.so.1`.
- **The proxy service** — provides NPU access, must be running before
  connecting (`rt-sdk-ara2.service` on EdgeFirst Yocto images shipping
  NXP's rt-sdk-ara2 integration; listens on `/var/run/proxy.sock` by
  default -- see [`DEFAULT_SOCKET`](crates/ara2/src/lib.rs) and the
  `ARA2_SOCKET` environment variable for overriding this)
- **ARA-2 hardware** — PCIe accelerator card visible via `lspci`

## Building

### Native

```bash
cargo build --release
```

### Cross-compile for aarch64 (NXP i.MX)

```bash
cargo zigbuild --release --target aarch64-unknown-linux-gnu
```

## Performance

Benchmarked on NXP FRDM i.MX 95 + ARA-2 with YOLOv8m-seg (640×640),
showing the Python API adds minimal overhead over native Rust thanks to
DMA-BUF zero-copy tensor sharing — the GPU and NPU operate on the same
physical buffers with no CPU copies in the data path.

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
> under 1 ms across the entire pipeline. GPU preprocessing and NPU inference
> are identical since both use the same DMA-BUF tensors.

## Examples

| Example | Description |
|---------|-------------|
| [`yolov8.rs`](examples/yolov8.rs) | Rust — YOLOv8 detection + segmentation with letterbox preprocessing and 3-step mask pipeline (needs `--features decoder`) |
| [`yolov8.py`](examples/yolov8.py) | Python — Same 3-step pipeline via the `edgefirst.*` and `edgefirst-ara2` Python packages |
| [`yolov8_live.rs`](examples/yolov8_live.rs) | Rust — Live camera inference: libcamera capture → NPU → Wayland display (needs `--features camera`) |
| [`yolov8_live.py`](examples/yolov8_live.py) | Python — Same live pipeline via the libcamera Python bindings and pywayland |
| [`async_multi_model.rs`](examples/async_multi_model.rs) | Rust — Two models in flight concurrently on one endpoint |
| [`async_multi_model.py`](examples/async_multi_model.py) | Python — Same multi-model demo via `edgefirst-ara2` |
| [`async_infer.rs`](examples/async_infer.rs) | Rust — Async inference benchmark: sync vs. submit/wait vs. overlap |
| [`async_infer.py`](examples/async_infer.py) | Python — Same async benchmark via `edgefirst-ara2` |
| [`async_pipeline.rs`](examples/async_pipeline.rs) | Rust — Pipelined inference with circular DMA-BUF buffer ring (2x+ speedup) |
| [`async_pipeline.py`](examples/async_pipeline.py) | Python — Same pipeline demo via `edgefirst-ara2` |
| [`endpoints.py`](examples/endpoints.py) | Python — Connect, list endpoints, check status |
| [`test_dvm_metadata.rs`](examples/test_dvm_metadata.rs) | Rust — Read and display DVM model metadata |

### Models

The `yolov8` examples run the official pre-trained models published in the
EdgeFirst model zoo on Hugging Face:

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

Every zoo export embeds an `edgefirst.json` schema and the class `labels.txt`
in a ZIP trailer appended to the `.dvm`, which `ara2::dvm_metadata` reads.
That is what supplies the two things the NPU runtime cannot report: the
per-output `normalized` flag and the named `dshape`. A `.dvm` without the
trailer still runs — the examples fall back to the built-in COCO labels and a
canonical Ultralytics axis naming — but a model whose boxes are in pixel
space will be misread without it.

Both `yolov8` examples pin `DecoderVersion::Yolov8`, so they decode the
`yolov8-det` and `yolov8-seg` repositories; the zoo's `yolo11` and `yolo26`
repositories ship `ara240/` builds too, but need the matching decoder
version.

### Running the Rust example

Cross-compile from your development machine and deploy to the target:

```bash
# Build (`decoder` is off by default; the example needs the HAL decoder types)
cargo zigbuild --release --features decoder --example yolov8 \
    --target aarch64-unknown-linux-gnu

# Deploy and run
scp target/aarch64-unknown-linux-gnu/release/examples/yolov8 <target>:/root/yolov8-ara2
ssh <target> "/root/yolov8-ara2 yolov8n-seg-int16.dvm zidane.jpg --benchmark 30 --save"
```

### Running the Python example

Create a virtual environment on the target and install the packages from PyPI:

```bash
# On target
python3 -m venv ~/venv
~/venv/bin/pip install edgefirst-ara2 'edgefirst-codec>=0.32' \
    'edgefirst-decoder>=0.32' 'edgefirst-image>=0.32'
```

Copy the script and run:

```bash
# From dev machine
scp examples/yolov8.py <target>:/root/

# On target
~/venv/bin/python3 /root/yolov8.py yolov8n-seg-int16.dvm zidane.jpg --benchmark 30 --save
```

## Testing

Tests require an NXP i.MX + ARA-2 system with the proxy running:

```bash
# All tests (on-target with hardware)
cargo test -p ara2

# Metadata tests only (no hardware needed)
cargo test -p ara2 dvm_metadata

# Model tests (needs a .dvm file)
ARA2_TEST_MODEL=/path/to/model.dvm cargo test -p ara2 model
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and ownership model
- [TESTING.md](TESTING.md) — Test guide, on-target setup, and debugging
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines
- [SECURITY.md](SECURITY.md) — Security policy
- [CHANGELOG.md](CHANGELOG.md) — Release history

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 Au-Zone Technologies. All Rights Reserved.

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=67b03702-12df-456b-86f9-f246395421b5" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
