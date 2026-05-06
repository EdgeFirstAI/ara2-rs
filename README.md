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
| [`ara2-sys`](crates/ara2-sys) | FFI bindings to `libaraclient.so` via `libloading` |

### Integration with edgefirst-hal

The `ara2` crate depends on [`edgefirst-hal`](https://crates.io/crates/edgefirst-hal)
for:

- **Tensor memory management** — DMA-backed tensors for zero-copy NPU transfers
- **Image preprocessing** — Hardware-accelerated format conversion and scaling
- **Post-processing** — YOLO decoding, overlay rendering, segmentation masks

### Python Bindings

Python bindings are available as a separate package via PyPI:

```bash
pip install edgefirst-ara2
```

See [`crates/ara2-py/README.md`](crates/ara2-py/README.md) for the Python API reference.

## Quick Start

```rust
use ara2::{Session, DEFAULT_SOCKET};
use edgefirst_hal::tensor::{TensorMemory, TensorTrait as _};

// Connect to the ARA-2 proxy service
let session = Session::create_via_unix_socket(DEFAULT_SOCKET)?;

// Enumerate NPU endpoints and check status
let endpoints = session.list_endpoints()?;
let endpoint = &endpoints[0];
println!("Endpoint state: {:?}", endpoint.check_status()?);

// Load a compiled model (.dvm) and allocate DMA tensors
let mut model = endpoint.load_model_from_file("model.dvm".as_ref())?;
model.allocate_tensors(Some(TensorMemory::Dma))?;

// Run inference
let timing = model.run()?;
println!("NPU inference: {:?}", timing.run_time);
# Ok::<(), ara2::Error>(())
```

## Runtime Requirements

The following must be present on the target system:

- **`libaraclient.so.1`** — Kinara client library (from the ARA-2 SDK)
- **`ara2-proxy`** — System service providing NPU access, must be running
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
| [`yolov8.rs`](examples/yolov8.rs) | Rust — YOLOv8 detection + segmentation with letterbox preprocessing and 3-step mask pipeline |
| [`yolov8.py`](examples/yolov8.py) | Python — Same 3-step pipeline via `edgefirst-hal` and `edgefirst-ara2` Python packages |
| [`endpoints.py`](examples/endpoints.py) | Python — Connect, list endpoints, check status |
| [`test_dvm_metadata.rs`](examples/test_dvm_metadata.rs) | Rust — Read and display DVM model metadata |

### Running the Rust example

Cross-compile from your development machine and deploy to the target:

```bash
# Build
cargo zigbuild --release --example yolov8 --target aarch64-unknown-linux-gnu

# Deploy and run
scp target/aarch64-unknown-linux-gnu/release/examples/yolov8 <target>:/root/yolov8-ara2
ssh <target> "/root/yolov8-ara2 model.dvm image.jpg --benchmark 30 --save"
```

### Running the Python example

Create a virtual environment on the target and install the packages from PyPI:

```bash
# On target
python3 -m venv ~/venv
~/venv/bin/pip install edgefirst-ara2 edgefirst-hal
```

Copy the script and run:

```bash
# From dev machine
scp examples/yolov8.py <target>:/root/

# On target
~/venv/bin/python3 /root/yolov8.py model.dvm image.jpg --benchmark 30 --save
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
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines
- [SECURITY.md](SECURITY.md) — Security policy
- [CHANGELOG.md](CHANGELOG.md) — Release history

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2025 Au-Zone Technologies. All Rights Reserved.
