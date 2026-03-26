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

The `ara2` crate integrates with [`edgefirst-hal`](https://crates.io/crates/edgefirst-hal)
(enabled by default via the `hal` feature) for:

- **Tensor memory management** — DMA-backed tensors for zero-copy NPU transfers
- **Image preprocessing** — Hardware-accelerated format conversion and scaling
- **Post-processing** — YOLO decoding, overlay rendering, segmentation masks

Disable the `hal` feature for a minimal FFI-only build:

```bash
cargo build --no-default-features
```

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

Benchmarked on NXP i.MX 8M Plus + ARA-2 with YOLOv8n (640x640), showing
the Python API adds minimal overhead over native Rust thanks to DMA-BUF
zero-copy tensor sharing — the GPU and NPU operate on the same physical
buffers with no CPU copies in the data path.

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

> Steady-state mean over 20 iterations after warmup. The Python overhead
> is entirely in postprocessing (numpy array marshalling); GPU preprocessing
> and NPU inference are identical since both use the same DMA-BUF tensors.

## Examples

| Example | Description |
|---------|-------------|
| [`yolov8.rs`](examples/yolov8.rs) | Rust — YOLOv8 detection/segmentation with HAL pre/post-processing |
| [`yolov8.py`](examples/yolov8.py) | Python — YOLOv8 detection with DMA-BUF pipeline and HAL decoder |
| [`endpoints.py`](examples/endpoints.py) | Python — Connect, list endpoints, check status |
| [`test_dvm_metadata.rs`](examples/test_dvm_metadata.rs) | Rust — Read and display DVM model metadata |

Run examples:

```bash
# Rust
cargo run --release --example yolov8 -- model.dvm image.jpg --benchmark 20

# Python
python examples/yolov8.py model.dvm image.jpg --benchmark 20
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
