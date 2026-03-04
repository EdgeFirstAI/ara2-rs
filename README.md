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
| EdgeFirst FRDM | NXP i.MX 8M Plus | Tested |
| EdgeFirst FRDM | NXP i.MX 95 | Tested |

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
cargo build --release --target aarch64-unknown-linux-gnu
```

The cross-linker is configured in `.cargo/config.toml`.

## Examples

| Example | Description |
|---------|-------------|
| [`yolov8`](examples/yolov8.rs) | YOLOv8 detection/segmentation with edgefirst-hal pre/post-processing |
| [`test_dvm_metadata`](examples/test_dvm_metadata.rs) | Read and display DVM model metadata |

Run an example:

```bash
cargo run --release --example yolov8 -- model.dvm image.jpg --save
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
