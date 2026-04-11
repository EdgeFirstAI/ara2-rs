# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-11

### Added

- **Rust live-camera example** (`examples/yolov8_live.rs`) — libcamera
  capture, zero-copy DMA-BUF tensor input, and direct Wayland DMA-BUF
  display for real-time YOLOv8 inference on NXP i.MX platforms.
- **3-step segmentation pipeline** — split preprocessing, inference, and
  postprocessing into discrete HAL steps; reflected in both Rust and
  Python YOLOv8 examples.
- **`camera` Cargo feature** on the `ara2` crate that gates the
  libcamera-based `yolov8_live` example behind optional dependencies
  (`libcamera`, `wayland-client`, `wayland-protocols`). Building the
  library and file-based examples no longer requires libcamera on the host.
- Monolithic YOLO detection decoder path for models with a single
  `[1, nc+4, N]` output tensor (previously only the split boxes+scores
  layout was supported).
- `--format {nv12,yuyv}` CLI flag on live-camera examples for pixel-format
  performance comparison (YUYV is ~1.3 ms faster than NV12 on imx95-frdm).
- `--color-mode {class,instance,track}` on all four YOLOv8 examples
  (previously hardcoded to Instance).
- `--socket` flag on `examples/yolov8.rs` for parity with the other examples.
- Comprehensive API documentation (doc comments / docstrings) across all
  example files.

### Changed

- **Live display backend:** replaced EGL/GL with direct Wayland
  `zwp_linux_dmabuf_v1` submission — no OpenGL context required, zero-copy
  from NPU output to compositor.
- **Python camera capture:** replaced GStreamer with native libcamera
  Python bindings in `yolov8_live.py` for lower latency and fewer
  transitive dependencies.
- **Example CLIs:** migrated Rust YOLOv8 examples to `clap` derive with
  per-variant `--help` descriptions and typo suggestions.

### Fixed

- **YOLO detection box mapping in `examples/yolov8.rs`:** decoder produces
  normalized coordinates in the letterboxed model input frame (e.g. 640×640),
  not the original image. Un-pad and rescale by `1 / letterbox_scale` so
  boxes are no longer stretched on non-square source images.

## [0.2.0] - 2026-03-26

### Added

- **Python API** (`edgefirst-ara2` on PyPI) — complete PyO3 bindings with:
  - Session, Endpoint, and Model wrappers with full API parity
  - numpy tensor I/O (`set_input_tensor`, `get_output_tensor`, `dequantize`)
  - DMA-BUF file descriptor access (`input_tensor_fd`, `output_tensor_fd`) for
    zero-copy GPU preprocessing with `edgefirst-hal.import_image()`
  - DVM metadata API (`read_metadata`, `read_labels`, `has_metadata`)
  - Python exception hierarchy (`Ara2Error` → `LibraryError`, `HardwareError`,
    `ProxyError`, `ModelError`, `TensorError`, `MetadataError`)
  - Bounds checking on all tensor index accessors (raises `IndexError`)
  - Allocation guards (`run()` before `allocate_tensors()` raises `TensorError`)
  - Context manager support (`with` statement) on Session and Model
  - `os.PathLike` support on all path parameters
  - Comprehensive `.pyi` type stubs with docstrings
- Python YOLOv8 example (`examples/yolov8.py`) with DMA-BUF pipeline,
  HAL decoder integration, and `--benchmark` mode
- Rust YOLOv8 `--benchmark` mode with matching output format
- PyPI publishing via GitHub Actions with OIDC trusted publishing
- `python.yml` CI workflow for building manylinux2014 wheels (x86_64, aarch64)

### Changed

- Upgraded `edgefirst-hal` from 0.11.0 to 0.13.0
- Migrated Rust YOLOv8 example to HAL 0.13 `import_image` / `PlaneDescriptor` API
- Upgraded `pyo3` from 0.23 to 0.24, added `numpy` 0.24
- Release workflow now builds and publishes Python wheels alongside Rust crates
- Python version derived from `Cargo.toml` via `dynamic = ["version"]`
- Updated all GitHub Action hashes to latest versions (checkout v6.0.2,
  upload-artifact v7.0.0, rust-cache v2.9.1, install-action v2.69.10)
- Updated `examples/endpoints.py` to use `edgefirst_ara2` module name

### Fixed

- Release workflow tag patterns (glob-style `v[0-9]*` instead of regex `v[0-9]+`)
- Release workflow SBOM collection (recursive find for nested artifact paths)
- Rust YOLOv8 args parsing bounds check (prevents panic on missing flag value)

## [0.1.3] - 2026-03-09

### Changed

- Updated `edgefirst-hal` dependency to 0.9.0

## [0.1.2] - 2026-03-03

### Fixed

- SBOM artifact path (per-crate `crates/*/bom.json` instead of root `bom.json`)
- Deduplicated SBOM generation (release workflow reuses `sbom.yml` via `workflow_call`)
- Removed flaky path filters from CI workflows

## [0.1.1] - 2026-03-03

### Fixed

- CString null-termination for UNIX and TCP socket FFI calls
- Endpoint list memory leak (added `EndpointList` with proper `Drop`)
- `output_info()` and `output_quants()` now return `Result` instead of panicking
- Feature-gate `image` dependency behind `hal` feature
- Error source chaining via `std::error::Error::source()`

### Added

- GitHub Actions CI/CD workflows (test, build, SBOM, release)
- Trusted publishing to crates.io via OIDC
- Configurable inference timeout (`set_timeout_ms()`, `DEFAULT_TIMEOUT_MS`)
- `Debug` impls for `Session`, `Endpoint`, and `Model`
- `DEFAULT_SOCKET` public constant
- Public `input_info()` and `output_info()` methods on `Model`
- Re-exports for `DvmMetadata` sub-types and `InputTensor`/`OutputTensor`
- Rustdoc for `DramStatistics`, `InputTensor`, and `OutputTensor` fields
- YOLOv8 detection/segmentation example with edgefirst-hal integration
- `ara2-sys` README and crates.io metadata (keywords, categories)

### Changed

- Switched from nightly to stable Rust toolchain (edition 2024)
- Cross-compilation uses zigbuild instead of `.cargo/config.toml`

## [0.1.0] - 2025-02-02

### Added

- Initial public release of ARA2 client library
- Core Rust library (`ara2` crate) with support for:
  - UNIX socket connections to ARA-2 proxy
  - TCP/IPv4 socket connections
  - Endpoint enumeration and status monitoring
  - Model loading and inference execution
  - DRAM statistics and performance timing
- Python bindings (`edgefirst-ara2` package) with:
  - PyO3-based bindings using stable ABI (Python 3.11+)
  - Full API parity with Rust library
  - Type stubs for IDE support
- FFI layer (`ara2-sys` crate) for libaraclient.so integration
- Documentation:
  - README with quick start guide
  - Python-specific documentation (PYTHON.md)
  - Example code for Rust and Python

### Dependencies

- Requires `edgefirst-hal` for HAL integration
- Requires `libaraclient.so` runtime library

[Unreleased]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EdgeFirstAI/ara2-rs/releases/tag/v0.1.0
