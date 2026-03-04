# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EdgeFirstAI/ara2-rs/releases/tag/v0.1.0
