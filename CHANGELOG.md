# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/EdgeFirstAI/ara2-rs/releases/tag/v0.1.0
