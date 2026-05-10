# Testing

This document describes how to test the ARA-2 client library.

## Overview

Most tests require ARA-2 hardware and a running proxy service. Only the
DVM metadata tests can run on any machine.

| Category | Hardware | Env Vars | Description |
|----------|----------|----------|-------------|
| `dvm_metadata` | No | None | Pure data parsing — runs in CI |
| `session` | Yes | None | Proxy connection tests |
| `endpoint` | Yes | None | NPU endpoint enumeration and status |
| `model` | Yes | `ARA2_TEST_MODEL` | Model loading, tensor allocation, inference |

## Prerequisites (on-target)

1. **ARA-2 hardware** visible via PCIe:
   ```bash
   lspci | grep -i kinara
   ```
2. **Proxy service** running (`dvproxy`, managed by a systemd unit such as
   `ara2.service` or `dvproxy.service` depending on the platform):
   ```bash
   systemctl status ara2    # EdgeFirst Yocto images
   systemctl status dvproxy # other platforms
   ```
3. **Proxy socket** available:
   ```bash
   ls -la /var/run/ara2.sock
   ```
4. **Client library** installed:
   ```bash
   ls /usr/lib/libaraclient.so.1
   ```

## Running Tests

### All tests (on-target)

```bash
cargo test -p ara2
```

### Metadata tests only (no hardware needed)

```bash
cargo test -p ara2 dvm_metadata
```

### Model tests (requires a .dvm file)

```bash
ARA2_TEST_MODEL=/path/to/model.dvm cargo test -p ara2 model
```

### With nextest

```bash
cargo nextest run -p ara2
```

### With debug logging

```bash
RUST_LOG=debug cargo test -p ara2 -- --nocapture
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ARA2_TEST_MODEL` | Path to a `.dvm` model file for model tests |
| `RUST_LOG` | Log level: `debug`, `info`, `warn`, `error` |

## Running Examples on Target

### Cross-compile and deploy

```bash
# Build Rust examples
cargo zigbuild --release --example async_infer --example async_pipeline \
  --target aarch64-unknown-linux-gnu

# Build Python wheel
maturin build --release -m crates/ara2-py/Cargo.toml \
  --zig --target aarch64-unknown-linux-gnu --compatibility manylinux2014

# Deploy
scp target/aarch64-unknown-linux-gnu/release/examples/async_infer \
    target/aarch64-unknown-linux-gnu/release/examples/async_pipeline <target>:/tmp/
scp target/wheels/edgefirst_ara2-*.whl <target>:/tmp/
scp examples/async_infer.py examples/async_pipeline.py <target>:/tmp/
```

### Run on target

```bash
# Rust — basic async benchmark
ssh <target> /tmp/async_infer /root/models/yolov8n_640x640.dvm 10

# Rust — pipelined inference with circular buffer (depth=2)
ssh <target> /tmp/async_pipeline /root/models/yolov8n_640x640.dvm 50 2

# Python
ssh <target> 'pip install --force-reinstall --no-deps /tmp/edgefirst_ara2-*.whl && \
  python3 /tmp/async_infer.py /root/models/yolov8n_640x640.dvm 10'

# Python — pipelined
ssh <target> 'python3 /tmp/async_pipeline.py /root/models/yolov8n_640x640.dvm 50 2'
```

## CI

The GitHub Actions workflows run the following checks:

| Workflow | What runs |
|----------|-----------|
| `test.yml` | `cargo fmt --check`, `cargo clippy`, `cargo test -p ara2 dvm_metadata` |
| `build.yml` | Release build for x86_64 and aarch64 |
| `python.yml` | Python wheel build with maturin + zig |

Hardware-dependent tests (`session`, `endpoint`, `model`) do not run in
CI — they require a self-hosted runner with ARA-2 hardware.

## Debugging

### Verifying hardware setup

```bash
# Check PCIe device
lspci | grep -i kinara

# Check proxy service (dvproxy — service name varies by platform)
systemctl status ara2    # EdgeFirst Yocto images
systemctl status dvproxy # other platforms
journalctl -u ara2 --no-pager -n 50

# Check socket
ls -la /var/run/ara2.sock
```

### Common error codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | `DV_SUCCESS` | Operation succeeded |
| 1 | `DV_FAILURE_UNKNOWN` | Unknown failure |
| 100 | `DV_ENDPOINT_OUT_OF_MEMORY` | NPU DRAM full — unload models |
| 200 | `DV_RESOURCE_NOT_FOUND` | Invalid handle or missing resource |
| 220 | `DV_SESSION_UNIX_SOCKET_FILE_TOO_LONG` | Socket path exceeds limit |
| 230 | `DV_ENDPOINT_INVALID_HANDLE` | Stale endpoint reference |
| 240+ | `DV_MODEL_*` | Model loading/inference errors |
| 300+ | `DV_ERROR_CATEGORY_SW_CLIENT_FATAL` | Client library crash |
| 400+ | `DV_ERROR_CATEGORY_SW_SERVER_FATAL` | Proxy crash |
| 500+ | `DV_ERROR_CATEGORY_HW_FATAL` | Hardware failure |
