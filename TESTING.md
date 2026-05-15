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
  --example async_multi_model --target aarch64-unknown-linux-gnu

# Build Python wheel
maturin build --release -m crates/ara2-py/Cargo.toml \
  --zig --target aarch64-unknown-linux-gnu --compatibility manylinux2014

# Deploy
scp target/aarch64-unknown-linux-gnu/release/examples/async_infer \
    target/aarch64-unknown-linux-gnu/release/examples/async_pipeline \
    target/aarch64-unknown-linux-gnu/release/examples/async_multi_model <target>:/tmp/
scp target/wheels/edgefirst_ara2-*.whl <target>:/tmp/
scp examples/async_infer.py examples/async_pipeline.py \
    examples/async_multi_model.py <target>:/tmp/
```

### Run on target

```bash
# Rust — basic async benchmark
ssh <target> /tmp/async_infer /root/models/yolov8n_640x640.dvm 10

# Rust — pipelined inference with circular buffer (depth=2)
ssh <target> /tmp/async_pipeline /root/models/yolov8n_640x640.dvm 50 2

# Rust — multi-model (dual + A/B alternating)
ssh <target> /tmp/async_multi_model \
  /root/models/yolov8n-seg.dvm /root/models/yolo11n-seg.dvm 50

# Python
ssh <target> 'pip install --force-reinstall --no-deps /tmp/edgefirst_ara2-*.whl && \
  python3 /tmp/async_infer.py /root/models/yolov8n_640x640.dvm 10'

# Python — pipelined
ssh <target> 'python3 /tmp/async_pipeline.py /root/models/yolov8n_640x640.dvm 50 2'

# Python — multi-model
ssh <target> 'python3 /tmp/async_multi_model.py \
  /root/models/yolov8n-seg.dvm /root/models/yolo11n-seg.dvm 50'
```

## Async Processing Benchmarks

The async inference API (`submit()`/`wait()`) enables pipelined execution
where the CPU and NPU work in parallel. Three example benchmarks
demonstrate increasing levels of overlap.

### Examples Overview

| Example | Pattern | Description |
|---------|---------|-------------|
| `async_infer` | Single model | Basic submit/wait vs sync comparison |
| `async_pipeline` | Single model × N slots | Circular DMA-BUF buffer ring — full CPU/NPU overlap |
| `async_multi_model` | Two different models | Dual-model and A/B alternating scheduling |

### Cross-compile and deploy all benchmarks

```bash
# Build all async examples
cargo zigbuild --release --target aarch64-unknown-linux-gnu \
  --example async_infer --example async_pipeline --example async_multi_model

# Deploy to target
scp target/aarch64-unknown-linux-gnu/release/examples/async_infer \
    target/aarch64-unknown-linux-gnu/release/examples/async_pipeline \
    target/aarch64-unknown-linux-gnu/release/examples/async_multi_model \
    <target>:/tmp/

# Deploy Python examples
scp examples/async_infer.py examples/async_pipeline.py \
    examples/async_multi_model.py <target>:/tmp/
```

### Single-model pipeline (`async_pipeline`)

Uses a circular buffer of N model slots, each with its own DMA-BUF
tensors. While the NPU executes on slot N, the CPU fills slot N+1 and
reads results from slot N−1.

```bash
# Rust (100 iterations, depth=3)
ssh <target> /tmp/async_pipeline /root/models/yolov8n-seg.dvm 100 3

# Python
ssh <target> python3 /tmp/async_pipeline.py /root/models/yolov8n-seg.dvm 100 3
```

**Reference results** (yolov8n-seg, 100 iterations):

| Platform | Sync fps | Pipeline (depth=3) fps | Speedup |
|----------|----------|------------------------|---------|
| imx8mp (Cortex-A53 + ara240) | 68 | 143 | 2.1× |
| imx95 (Cortex-A55 + ara2400) | 69 | 216 | 3.1× |

### Multi-model benchmarks (`async_multi_model`)

Loads two different models on the same endpoint and benchmarks two
scheduling patterns:

```bash
# Rust (100 iterations)
ssh <target> /tmp/async_multi_model \
  /root/models/yolov8n-seg.dvm /root/models/yolo11n-seg.dvm 100

# Python
ssh <target> python3 /tmp/async_multi_model.py \
  /root/models/yolov8n-seg.dvm /root/models/yolo11n-seg.dvm 100
```

#### Dual-model: same image → both models

Every frame is processed by model A **and** model B. The async variant
submits both requests then waits, overlapping model A's output DMA with
model B's input transfer and queuing.

**Reported FPS is per-frame** — each frame produces 2 inferences, so
the total inference rate is 2× the reported number.

| Platform | Sync fps | Async fps | Speedup | Total inferences/sec |
|----------|----------|-----------|---------|---------------------|
| imx8mp | 33 | 46 | 1.39× | 92 |
| imx95 | 35 | 53 | 1.51× | 105 |

#### A/B alternating: even frames → A, odd frames → B

Each model acts as a natural double-buffer — while model A runs on the
NPU for an even frame, model B's inputs are filled by the CPU for the
next odd frame.

**Reported FPS is total inference rate** — each individual model
processes half the frames.

| Platform | Sync fps | Async fps | Speedup | Per-model fps |
|----------|----------|-----------|---------|--------------|
| imx8mp | 67 | 112 | 1.67× | ~56 |
| imx95 | 69 | 143 | 2.07× | ~71 |

### Interpreting results

- **Sync throughput is nearly identical** across platforms (~34 fps dual,
  ~69 fps alternating) because the NPU inference time dominates.
- **Async speedup scales with CPU speed** — the faster Cortex-A55 cores
  on imx95 reduce fill/read time, allowing more overlap with NPU execution.
- **A clean run ends with `inflight: 0`**. If the inflight count is
  non-zero, a request was not properly waited on or dropped.
- **Proxy disconnect messages** (`session got disconnected`) at exit are
  normal — the session cleans up when the process terminates.

### Python benchmarks

The Python wheel must be installed on the target before running Python
examples:

```bash
# Build wheel
maturin build --release -m crates/ara2-py/Cargo.toml \
  --zig --target aarch64-unknown-linux-gnu --compatibility manylinux2014

# Install on target
scp target/wheels/edgefirst_ara2-*.whl <target>:/tmp/
ssh <target> pip install --force-reinstall --no-deps /tmp/edgefirst_ara2-*.whl
```

Python async benchmarks release the GIL during `wait()`, so NPU
execution overlaps with other Python threads. Expect ~30-40% lower
throughput than Rust due to GIL acquisition for `set_input_tensor()` and
`get_output_tensor()` numpy copies.

## CI

The GitHub Actions workflows run the following checks:

| Workflow | What runs |
|----------|-----------|
| `test.yml` | `cargo fmt --all -- --check`, `cargo clippy --workspace --exclude ara2-py --all-targets -- -D warnings`, `cargo nextest run -p ara2 -E 'test(dvm_metadata)'` |
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
