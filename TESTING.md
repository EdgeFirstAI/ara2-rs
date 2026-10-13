# Testing

This document describes how to test the ARA-2 client library.

## Overview

Most tests require ARA-2 hardware and a running proxy service. Tests that run anywhere are named `dvm_metadata` or `offline` (module `offline_tests`), which is what CI selects.

| Category | Hardware | Env Vars | Description |
|----------|----------|----------|-------------|
| `dvm_metadata` | No | None | Pure data parsing — runs in CI |
| `offline` | No | None | ABI selection, `ARA2_SOCKET` parsing, endpoint choice, struct reads per version, status and version mapping — runs in CI |
| `session` | Yes | None | Proxy connection, transport, versions, status names |
| `endpoint` | Yes | None | NPU endpoint enumeration and status |
| `model` | Yes | `ARA2_TEST_MODEL` | Model loading, layout sanity, tensor allocation, fd passing, inference |
| `verify_sdk` (ara2-sys) | No | `ARA2_SDK_V*` | Declarations against SDK headers — skipped without an SDK |

A new test that needs no hardware must be named `offline` (or sit in an `offline_tests` module) for CI to run it.

## Prerequisites (on-target)

1. **ARA-2 hardware** visible via PCIe:
   ```bash
   lspci | grep -i kinara
   ```
2. **Proxy service** running: `rt-sdk-ara2.service` under NXP's packaging, `ara2.service` under meta-kinara's, or whatever unit the board's r3.0 install uses.
3. **Proxy endpoint** reachable. `Session::connect()` finds it without configuration (see the README's "Connecting"); set `ARA2_SOCKET` to pin one.
4. **Client library** installed (one of `ara2::LIBRARY_NAMES`), on `LD_LIBRARY_PATH` if it is not in a system directory.

## Board matrix

| Board and runtime | DVAPI | Library | Proxy transport | Coverage |
|-------------------|-------|---------|-----------------|----------|
| i.MX 8M Plus FRDM with Kinara runtime 1.2.1 | 1.1 | system | UNIX `/var/run/ara2.sock` | SDK verifier and offline layout tests only for this release |
| i.MX 95 board with NXP rt-sdk-ara2 2.1.1 | 1.3 | `/usr/lib/libaraclient_aarch64.so` | UNIX `/var/run/proxy.sock` | Full suite and Python tests |
| i.MX 95 FRDM with arart r3.0 | 3.0 | `LD_LIBRARY_PATH=/opt/ara2/art/linux/aarch64/lib` | TCP `127.0.0.1:5000` by default, UNIX with `interface_type: "SOCKET"` | Full suite and Python tests over TCP |

Run the hardware suite single-threaded, since the tests share one NPU.

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

Any compiled `.dvm` works. The EdgeFirst model zoo publishes ARA-2 builds
under `ara240/`:

```bash
curl -LO https://huggingface.co/EdgeFirst/yolov8-det/resolve/main/ara240/yolov8n-det-int16.dvm
curl -LO https://huggingface.co/EdgeFirst/yolov8-seg/resolve/main/ara240/yolov8n-seg-int16.dvm
```

### Transport and fd-passing tests

These tests check their own expectations against the transport the session is on, so the same suite passes over UNIX and TCP:

- `session::tests::connect_uses_the_resolved_transport` — `Session::connect()` lands on the transport `ara2::proxy_endpoint()` resolved, and `supports_fd_passing()` matches it.
- `model::tests::fd_tensors_follow_the_transport` — over TCP, `TensorMemory::DmaBuf` and `TensorMemory::Shm` fail with `Error::FdPassingUnavailable` in `allocate_tensors`, `allocate_input_set` and `allocate_output_set`, and heap inference runs. Over UNIX, DMA-BUF and SHM inference on a fixed input pattern produce outputs byte-identical to heap inference. With `--nocapture` it prints which branch it took.
- `model::tests::shm_registrations_are_released`, `dmabuf_request_dropped_before_completion_is_safe`, `registration_outlives_dropped_set_while_pending` — exercise shared-memory registration over UNIX; over TCP they check the refusal and stop.
- `model::tests::model_layout_reads_are_sane` — every model field read through the version layout is in range and consistent, and input 0 matches the `.dvm` metadata's size and channels where the file has them. A wrong layout reads garbage here rather than failing a call.

To cover both branches on one board, run the suite once per transport. On an r3.0 board whose proxy listens on TCP, switch its configuration to a UNIX socket temporarily, with the board owner's agreement.

Where the proxy's unit runs chip bring-up before starting the proxy (as the r3.0 install on the i.MX 95 FRDM does), apply each configuration change with a power cycle. That bring-up only succeeds on a cold chip: `systemctl restart` and a warm reboot both leave the chip initialised, the bring-up fails, and the proxy stays down until the board is powered off and on.

```bash
cp /opt/ara2/proxy_run.yaml /opt/ara2/proxy_run.yaml.tcp-backup
# In the proxy: section set
#     interface_type: "SOCKET"
#     interface_socket_file: ["/var/run/proxy.sock"]
poweroff                      # then cut and restore board power
# After boot, wait for: systemctl is-active <proxy unit> && test -S /var/run/proxy.sock
# Run the suite with no ARA2_SOCKET; connect() must pick /var/run/proxy.sock.
mv /opt/ara2/proxy_run.yaml.tcp-backup /opt/ara2/proxy_run.yaml
poweroff                      # then cut and restore board power
# After boot: systemctl is-active <proxy unit>
grep ':1388' /proc/net/tcp    # port 5000 listening again
```

On an install whose unit does no chip bring-up and only starts the proxy, `systemctl restart <proxy unit>` in place of each power cycle is enough.

### SDK verifier

`crates/ara2-sys/tests/verify_sdk.rs` checks ara2-sys's declarations against SDK headers. Point it at each SDK's include directory (the one holding `dvapi.h`):

```bash
ARA2_SDK_V1_1=<runtime-1.2.1>/art/linux/x86/include \
ARA2_SDK_V1_3=<imx-nxp-ara2-2.1.1>/usr/include/sdk_ara \
ARA2_SDK_V3_0=<arart-r3.0>/arart-r3.0-x86/x86/include \
ARA2_SDK_REQUIRE=all cargo test -p ara2-sys --test verify_sdk
```

Or one version at a time with `crates/ara2-sys/update.sh --verify <1.1|1.3|3.0> <dir>`, which also prints what the SDK declares that ara2-sys does not. Unset versions are skipped unless `ARA2_SDK_REQUIRE` names them.

### Python tests

`tests/test.py` covers the Python bindings. Install the wheel into a virtual environment on the target rather than the system Python:

```bash
uvx maturin build --release -m crates/ara2-py/Cargo.toml \
  --target aarch64-unknown-linux-gnu --zig --features pyo3/abi3-py311
scp target/wheels/edgefirst_ara2-*aarch64*.whl tests/test.py <target>:ara2-rs-test/
ssh <target> 'cd ~/ara2-rs-test && python3 -m venv venv && . venv/bin/activate && \
  pip install --force-reinstall edgefirst_ara2-*.whl pytest && python3 -m pytest -q test.py'
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
| `ARA2_SOCKET` | Pins the proxy endpoint: a UNIX socket path, `unix:/path` or `tcp:IPV4:PORT` |
| `ARA2_ABI` | Forces a struct layout (`1.1`, `1.3`, `3.0`); see the README for when it is honoured |
| `ARA2_SDK_V1_1`, `ARA2_SDK_V1_3`, `ARA2_SDK_V3_0` | SDK include directories for `verify_sdk` |
| `ARA2_SDK_REQUIRE` | Versions `verify_sdk` must not skip (comma-separated, or `all`) |
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
ssh <target> /tmp/async_infer /root/models/yolov8n-det-int16.dvm 10

# Rust — pipelined inference with circular buffer (depth=2)
ssh <target> /tmp/async_pipeline /root/models/yolov8n-det-int16.dvm 50 2

# Rust — multi-model (dual + A/B alternating)
ssh <target> /tmp/async_multi_model \
  /root/models/yolov8n-seg-int16.dvm /root/models/yolo11n-seg-int16.dvm 50

# Python
ssh <target> 'pip install --force-reinstall --no-deps /tmp/edgefirst_ara2-*.whl && \
  python3 /tmp/async_infer.py /root/models/yolov8n-det-int16.dvm 10'

# Python — pipelined
ssh <target> 'python3 /tmp/async_pipeline.py /root/models/yolov8n-det-int16.dvm 50 2'

# Python — multi-model
ssh <target> 'python3 /tmp/async_multi_model.py \
  /root/models/yolov8n-seg-int16.dvm /root/models/yolo11n-seg-int16.dvm 50'
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
ssh <target> /tmp/async_pipeline /root/models/yolov8n-seg-int16.dvm 100 3

# Python
ssh <target> python3 /tmp/async_pipeline.py /root/models/yolov8n-seg-int16.dvm 100 3
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
  /root/models/yolov8n-seg-int16.dvm /root/models/yolo11n-seg-int16.dvm 100

# Python
ssh <target> python3 /tmp/async_multi_model.py \
  /root/models/yolov8n-seg-int16.dvm /root/models/yolo11n-seg-int16.dvm 100
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
| `ci.yml` | `cargo fmt`, clippy with and without `ara2/codec,ara2/decoder`, `cargo nextest run --workspace -E 'test(dvm_metadata) \| test(offline) \| binary(verify_sdk)'`, builds and Python wheels |
| `nightly.yml` | The same test selection on a schedule |

Hardware-dependent tests (`session`, `endpoint`, `model`) do not run in CI — they require ARA-2 hardware. The SDK verifier does not run either, as no SDK is available there; run it by hand when an SDK changes.

## Debugging

### Verifying hardware setup

```bash
# Check PCIe device
lspci | grep -i kinara

# Check proxy service
systemctl status rt-sdk-ara2
journalctl -u rt-sdk-ara2 --no-pager -n 50

# Check socket
ls -la /var/run/proxy.sock
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
