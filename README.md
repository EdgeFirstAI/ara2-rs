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
use edgefirst_tensor::TensorTrait as _;

// Connect to the ARA-2 proxy service
let session = Session::connect()?;

// Enumerate NPU endpoints and check status
let endpoints = session.list_endpoints()?;
let endpoint = &endpoints[0];
println!("Endpoint state: {:?}", endpoint.check_status()?);

// Load a compiled model (.dvm) and allocate tensors. `None` picks the best
// memory the platform offers (DMA-BUF where available), or heap over TCP.
let mut model = endpoint.load_model_from_file("model.dvm".as_ref())?;
model.allocate_tensors(None)?;

// Run inference
let stats = model.run()?;
println!("NPU inference: {:?}", stats.timing.run_time);
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
stats = request.wait()
print(f"NPU inference: {stats.timing.run_time * 1e3:.2f} ms")
```

See the [`async_infer`](examples/async_infer.rs) example for a complete
benchmark comparing synchronous vs. asynchronous inference, and
[`async_pipeline`](examples/async_pipeline.rs) for pipelined inference
with a circular buffer of DMA-BUF tensor sets (2x+ throughput improvement).

## Telemetry

`Model::run()` and `InferRequest::wait()` return an `InferenceStats`: the `timing` durations (sub-microsecond precision), the `clocks` the inference ran at, `hw_cycles` and `fp_cycles`, `retries`, and the `input_start`, `inference_start` and `output_start` timestamps.
The timestamps are `MonotonicTime` values on `CLOCK_MONOTONIC`, comparable with `ara2::monotonic_now()` and with a Perfetto trace taken on the same host.
Only the DVAPI 3.0 runtime stamps on `CLOCK_MONOTONIC`; DVAPI 1.x stamps the wall clock, so the timestamps are `None` there rather than a value on the wrong clock.

```rust
use ara2::Session;

let session = Session::connect()?;
let endpoint = &session.list_endpoints()?[0];

let info = endpoint.info()?;                  // static: chip, DRAM vendor, host interface
let stats = endpoint.statistics()?;           // sampled: clocks, voltage, temperature, DRAM
let busy = endpoint.is_busy()?;               // an inference is running or queued

let mut model = endpoint.load_model_from_file("model.dvm".as_ref())?;
model.allocate_tensors(None)?;
let run = model.run()?;
println!("{} cycles at {} MHz", run.hw_cycles, run.clocks.sys_mhz);
if let Some(start) = run.inference_start {
    println!("queued at {:?}", start.as_duration());
}
# Ok::<(), ara2::Error>(())
```

`Endpoint::statistics()` reports the endpoint state, `Clocks` (system, DRAM, NNP and SBP), core voltage, temperature, soft-reset count and DRAM usage. The DVAPI 1.1 runtime reports the core voltage as 0.0; DVAPI 1.3 and later measure it.
`Endpoint::info()` reports the device and vendor identifiers, the chip, the DRAM vendor name and the host interface.
Each field a runtime does not populate is `None`, never zero.
The inference timestamps (`input_start`, `inference_start`, `output_start`) are on `CLOCK_MONOTONIC`, and so line up with Perfetto, on DVAPI 3.0 only.

Per-model statistics, inference queue statistics, the endpoint power state and power-state control are not exposed, because the runtime does not yet report or apply them reliably.

### Availability by DVAPI version

`None` means the field is absent on that version; an error means the call returns `Error::Unsupported`.
The DVAPI 1.3 column was observed on an i.MX 95 board (runtime 1.3.2.0).
DVAPI 1.1 follows the runtime's declared layout and has not been re-measured for this release.
The DVAPI 3.0 column was observed on an i.MX 95 board (runtime 3.0.0.0, firmware 0.128.11).

| Item | 1.1 | 1.3 | 3.0 |
|------|-----|-----|-----|
| `InferenceStats` timing, cycles, retries | yes | yes | yes |
| `InferenceStats` `input_start`, `inference_start`, `output_start` | `None` | `None` | `CLOCK_MONOTONIC` |
| `Clocks` `nnp_mhz`, `sbp_mhz` | `None` | yes | yes |
| `DramStatistics::largest_contiguous_free` | `None` | `None` | yes |
| `Endpoint::is_busy()` | yes | yes | yes |
| `Session::set_log_level()` | error | yes | yes |

## Control

**Log level.**
`Session::set_log_level(LogLevel)` sets the client library's own logging on DVAPI 1.3 and 3.0, and returns `Error::Unsupported` on 1.1.

### Cost of sampling

`Endpoint::statistics()` is a round trip to the proxy.
The `telemetry` example measures its cost by timing the same inference loop alone and while a second thread samples every 100 ms.
On the DVAPI 1.3 board (i.MX 95 with `yolov8n-det-int16`, 200 iterations, three runs) inference ran at 126.8, 126.3 and 125.8 fps alone and 125.7, 126.4 and 126.0 fps while sampling, so sampling every 100 ms made no measurable difference.
A `statistics()` call took 1.05 to 1.11 ms on average and at most 1.5 ms.
On the DVAPI 3.0 board (i.MX 95, proxy over TCP, a 3.0-compiled detection model, three runs) inference ran at 29.9, 29.9 and 29.8 fps alone and 30.0, 29.9 and 29.9 fps while sampling, again no measurable difference.
A `statistics()` call took 7.4 to 10.0 ms on average and at most 40 ms there.

## Runtime Requirements

Requires a `libaraclient` implementing DVAPI 1.1, 1.3 or 3.0. The following must be present on the target system:

- **`libaraclient`** — Kinara client library. Three DVAPI generations are in circulation and all are supported:

  | Packaging | DVAPI | Typical proxy transport |
  |-----------|-------|-------------------------|
  | Kinara ARA-2 runtime 1.2.1, packaged by [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara) | 1.1 | UNIX `/var/run/ara2.sock` |
  | NXP `imx-nxp-ara2` (rt-sdk-ara2 2.1.1) | 1.3 | UNIX `/var/run/proxy.sock` |
  | Kinara arart r3.0 | 3.0 | TCP `127.0.0.1:5000` (sample config); UNIX with `interface_type: "SOCKET"` |

  [`open_library`](crates/ara2/src/lib.rs) probes the loaded library with `dv_get_client_lib_version` and reads every version-specific struct at that version's layout; 3.0 moves fields within `dv_model`, so no one layout serves all three. A proxy accepts only clients of its own DVAPI version, so a system carries one library. Models are compiled for one runtime generation too: see [Model compatibility](#model-compatibility). The library's name has also differed in every packaging so far and there is no soname to rely on, so the architecture-suffixed name for the target is tried first (`libaraclient_aarch64.so`, `libaraclient_x86_64.so`), then `libaraclient.so` and `libaraclient.so.1`.
- **The proxy service** — provides NPU access, must be running before connecting. It is `rt-sdk-ara2.service` under NXP's packaging and `ara2.service` under meta-kinara's.
- **ARA-2 hardware** — PCIe accelerator card visible via `lspci`

### Model compatibility

A `.dvm` is compiled for one runtime generation. Converters built on the DVAPI 1.x SDKs write model format 2, which loads on DVAPI 1.x runtimes only (verified on 1.1 and 1.3); converters built on the DVAPI 3.0 SDK write format 3, which loads on DVAPI 3.x only. The client library rejects a model of the other generation with `DV_MODEL_LOAD_FAILURE` (520) and logs only that it could not parse the file, exactly as it does for a damaged one.

`ara2::read_model_format` and `read_model_format_from_file` read the format from a model without loading it (the file reader reads only the parts that hold it), returning `None` when the file's structure is not recognised. `ModelCompatibility::new(format, session.abi())` compares it with the runtime. `Endpoint::load_model_from_file` does both before loading, logs a warning for a model of the other generation, and turns a `DV_MODEL_LOAD_FAILURE` into `Error::ModelLoadFailed`, which keeps the library's status and says which case applies:

```text
DV_MODEL_LOAD_FAILURE (520): yolov8n-det-int16.dvm is model format 2 (built for DVAPI 1.x runtimes; converted by ara2-converter 2.5.0), but this runtime is DVAPI 3.0; reconvert it with a converter built on the DVAPI 3.x SDK
```

When the format matches the runtime the message says the file may be damaged, and when the format cannot be read it names both causes. The converter and SDK versions come from the `ara2` section of the embedded `edgefirst.json` when present (`Ara2Info::converter_version`, `sdk_version`, `dvm_format`).

### Connecting

`Session::connect()` chooses one endpoint by inspection and makes a single connection attempt. It takes the first of:

1. `ARA2_SOCKET`, if set: a UNIX socket path (`/var/run/proxy.sock`), `unix:/path`, or `tcp:IPV4:PORT` (`tcp:127.0.0.1:5000`).
2. The first existing path in [`SOCKET_PATHS`](crates/ara2/src/lib.rs) (`/var/run/proxy.sock`, `/var/run/ara2.sock`).
3. A running proxy found by `ara2::discover()`, taking a UNIX endpoint over a TCP one.

`ara2::proxy_endpoint()` reports the endpoint `connect()` would open without opening it; when nothing is found the error lists what was checked.

Only a UNIX socket can pass file descriptors. On a TCP session (`Session::supports_fd_passing()` is false), `allocate_tensors(None)` and the input/output set allocators default to heap memory, and an explicit `TensorMemory::Shm` or `TensorMemory::DmaBuf` fails with `Error::FdPassingUnavailable` before any call reaches the library. Inference works on both transports; zero-copy DMA-BUF needs UNIX. Tensors the client allocates over TCP are heap memory with no file descriptor, so `clone_fd()` on them fails and they cannot be passed to the HAL by fd: check `tensor.memory()` (or `session.supports_fd_passing()`) and copy through `map()` instead, as the `yolov8` examples do. `connect()` logs a warning when it selects a TCP endpoint. For r3.0, setting `interface_type: "SOCKET"` and `interface_socket_file` in the proxy configuration gives a UNIX socket with fd passing.

### Versions

`Session::versions()` returns the component versions actually running (client library, proxy, firmware and so on), one entry per component. `Session::supported_versions()` returns every version the proxy declares support for, and may list a component several times (r3.0 lists four firmware versions).

### Forcing a layout

`ARA2_ABI` (`1.1`, `1.3` or `3.0`) forces a layout, for bringing up a library reporting a version this build does not recognise. It is refused with `Error::AbiOverrideRejected` when it contradicts the version the library reports: a recognised version other than the forced one, or an unrecognised version with a different major version. Forcing the version the library already reports is accepted, as is any force when the version probe failed. So forcing `3.0` on a future 3.1 library works, and forcing `1.3` on a 3.0 library does not. A library the override contradicts is skipped and the search moves on to the next name in `LIBRARY_NAMES`, so `ARA2_ABI` can also pick one of several installed libraries; the first refusal is reported only when no candidate is accepted.

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
| [`telemetry.rs`](examples/telemetry.rs) | Rust — Print everything the runtime reports and measure the cost of sampling endpoint statistics |
| [`telemetry.py`](examples/telemetry.py) | Python — The same report via `edgefirst-ara2` |
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
