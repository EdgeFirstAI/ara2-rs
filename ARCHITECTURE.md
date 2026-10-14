# Architecture

This document describes the architecture of the ARA-2 client library workspace.

## Overview

The ARA-2 client library provides Rust interfaces for communicating with
ARA-2 neural network accelerator devices via the ARA-2 proxy service.

```
┌──────────────────────────────────────────────────────────────────┐
│                      User Applications                           │
│                                                                  │
│  use ara2::Session;                                              │
│  use edgefirst_tensor::Tensor;                                   │
│  use edgefirst_image::ImageProcessor;                            │
│  use edgefirst_decoder::DecoderBuilder;                          │
└──────────────────────────────────────────────────────────────────┘
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌──────────────────────────────────┐
│          ara2             │   │  edgefirst-tensor / -image       │
│     (Core Library)        │   │                                  │
│                           │──▶│  • DMA/SHM tensor allocation     │
│  • Session management     │   │  • G2D/OpenGL image processing   │
│  • Endpoint enumeration   │   │  • Overlay / mask rendering      │
│  • Model loading/infer    │   ├──────────────────────────────────┤
│  • DVM metadata parsing   │   │  edgefirst-decoder / -codec      │
│                           │   │  (examples only)                 │
│                           │   │  • YOLO decode + NMS             │
│                           │   │  • JPEG/PNG decode into a tensor │
└─────────────┬─────────────┘   └──────────────────────────────────┘
              │
              ▼
┌───────────────────────────┐
│         ara2-sys          │
│      (FFI Bindings)       │
│                           │
│  • C type definitions     │
│  • Dynamic lib loading    │
│  • Symbol resolution      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  libaraclient (SDK name)  │
│   (Kinara Runtime Lib)    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│       ARA-2 Proxy         │
│   (UNIX socket or TCP)    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     ARA-2 Hardware        │
│  (Neural Accelerator)     │
└───────────────────────────┘
```

## Workspace Structure

```
ara2-rs/
├── Cargo.toml              # Workspace configuration
├── crates/
│   ├── ara2/               # Core Rust library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs          # Public API, library loading, ABI selection
│   │       ├── raw.rs          # The only reader of version-specific structs
│   │       ├── connect.rs      # Proxy endpoint resolution for connect()
│   │       ├── discover.rs     # Running proxies and proxy config parsing
│   │       ├── session.rs      # Session management
│   │       ├── endpoint.rs     # Endpoint operations
│   │       ├── model.rs        # Model loading/inference (sync + async)
│   │       ├── status.rs       # Detailed status (DVAPI 1.3+)
│   │       ├── version.rs      # Component versions
│   │       ├── error.rs        # Error types
│   │       └── dvm_metadata.rs # DVM metadata parsing
│   │
│   ├── ara2-sys/           # FFI bindings
│   │   ├── Cargo.toml
│   │   ├── update.sh       # Regenerate bindings; --verify against an SDK
│   │   ├── include/        # Our DVAPI declarations (common + per version)
│   │   ├── tests/verify_sdk.rs # Checks declarations against SDK headers
│   │   └── src/
│   │       ├── lib.rs      # Module wiring and compile-time layout pins
│   │       ├── ffi.rs      # Generated: common types + shared functions
│   │       └── v1_1.rs, v1_3.rs, v3_0.rs # Generated: per-version layouts
│   │
│   └── ara2-py/            # Python bindings (PyO3)
│       ├── Cargo.toml
│       ├── pyproject.toml
│       ├── edgefirst_ara2.pyi  # Type stubs for IDE support
│       └── src/
│           ├── lib.rs      # Module registration
│           ├── session.rs  # Session pyclass
│           ├── endpoint.rs # Endpoint pyclass
│           ├── model.rs    # Model + InferRequest pyclasses
│           ├── error.rs    # Exception hierarchy
│           ├── types.rs    # Data type pyclasses
│           └── metadata.rs # DVM metadata pyclasses
│
└── examples/               # Example applications
    ├── yolov8.rs           # YOLOv8 detection/segmentation
    ├── yolov8.py           # YOLOv8 Python equivalent
    ├── async_infer.rs      # Async inference benchmark (Rust)
    ├── async_infer.py      # Async inference benchmark (Python)
    ├── async_pipeline.rs   # Pipelined async inference (Rust)
    ├── async_pipeline.py   # Pipelined async inference (Python)
    ├── async_multi_model.rs # Multi-model async pipeline (Rust)
    ├── async_multi_model.py # Multi-model async pipeline (Python)
    ├── endpoints.py        # Endpoint discovery
    └── test_dvm_metadata.rs
```

## Core Library (ara2)

### Ownership Model

All major types use `Arc<SessionInner>` for shared ownership instead of
lifetimes. This enables cross-thread usage and simplifies the API:

```
Session ──Arc──▶ SessionInner (lib handle + session ptr)
    │                  ▲
    │                  │
    └─ list_endpoints()│
           │           │
           ▼           │
       Endpoint ──Arc──┘
           │       │
           │       └──Arc──▶ EndpointList (C-allocated buffer, freed on Drop)
           │
           └─ load_model_from_file()
                   │
                   ▼
               Model ──Arc──▶ SessionInner
                   │
                   └─ submit()
                          │
                          ▼
                      InferRequest ──Arc──▶ SessionInner
                          │
                          └─ (borrows Model's tensor buffers)
```

- **Session**: Cheaply cloneable (reference counted). Multiple handles share
  the same underlying connection. Freed via `dv_session_close` on last drop.
- **Endpoint**: Holds shared references to both the session and the C-allocated
  endpoint list buffer. The list is freed via `dv_endpoint_free_group` when all
  endpoints from the list are dropped.
- **Model**: NOT cloneable. Owns its loaded NPU resources. Automatically
  unloaded via `dv_model_unload` on drop.
- **InferRequest**: Created by `Model::submit()`. Holds an `Arc` to the session for the wait call, and the shared-memory registrations of the tensors it uses. Dropping a request that is still pending waits for it, bounded at 4 × `DEFAULT_TIMEOUT_MS`, before `dv_infer_free`; if it is still pending after that, its registrations are leaked rather than released under a running inference.

### Thread Safety

- `Session` is `Send + Sync` — can be shared across threads
- `Endpoint` is `Send + Sync` — can be shared across threads
- `Model` is `Send` but NOT `Sync` — can be moved between threads but
  inference operations must not be called concurrently
- `InferRequest` is `Send` — can be moved to another thread and waited on
  there (e.g., submit from a preprocessing thread, wait from a
  postprocessing thread)

For multi-model parallelism, load separate `Model` instances per thread.

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `codec` | no | Adds `Error::Codec` and its `From<edgefirst_codec::CodecError>` conversion, for callers that decode JPEGs or PNGs inside a function returning `ara2::Error` |
| `decoder` | no | Adds `OutputSpec::dshape_typed()`, returning `edgefirst_decoder::configs::DimName` pairs, so a decoder configuration can be built straight from parsed DVM metadata. `dshape` itself is always `Vec<(String, usize)>`: features unify across the dependency graph, so one that reshaped a public type would reshape it for every crate in the build |
| `camera` | no | Builds the libcamera/Wayland live example (`yolov8_live`). Implies `decoder`, and turns on `edgefirst-image/decode` for its fused `draw_masks` call |

`edgefirst-tensor` and `edgefirst-image` are unconditional dependencies:
`edgefirst-tensor` types appear in the public API of `Model`, so tensor
allocation and inference cannot be expressed without them, and
`edgefirst-image` supplies the `Error::ImageError` variant that carries a
preprocessing failure.

`edgefirst-decoder` and `edgefirst-codec` are not. Decoding model outputs and
loading images from disk are application concerns, so the examples and
benches take those crates as their own dependencies and `cargo tree -e normal
-p ara2` reaches neither. The `codec` and `decoder` features exist only for
callers that want those crates' types in `ara2`'s own API. `edgefirst-image`
is taken with `default-features = false` plus `opengl` and `static` for the
same reason — its default set turns on `codec`, which would pull
`edgefirst-codec` back in transitively.

## FFI Layer (ara2-sys)

The `ara2-sys` crate provides low-level FFI bindings to `libaraclient`:

- Generated by `bindgen` from our own declarations in `crates/ara2-sys/include/`; no SDK header is shipped
- Uses `libloading` for runtime dynamic library loading
- `ara2::open_library()` tries each name in `ara2::LIBRARY_NAMES` in turn, since the shared library has been renamed in every SDK drop so far (`libaraclient_aarch64.so`, `libaraclient_x86_64.so`, `libaraclient.so.1`)
- Each symbol resolves into its own `Result` field, so a library lacking a function still loads. Calling bindgen's generated method for a missing symbol panics, so code that binds an optional function checks the field first: the detailed-status `_s`/`_e` calls are guarded by `is_ok()` and fall back to the plain call when the symbol is absent

## Per-version struct layouts

DVAPI 1.1, 1.3 and 3.0 differ in struct layout, and 3.0 inserts a field in the middle of `dv_model` (`hw_cycles` at offset 40), moving every field after it. No version's layout is a prefix of another's, so each is declared in full.

```
ara2-sys                                   ara2
┌───────────────────────────────┐          ┌──────────────────────────────┐
│ crate root (dvapi_common.h)   │          │ open_library()               │
│  shared types, shared fns,    │          │  probe dv_get_client_lib_    │
│  dv_model etc. as OPAQUE      │◀─────────│  version -> Abi (+ ARA2_ABI) │
├───────────────────────────────┤          ├──────────────────────────────┤
│ v1_1 │ v1_3 │ v3_0            │          │ raw.rs: with_layout!(abi,    │
│ full dv_model, dv_endpoint_   │◀─────────│  ptr => dv_model, |m| ...)   │
│ statistics, ... per version;  │          │  casts the opaque pointer to │
│ version-only function tables  │          │  the Abi's module            │
└───────────────────────────────┘          └──────────────────────────────┘
```

- **Common and per-version modules.** The crate root holds everything identical across versions; structs whose layout differs are opaque there. `v1_1`, `v1_3` and `v3_0` declare those structs at their own layout under the same names, and carry function tables for functions that exist only in, or differ in signature in, that version. They are loaded from the same library handle as the shared table.
- **ABI selection.** `open_library()` calls `dv_get_client_lib_version` before anything else and maps major.minor to an `Abi` once per session. `ARA2_ABI` is refused with `Error::AbiOverrideRejected` when it contradicts the reported version: a recognised version other than the forced one, or an unrecognised version of a different major version. A force equal to the recognised version, a force when the probe failed, and a force of the same major version over an unrecognised minor are accepted.
- **One reader.** `raw.rs` is the only module that reads a version-specific struct. Its `with_layout!` macro casts an opaque pointer to the module the session's `Abi` selects and expands its body once per version, so every field access is type-checked against that version's own declaration. Everything else in `ara2` uses the accessors `raw.rs` provides.
- **Verification.** `crates/ara2-sys/update.sh --verify <version> <sdk-include-dir>` compiles C assertions of every declared size, offset, enumerator and prototype against the real SDK header. It is run by hand when an SDK drops, since SDKs are not available in CI. `ara2-sys/src/lib.rs` pins the verified sizes and offsets as compile-time assertions, so an edit that changes a layout fails the build.

## Telemetry

`telemetry.rs` holds the public telemetry types and the conversions from the raw structs: `InferenceStats`, `Clocks`, `MonotonicTime`, `Retries`, `EndpointInfo` and its parts, and `LogLevel`.
`Endpoint::statistics()` and `Endpoint::info()` build their results from version-normalised fields that `raw.rs` reads, so the public types never see a version-specific struct.

Two rules shape the module.

- **Expose only what hardware populates.** A counter is exposed only if at least one runtime fills it in on real hardware. Compiler statistics, the queue submission time, DDR addresses, the statistics filter options, the LLM and MCS APIs and the `_with_flags` power functions are not exposed. Per-model statistics, inference queue statistics, the endpoint power state and power-state control are not exposed either, because the runtime does not yet report or apply them reliably.
- **Absent is `None`, never zero.** A field the loaded DVAPI version lacks reads as `None`. Examples are the inference timestamps on 1.x, the NNP and SBP clocks on 1.1 and `largest_contiguous_free` before 3.0. Calls that exist only in some versions (`is_busy`, `set_log_level`) return `Error::Unsupported` where the runtime lacks them.

Fields that exist in only some layouts are read through the `VersionFields` trait in `raw.rs`. Its methods default to `None`, only the layouts that have the field override them, and `with_layout!` calls them uniformly, so a new version-only field is one default method and one override rather than a branch at every call site.

Inference timestamps are `MonotonicTime` values on `CLOCK_MONOTONIC`, converted only for DVAPI 3.0. DVAPI 1.x stamps the wall clock, and comparing those values with a monotonic reading would be silently wrong, so they are `None`. `monotonic_now()` reads the same clock for comparison, and the values line up with Perfetto traces. On DVAPI 3.0 hardware each stamp falls between `monotonic_now()` readings taken around the inference.

## Connection

`Session::connect()` makes a single connection attempt, against an endpoint chosen by inspection (`connect.rs`): `ARA2_SOCKET` (`/path`, `unix:/path` or `tcp:IPV4:PORT`), then the first existing path in `SOCKET_PATHS`, then a running proxy found by `discover()`, preferring a UNIX endpoint over a TCP one. One attempt only, because a failed create leaves the DVAPI 1.1 client in a state that can break a later one. The choice is a pure function over (override, existing paths, discovered proxies), unit-tested offline.

Only UNIX sessions pass file descriptors. On TCP, `None` memory resolves to heap and an explicit `Shm` or `DmaBuf` returns `Error::FdPassingUnavailable` before any DVAPI call. The resulting heap tensors have no fd, so `clone_fd()` on them fails and callers copy through `map()` rather than sharing by fd.

## Error Handling

A unified `Error` enum covers all failure modes:

```rust
pub enum Error {
    Io(std::io::Error),         // File I/O errors
    Library(libloading::Error), // Library loading failures
    Ara2 { code: i32, name: String, detail: Option<StatusDetail> }, // named NPU/proxy status
    Unsupported { feature: &'static str, abi: Abi }, // not in the loaded DVAPI version
    NullPointer(String),        // Null pointer from FFI
    InferenceFailed,            // Async inference failed on NPU
    InferenceNotCompleted(u32), // Unexpected completion status
    Codec(edgefirst_codec::CodecError),   // JPEG/PNG decode errors
    TensorError(edgefirst_tensor::Error), // Tensor allocation/mapping
    ImageError(edgefirst_image::Error),   // Image conversion/rendering
    // ... see crates/ara2/src/error.rs for the full list
}
```

`Error::Ara2` names the code with the library's own `dv_stringify_status_code`. On DVAPI 1.3 and 3.0, model loading, asynchronous submission and waiting go through the `_s` / `_e` detailed-status variants, so their errors also carry the component, severity and sub-code.

`Error::ModelLoadFailed` wraps a `DV_MODEL_LOAD_FAILURE` from `load_model_from_file` with the model's format compatibility, described below. A file that could not be opened for the format check keeps its plain `Error::Ara2`, since that is not a parse failure.

## Model format

A `.dvm` built for the other runtime generation fails to load with the same status and log line as a damaged file, so `dvm_format.rs` reads the model format before loading and `load_model_from_file` explains a failure with it (`ModelCompatibility`: `Mismatch`, `Matching` or `Unknown`). Format 2 belongs to DVAPI 1.x runtimes and format 3 to DVAPI 3.x; any other format is `Unknown`.

The `.dvm` is a FlatBuffers buffer whose schema Kinara does not publish. The converter stores the format as a one-field sub-table holding a single byte, the major version `Model::version()` reports after a load; the bytes after it are padding. Its field id differs between generations but its slot sits at byte offset 80 of the root table in both, so the reader finds it by that slot and validates each step: root offset and root table in bounds, a root vtable of sane size that lists slot 80, a forward offset to a sub-table whose vtable declares exactly one field at offset 4, and a format from 1 to 15. Any failed check is reported as unknown rather than guessed at, since without the schema a guess cannot be told from a coincidence. The root vtable usually sits near the end of the file, so the file reader seeks to each piece instead of reading a fixed prefix or the whole model.

On a load failure the converter provenance (`converter_version`, `sdk_version`) is read from the `ara2` section of the embedded `edgefirst.json`, reading only the ZIP trailer; missing or malformed metadata leaves it `None` and never changes the error.

All error variants implement `std::error::Error` with proper `source()`
chaining for integration with `anyhow` and `eyre`. `ModelLoadFailed` exposes its status as the `status` field rather than through `source()`, since its message already includes it.

## Inference Pipeline

### Synchronous

```
1. Session::connect()
   └─▶ Connect to the ARA-2 proxy (UNIX socket, or TCP without fd passing)

2. session.list_endpoints()
   └─▶ Query proxy for available NPU devices

3. endpoint.load_model_from_file("model.dvm")
   └─▶ Upload compiled model to NPU DRAM

4. model.allocate_tensors(Some(TensorMemory::DmaBuf))
   └─▶ Allocate DMA-backed input/output buffers

5. Write input data to model.input_tensor(0)
   └─▶ Zero-copy via DMA file descriptor

6. model.run()
   └─▶ Synchronous inference on ARA-2 hardware

7. Read output data from model.output_tensor(i)
   └─▶ Dequantize and decode results
```

### Asynchronous (submit/wait)

Steps 1–5 are identical. Step 6 is replaced with:

```
6a. model.submit()
    └─▶ Non-blocking — returns InferRequest immediately
    └─▶ NPU begins executing inference in the background

6b. CPU work (preprocess next frame, postprocess previous, etc.)
    └─▶ Overlaps with NPU execution

6c. request.wait(timeout_ms)
    └─▶ Blocks until NPU finishes, returns InferenceStats
    └─▶ InferRequest is consumed and freed

7. Read output data from model.output_tensor(i)
   └─▶ Same as synchronous path
```

The async path enables pipeline parallelism: while the NPU executes
inference on frame N, the CPU can preprocess frame N+1 or postprocess
frame N-1. This is the pattern used by the profiler's pipelining engine.

### Monitoring

`session.inflight_count()` returns the number of submitted requests that
have not yet completed. Useful for pipeline depth monitoring and
backpressure.
