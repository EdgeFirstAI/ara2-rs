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
│     libaraclient.so.1     │
│   (Kinara Runtime Lib)    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    ARA-2 Proxy (dvproxy)  │
│    (System Service)       │
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
│   │       ├── lib.rs          # Public API and re-exports
│   │       ├── session.rs      # Session management
│   │       ├── endpoint.rs     # Endpoint operations
│   │       ├── model.rs        # Model loading/inference (sync + async)
│   │       ├── error.rs        # Error types
│   │       └── dvm_metadata.rs # DVM metadata parsing
│   │
│   ├── ara2-sys/           # FFI bindings
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs      # Lint configuration
│   │       └── ffi.rs      # Generated C bindings (bindgen)
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
- **InferRequest**: Created by `Model::submit()`. Holds an `Arc` to the
  session for the wait call. Borrows the model's tensor buffers (caller must
  keep the `Model` alive). Freed via `dv_infer_free` on drop (cancels if
  still pending).

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
| `decoder` | no | Types `OutputSpec::dshape` as `edgefirst_decoder::configs::DimName` pairs instead of the metadata's raw axis-name strings, so a decoder configuration can be built straight from parsed DVM metadata |
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

The `ara2-sys` crate provides low-level FFI bindings to `libaraclient.so`:

- Generated by `bindgen` from `dvapi.h`
- Uses `libloading` for runtime dynamic library loading
- All symbols resolved lazily at first use
- Type aliases map C types to Rust equivalents

## Error Handling

A unified `Error` enum covers all failure modes:

```rust
pub enum Error {
    Io(std::io::Error),         // File I/O errors
    Library(libloading::Error), // Library loading failures
    Ara2(dv_status_code),       // NPU/proxy error codes
    NullPointer(String),        // Null pointer from FFI
    InferenceFailed,            // Async inference failed on NPU
    InferenceNotCompleted(u32), // Unexpected completion status
    Codec(edgefirst_codec::CodecError),   // JPEG/PNG decode errors
    TensorError(edgefirst_tensor::Error), // Tensor allocation/mapping
    ImageError(edgefirst_image::Error),   // Image conversion/rendering
    // ... see crates/ara2/src/error.rs for the full list
}
```

All error variants implement `std::error::Error` with proper `source()`
chaining for integration with `anyhow` and `eyre`.

## Inference Pipeline

### Synchronous

```
1. Session::create_via_unix_socket()
   └─▶ Connect to ara2-proxy via UNIX socket

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
    └─▶ Blocks until NPU finishes, returns ModelTiming
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
