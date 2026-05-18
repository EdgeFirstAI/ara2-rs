# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] - 2026-05-18

### Changed (BREAKING)

- **edgefirst-hal** upgraded from 0.22.0 to 0.23.0. The standalone
  `load_image` function has been removed from the image crate;
  callers now use the `ImageLoad` trait from the new `edgefirst_codec`
  crate (re-exported as `edgefirst_hal::codec`). The new pattern
  pre-allocates a tensor, then decodes in-place via
  `tensor.load_image(&mut decoder, &bytes, &opts)`.
- **`image` crate dependency removed.** The `Error::Image` variant
  (wrapping `image::ImageError`) is replaced by `Error::Codec`
  (wrapping `edgefirst_hal::codec::CodecError`). Downstream `match`
  arms on `ara2::Error::Image(_)` must be updated.
- The `yolov8` example now decodes directly into a GPU-accessible
  tensor via `ImageDecoder`, eliminating the previous CPU→GPU copy
  step.

### Migration

| 0.9.x | 0.10.0 |
|-------|--------|
| `ara2 = "0.9"` | `ara2 = "0.10"` |
| `edgefirst-hal = "0.22"` (downstream pin) | `edgefirst-hal = "0.23"` |
| `ara2::Error::Image(e)` | `ara2::Error::Codec(e)` |
| `load_image(&bytes, Some(fmt), mem)` | `peek_info` → `Tensor::image` → `tensor.load_image(&mut decoder, &bytes, &opts)` |

## [0.9.0] - 2026-05-11

### Changed (BREAKING)

- **edgefirst-hal** upgraded from 0.21.0 to 0.22.0. HAL types
  (`Tensor<u8>`, `TensorMemory`, `edgefirst_hal::tensor::Error`,
  `edgefirst_hal::image::Error`) appear in the public API surface of
  `ara2`, so this is a transitive ABI break: downstream crates
  pinning `edgefirst-hal = "0.21"` must bump in lockstep. No source
  changes were required in `ara2` or its examples — the workspace
  builds cleanly against the new HAL with the pin bump alone.

### Migration

| 0.8.x | 0.9.0 |
|-------|-------|
| `ara2 = "0.8"` | `ara2 = "0.9"` |
| `edgefirst-hal = "0.21"` (downstream pin) | `edgefirst-hal = "0.22"` |

## [0.8.0] - 2026-05-08

### Changed (BREAKING)

- **edgefirst-hal** upgraded from 0.20.0 to 0.21.0. HAL types
  (`Tensor<u8>`, `TensorMemory`, `edgefirst_hal::tensor::Error`,
  `edgefirst_hal::image::Error`) appear in the public API surface of
  `ara2`, so this is a transitive ABI break: downstream crates
  pinning `edgefirst-hal = "0.20"` must bump in lockstep. No source
  changes were required in `ara2` or its examples — the workspace
  builds cleanly against the new HAL with the pin bump alone.
- **ndarray** upgraded from 0.16 to 0.17. This aligns with
  edgefirst-hal 0.21's internal ndarray version, eliminating the
  previous dual-version situation where numpy pulled ndarray 0.17
  alongside ara2's ndarray 0.16.

### Migration

| 0.7.x | 0.8.0 |
|-------|-------|
| `ara2 = "0.7"` | `ara2 = "0.8"` |
| `edgefirst-hal = "0.20"` (downstream pin) | `edgefirst-hal = "0.21"` |
| `ndarray = "0.16"` (if used directly) | `ndarray = "0.17"` |

## [0.7.0] - 2026-05-07

### Changed (BREAKING)

- **edgefirst-hal** upgraded from 0.19.0 to 0.20.0. HAL types
  (`Tensor<u8>`, `TensorMemory`, `edgefirst_hal::tensor::Error`,
  `edgefirst_hal::image::Error`) appear in the public API surface of
  `ara2`, so this is a transitive ABI break: downstream crates
  pinning `edgefirst-hal = "0.19"` must bump in lockstep. No source
  changes were required in `ara2` or its examples — the workspace
  builds cleanly against the new HAL with the pin bump alone.

### Migration

| 0.6.x | 0.7.0 |
|-------|-------|
| `ara2 = "0.6"` | `ara2 = "0.7"` |
| `edgefirst-hal = "0.19"` (downstream pin) | `edgefirst-hal = "0.20"` |

## [0.6.0] - 2026-05-06

### Changed (BREAKING)

- **edgefirst-hal** upgraded from 0.18.0 to 0.19.0. HAL types
  (`Tensor<u8>`, `TensorMemory`, `edgefirst_hal::tensor::Error`,
  `edgefirst_hal::image::Error`) appear in the public API surface of
  `ara2`, so this is a transitive ABI break: downstream crates
  pinning `edgefirst-hal = "0.18"` must bump in lockstep. No source
  changes were required in `ara2` itself — all decoder use goes
  through the high-level `Decoder`/`materialize_masks` facade, which
  absorbs the 0.19 internals (binary `MaskResolution::Proto` masks,
  `ProtoData` layout-aware shape, new `pre_nms_top_k` / `max_det`
  decoder knobs).

### Removed (BREAKING)

- The `hal` Cargo feature on the `ara2` crate has been removed.
  `edgefirst-hal` and `image` are now mandatory dependencies — the
  `Model` API exposes `Tensor<u8>` and `TensorMemory` in its public
  signatures, so an FFI-only build was never a meaningful
  configuration. Consumers using `features = ["hal"]` will now get a
  Cargo error because `ara2` no longer defines that feature (for
  example: `package 'ara2' depends on feature 'hal' but 'ara2' does
  not have that feature`); consumers using `default-features = false`
  will continue to build, but HAL is now unconditionally pulled in.

### Migration

| 0.5.x | 0.6.0 |
|-------|-------|
| `ara2 = { version = "0.5", features = ["hal"] }` | `ara2 = "0.6"` |
| `cargo build -p ara2 --no-default-features` | `cargo build -p ara2` |
| `edgefirst-hal = "0.18"` (downstream pin) | `edgefirst-hal = "0.19"` |

## [0.5.0] - 2026-04-26

### Changed

- **edgefirst-hal** upgraded from 0.15 to 0.18.0. The `materialize_masks`
  API now takes an explicit `MaskResolution` parameter; the `yolov8`
  example passes `MaskResolution::Proto` for unchanged behaviour.
- **libloading** upgraded from 0.8 to 0.9. Internal FFI loading adjusted
  for the new `AsFilename` trait bound.
- **PyO3** upgraded from 0.24 to 0.28; **numpy** from 0.24 to 0.28.
  Selected `#[pyclass]` value types now opt-in to `from_py_object` as
  part of the migration. Return types migrated from `PyObject` to
  `Bound<'py, PyAny>`.
- **criterion** upgraded from 0.7 to 0.8.
- **clap** upgraded from 4.5 to 4.6.
- Minor dependency bumps: image, log, zip and transitive dependencies.

## [0.4.0] - 2026-04-12

### Added

- `OutputQuantization::effective(qmode)` helper that normalizes
  Kinara's per-qmode dequantization formulas into a single
  `(scale, offset)` pair. Only qmode 9 is currently supported; other
  modes return `Error::UnsupportedQmode`.
- `InputPreprocess` struct (core) / class (`ara2-py`) holding the
  per-channel image normalization parameters (`mean`, `scale`,
  `bgr_to_rgb`, `aspect_resize`, `mirror`, `center_crop`), queried via
  `Model::input_preprocess(i)` / `model.input_preprocess(i)`.
- `InputQuantization.qmode` and `InputQuantization.offset` fields
  sourced from `dv_model_input_preprocess_param::qmode` and `::offset`.
- `Ara2Info` metadata section (`DvmMetadata.ara2`) parsing the
  optional `ara2.qmode` field from `edgefirst.json` embedded in DVMs.
- `Session.close()` and `Model.close()` Python methods; both are
  idempotent and are called by the `__exit__` path so `with` blocks
  now actually release resources.
- `Error::UnsupportedQmode(i32)` variant raised by `dequantize()` when
  a model uses a quantization mode other than 9.

### Changed

- `Model::dequantize()` (core and Python) now uses the correct
  qmode-9 formula `(raw - offset) * scale`. Previously applied the
  qmode 0-3 formula `raw / qn`, which silently produced values off
  by several orders of magnitude on current production models.
- `ara2-py::Model.set_input_tensor` accepts any numpy array whose
  total byte length matches the tensor size — the buffer is obtained
  via `tobytes()` and memcpy'd verbatim. Callers no longer need to
  `.view(np.uint8)` before calling. Non-contiguous arrays are handled
  transparently (numpy makes a contiguous copy on the fly).
- `ara2-py::Model.get_output_tensor` returns a typed array
  (`int8`/`uint8`/`int16`/`uint16`/`float32`) reshaped to the
  tensor's declared `(C, H, W)` shape. Callers that relied on the
  legacy flat-`uint8` return must either use the typed array directly
  or call `.ravel()`.
- `ara2-py::Session` and `ara2-py::Model` now wrap `Option<inner>`
  internally. Method calls on a closed instance raise
  `Ara2Error("session is closed")` / `"model is closed"` instead of
  returning stale data.

### Fixed

- **Quantization:** `dequantize()` produced grossly wrong values on
  every qmode-9 model (which is every production DVM today).
- **Input dtype erasure:** `set_input_tensor` rejected any array that
  wasn't `uint8`, forcing a `.view(np.uint8)` workaround in every
  consumer.
- **Output dtype erasure:** `get_output_tensor` returned `uint8`
  regardless of the tensor's actual signedness, forcing a manual
  `.view(int8)` on consumers.
- **Input zero-point confusion:** Consumers were reading
  `InputQuantization.mean` as an integer zero-point — but `mean` was
  the per-channel float normalization mean, not a quantization
  zero-point. The actual zero-point is now exposed as
  `InputQuantization.offset`.
- **Missing close:** `Session.__exit__` and `Model.__exit__` were
  no-ops; there was no way to deterministically release resources
  outside a context manager. `close()` methods plus a real `__exit__`
  body fix both cases.

### Removed (BREAKING)

- `OutputQuantization.scale` field (was `output_scale` in the C
  struct; unused by every downstream consumer).
- `InputQuantization.mean` and `InputQuantization.scale` fields —
  moved to `InputPreprocess`.

### Migration

Consumers of `edgefirst-ara2 0.3.x` must update:

| 0.3.x | 0.4.0 |
|-------|-------|
| `int(iq.mean)` as zero-point | `iq.offset` (true zero-point) |
| `iq.mean`, `iq.scale` (per-channel) | `model.input_preprocess(i).mean`, `.scale` |
| `oq.scale` (used with old `dequantized = raw / qn` formula) | `oq.qn` (used with new `dequantized = (raw - offset) * qn` formula — see Changed section) |
| `model.set_input_tensor(0, arr.view(np.uint8))` | `model.set_input_tensor(0, arr)` |
| `raw.view(np.int8)` after `get_output_tensor` | unnecessary — returned array is already typed |
| `session.__exit__(None, None, None)` | `session.close()` |
| `model.__exit__(None, None, None)` | `model.close()` |

Non-qmode-9 DVMs now raise `Ara2Error("unsupported quantization mode: qmode=N ...")` from `dequantize()`. If you encounter this, file an issue with the model so qmode 0-3 support can be added with a test fixture.

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

[Unreleased]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/EdgeFirstAI/ara2-rs/releases/tag/v0.1.0
