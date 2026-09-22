# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (BREAKING)

- **`ara2-sys` is built additively from DVAPI 1.1.2, with DVAPI 1.3.2's additions layered on top.** DVAPI is the `dvapi.h` interface version `libaraclient` reports through `dv_get_client_lib_version`; it is not the SDK packaging version. NXP's rt-sdk-ara2 SDK 2.1.1 ships DVAPI 1.3.2, and the Kinara ARA-2 runtime that [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara) packages as 1.2.1 ships DVAPI 1.1.2. Base bindings come from the 1.1.2 headers and `dvapi-1.3.h` carries the measured 1.3 delta — 8 functions, 2 types, 3 enums, 9 status codes and the extended `dv_model_output_param_1_3`. Every 1.3 layout change is a field appended to the end of a struct, so the 1.1 layout remains valid for reading fields both generations share; `assert_appended!` in `ara2-sys` makes a future reorder or retype a build error. The function table needs no versioning: `--dynamic-loading` resolves each symbol into its own `Result`, so a 1.1 library simply leaves the 1.3 entries `Err`.
- **`State` is the union of both DVAPI generations and gains `State::Unknown`.** DVAPI 1.3 reuses endpoint state values 4 and 5: `ACTIVE_BOOSTED` and `THERMAL_INACTIVE` in 1.1 are `THERMAL_ACTIVE_SLOW` and `FAIL_SAFE` in 1.3. The integers are unchanged, so a single mapping compiles and runs while reporting the wrong name. `State` now carries `ActiveBoosted`, `ThermalInactive`, `ThermalActiveSlow` and `FailSafe`, and reports whichever the loaded library's own header defines. `State` is `#[non_exhaustive]`, so a later generation adding a state is no longer a breaking change.
- **`State::try_from` is replaced by `State::from_raw(value, abi)`, and `Error::EndpointStateInvalid` is removed.** Decoding needs the DVAPI generation to resolve values 4 and 5, and is now infallible: an unrecognised value yields `State::Unknown(raw, abi)` rather than an error, so reading endpoint telemetry no longer fails because of a state the build does not know. The match is against each generation's generated `DV_ENDPOINT_STATE*` constants rather than bare integers, so a future reassignment is a build error.
- **`DEFAULT_SOCKET` changed from `/var/run/ara2.sock` to `/var/run/proxy.sock`**, matching `interface_socket_file` in NXP's rt-sdk-ara2 `proxy_config.yaml`. The proxy service is `rt-sdk-ara2.service` on images shipping that integration, not `ara2.service`/`dvproxy.service`. `Session::connect()` no longer depends on this being the right answer — see `SOCKET_PATHS` below.
- **The library loaded internally is no longer a single hardcoded name.** The `libaraclient` name has differed in every packaging seen so far — `libaraclient.so.1`, then `libaraclient_aarch64.so` — and there is no soname to rely on. `ara2::open_library()` now tries a short list of candidates, `ara2::LIBRARY_NAMES`: the architecture-suffixed name for the build target (`libaraclient_aarch64.so` on `aarch64`, `libaraclient_x86_64.so` on `x86_64`), then the unsuffixed `libaraclient.so`, then the soname-versioned `libaraclient.so.1`. A build for either architecture now finds the library under any of these schemes without a compile-time choice. `LIBRARY_NAMES` is public; on targets the suffixed scheme has not been published for it holds the unsuffixed names only.

### Added

- **DVAPI version probing.** `ara2::open_library()` calls `dv_get_client_lib_version` immediately after opening a candidate — it takes no session, so the answer is available before the library is asked to do anything else — and maps the reported `major.minor` to `ara2::Abi`. `Abi` selects the two things a shared layout cannot cover: the byte stride of the `dv_model_output_param` array (112 bytes under 1.1, 120 under 1.3) and which endpoint state names apply. Without this, indexing a 1.1 library's output-param array with a 120-byte stride lands at the wrong address for every output beyond index 0 — reading garbage that could reach a `Vec` capacity and panic (`capacity overflow`), or silently misread tensor shape and size fields. Index 0 was unaffected because a stride error contributes nothing at offset 0. An image can install both library names side by side, and bindgen's dynamic loading accepts either, so the wrong one cannot be ruled out by name alone.
- `ara2::Abi`, `ara2::DvapiVersion`, `Session::abi()` and `Session::dvapi_version()` (Rust), and `Abi`, `Session.abi` and `Session.dvapi_version` (Python), expose what was probed.
- `ARA2_ABI` (`ara2::ABI_OVERRIDE_ENV`) forces an ABI of `1.1` or `1.3` instead of probing, so a target running a DVAPI version this build does not recognise can be brought up without a new release.
- `Error::UnsupportedDvapi` reports a library whose DVAPI generation has no layout in this build, naming the file and the version found. `Error::AbiOverrideInvalid` reports an unusable `ARA2_ABI` value.
- **`ara2::discover()` finds running proxies** (Python: `ara2.discover()`), returning a `Proxy` per process whose `comm` is `kinara_main` — the one handle common to both packagings, which ship the executable as `proxy_ara240` and `proxy` respectively. More than one proxy can run, so it returns a list; none found is an empty list, not an error. Each `Proxy` carries its `pid`, `config` (the path from `-c`/`--config`, if it named one), `exe`, and `endpoints`. Endpoints are resolved by observing the sockets the process holds open — file descriptors matched against `/proc/net/unix` on `SO_ACCEPTCON`, and `/proc/net/tcp` on `TCP_LISTEN` — which reports where it is really listening whatever decided that. Reading another user's file descriptors needs privilege and the proxy runs as root, so without it this falls back to parsing the configuration named on the command line. A proxy whose endpoints cannot be resolved either way is still reported, with `endpoints` empty.
- **`Session::from_config(path)` connects from a proxy configuration file** (Python: `Session.from_config(path)`). `proxy.interface_type` selects the address list in force — `SOCKET` takes `interface_socket_file`, `IPV4` pairs `interface_ip_address` with `interface_port` positionally, `NAMED_PIPE` is a Windows transport and yields nothing — and the first endpoint declared is connected. This is the usual situation across packagings: NXP's rt-sdk-ara2 reads `/etc/rt-sdk-ara240/proxy_config.yaml` and the Kinara runtime meta-kinara packages reads `/etc/ara2.yaml`, and the two name different sockets. `ara2::ProxyConfig::read` exposes the parse on its own, and `ara2::ProxyEndpoint` names a UNIX socket or a TCP/IPv4 address. Adds a `serde_yaml_ng` dependency.
- `ara2::SOCKET_PATHS` lets `Session::connect()` find the proxy under either packaging. The socket path is a packaging decision, like the library name: NXP's rt-sdk-ara2 configures `/var/run/proxy.sock` and the Kinara runtime meta-kinara packages configures `/var/run/ara2.sock`. With `ARA2_SOCKET` unset, `socket_path()` and `Session::connect()` resolve to the first candidate that exists, so one binary serves both stacks without the deployment supplying a path. Selection is by existence rather than by attempting each in turn, because a failed connection attempt is not free: the DVAPI 1.1 client keeps state across one, and a subsequent create on the same handle can return `DV_SESSION_INVALID_HANDLE`.
- `Error::LibraryNotFound` names every candidate in `LIBRARY_NAMES` when none of them loads. Reporting only the last failure pointed diagnostics at `libaraclient.so.1`, the candidate least likely to be the one the target actually ships.
- `Session::connect()` (Rust and Python) resolves the proxy socket via the `ARA2_SOCKET` environment variable, falling back to `DEFAULT_SOCKET`, so a deployment can point at a non-default proxy socket without a recompile. `ara2::socket_path()` / `ara2.socket_path()` expose the same resolution for callers that need the path as a string (e.g. a CLI flag's default). All examples and the test/bench helpers now use `Session::connect()` in place of `Session::create_via_unix_socket(DEFAULT_SOCKET)`.

### Changed

- The release build's `wheels` job grants `id-token: write` and `attestations: write`, so the wheels carry build provenance once the shared workflow attests them. Provenance is generated where an artifact was built rather than where it is later uploaded from, because provenance generated at the download describes the download. A reusable workflow cannot grant itself what the caller has not, so the permissions are granted here — ahead of the revision that needs them, where they are inert, rather than in the same change and risking the order.

## [0.18.0] - 2026-09-18

### Changed (BREAKING)

- **`edgefirst-decoder` and `edgefirst-codec` are no longer dependencies of the `ara2` library.** Decoding model outputs and loading images from disk are application concerns; the client library did neither. Each crate had exactly one point of contact — `edgefirst-decoder` typed `OutputSpec::dshape`, and `edgefirst-codec` supplied `Error::Codec` — and between them they pulled a YOLO decoder, an NMS implementation and a PNG decoder into every build of a crate that only talks to the NPU, along with `serde_yaml_ng`, `unsafe-libyaml`, `ndarray-stats`, `argminmax`, `noisy_float` and `rand`. `cargo tree -e normal -p ara2` now reaches `edgefirst-tensor` and `edgefirst-image` and nothing else from the HAL: 83 crates, down from 101.

  `edgefirst-codec` stays available to the examples and benches as a dev-dependency, so it never propagates to a consumer. Two new off-by-default features put both crates' types back into `ara2`'s own API for callers that want them:

  | Feature | Effect |
  |---------|--------|
  | `decoder` | Adds `OutputSpec::dshape_typed()`, returning `edgefirst_decoder::configs::DimName` pairs |
  | `codec` | Adds `Error::Codec` and `From<edgefirst_codec::CodecError> for Error` |

- **`OutputSpec::dshape` is `Vec<(String, usize)>`**, carrying the metadata's own axis spellings (`"batch"`, `"num_boxes"`, `"num_protos"`, ...) rather than `Vec<(edgefirst_decoder::configs::DimName, usize)>`. This is the shape the Python bindings have always exposed, for the same reason, and it preserves an axis name the HAL does not model, where `DimName` collapses it to `DimName::Unknown`.

  The type does not vary by feature. `decoder` adds `OutputSpec::dshape_typed()`, which returns the `DimName` pairs. Cargo unifies features across the whole dependency graph, so a feature that changed a public type would change it for every crate in the build the moment any dependency enabled it — a downstream crate written against the string form would stop compiling because something unrelated to it turned `ara2/decoder` on. A feature may add to the API; it must not reshape it.

- **`Error` is `#[non_exhaustive]`, and `Error::Codec` is behind the `codec` feature.** Nothing in the library constructed `Codec`; it existed so a caller could `?` an `edgefirst-codec` call inside a function returning `ara2::Error`. Because the variant set now depends on a feature, and features unify across the graph, an exhaustive `match` downstream could otherwise stop compiling because an unrelated dependency enabled `ara2/codec`. `#[non_exhaustive]` requires a wildcard arm and makes that a stable contract instead. `edgefirst-ara2` does not enable the feature, so no codec error can reach its exception mapping and the Python surface is unchanged.

- **`ara2`'s `camera` feature now implies `decoder`.** `yolov8_live` builds a HAL decoder, so the feature that builds it has to supply one.

- **`edgefirst-image` is taken with `default-features = false` plus `opengl` and `static`.** Its default set turns on `codec`, which would have pulled `edgefirst-codec` back in transitively and undone the trim.

### Changed (BREAKING) — release process

- **A tag no longer builds anything, and release tags are no longer created by hand.** `release.yml` triggered on `v*.*.*` and built the wheels there; a tag-triggered workflow cannot be run by a pull request, so those builds were unreviewable and their failures surfaced only once the tag existed. The release is now three workflows, each owning one action: `release.yml` **builds** every artifact on a push to `release/X.Y.Z`, `tag-release.yml` **tags** the merge commit when that branch's PR is merged, and `publish.yml` **publishes** what was already built when the tag appears. Accepting the release PR is the gate — it cannot merge until `release.yml` is green, and green means every artifact the release ships already exists. `publish.yml` builds nothing; the sole exception is `cargo publish --no-verify`, which has no pre-built input.

  **Both Trusted Publishers must be re-pointed from `release.yml` to `publish.yml` before the next tag.** They match on workflow filename, so the split breaks them until updated, and the dispatch rehearsal does not catch it because a rehearsal skips the upload. Re-point first, rehearse second, tag third. The checklist is in `.github/copilot-instructions.md`.

- **`Cargo.lock` is committed.** Every shared CI lane runs `--locked` — clippy, cross-clippy, nextest and `cargo publish` — and fails outright when the lockfile has to be created. A lockfile in a library is ignored by downstream consumers, so this changes nothing for anyone depending on `ara2`.

- **The toolchain is pinned to 1.94.0** in `rust-toolchain.toml`, with `rustfmt` and `clippy`. Clippy's lint set is a property of the compiler, so an unpinned toolchain turns CI red on a Rust release with no change to this repository. It also removes the `--exclude ara2-py` carve-out the old lint job carried: the whole workspace lints clean on the pinned compiler.

### Changed

- **EdgeFirst HAL crates updated to `0.32`** (`edgefirst-tensor`, `edgefirst-image`, `edgefirst-decoder`, `edgefirst-codec`). No API changes reach `ara2`; the release is fixes and hardening upstream — GL convert deadlocks on PBO-backed sources, Mach-O alignment in release builds, and `crossbeam-epoch` 0.9.21 clearing RUSTSEC-2026-0204.

- **The `yolov8` example declares `required-features = ["decoder"]`**, so it is built with `cargo build --features decoder --example yolov8`. CI lints and checks twice — once with `--features ara2/codec,ara2/decoder` and once with the defaults — because `required-features` would otherwise drop the example from `--all-targets` and the gated code would never be compiled.

### Changed — CI

- **CI is tiered, and the workflows are callers of the shared reusable workflows in `EdgeFirstAI/.github`, pinned by SHA** ([CICD Pipelines](https://au-zone.atlassian.net/wiki/spaces/EAM/pages/2750906369/CICD+Pipelines)). `test.yml`, `build.yml`, `python.yml` and `sbom.yml` are replaced by `ci.yml` and `nightly.yml`. The four removed workflows ran unconditionally on every push to every branch: nine jobs, a release build for two architectures and two manylinux2014 wheels, with no concurrency group, no required check and no path filtering.

  A push to a non-draft PR now runs **Quick** only: format, host clippy, aarch64 cross-clippy, the host-runnable tests, ruff, and the dependency licence policy, budgeted at ten minutes. **Full** — the aarch64 lane, the full scancode SBOM — runs when a reviewer adds `ci:full`, on a dispatch, or on a merge-queue batch. **Nightly** runs `cargo audit` every night regardless, and the rest only when `main` has moved since the last nightly that reached a verdict. Draft PRs run nothing. One check, `ci-gate`, is required.

- **`ruff` runs over the Python bindings and examples**, pinned to 0.16.7. `ruff.toml` declares only this repository's exceptions to ruff's default set: the four rules that object to a `.pyi` whose bodies are a docstring followed by `...`, which for a compiled extension module is the documentation surface, and `RUF022`, because `__all__` is grouped by comment section to match the module layout.

- **The `yolov8_live.py` example no longer imports `numpy`** (unused) and no longer assigns an unused `timing`. Its `pywayland` import probe now names every protocol type explicitly with `# noqa: F401`, so a partial install fails at the probe rather than several hundred lines into a camera loop. The Python examples became executable, matching their shebangs.

- **`edgefirst_ara2.pyi` drops `from __future__ import annotations`** (no effect in a stub) and corrects the context-manager signatures: `__enter__` returns `Self`, and `__exit__`'s first parameter is `type[BaseException] | None`.

### Added

- **Tests for `dshape` parsing on the default build.** The metadata spec writes a dshape as an array of single-key maps and serde's own form is an array of tuples; both are accepted, an unrecognised axis name is preserved rather than collapsed, a map entry naming two axes is rejected rather than resolved by iteration order, and an absent `dshape` is empty rather than an error.

## [0.17.0] - 2026-09-11

### Changed (BREAKING)

- **edgefirst-hal replaced by the modular EdgeFirst HAL crates at `0.31`.**
  `edgefirst-hal` was a meta-crate that re-exported the real libraries under
  `edgefirst_hal::{tensor, image, decoder, codec}`; upstream deleted it in
  HAL 0.29, and `0.28.3` is its last release. `ara2` now depends on
  `edgefirst-tensor`, `edgefirst-image`, `edgefirst-decoder` and
  `edgefirst-codec` directly. Each sub-crate exports its types at its own
  crate root, so the intermediate module segment disappears:
  `edgefirst_hal::tensor::TensorMemory` is now `edgefirst_tensor::TensorMemory`.
  These types appear in the public API surface of `ara2`, so this is a
  transitive break — a downstream crate pinning `edgefirst-hal` must migrate
  in lockstep or it will fail type unification at the `ara2` boundary.
- **`TensorMemory::Dma` renamed to `TensorMemory::DmaBuf`**, and the enum is
  now `#[non_exhaustive]` upstream (`IoSurface`, `Pbo` and `Cuda` joined it,
  with Windows D3D11 textures served under `DmaBuf`). A `match` on it needs a
  wildcard arm. The Python binding's `allocate_tensors(memory=...)` string is
  unchanged: `"dma"` still selects DMA-BUF, and `memory_type()` still reports
  `"dma"`, `"shm"` or `"mem"` — an unmodelled backing now reports its upstream
  wire name rather than being mislabelled.
- **`TensorMap<T>` is no longer public; use `HostView<'_, T>`.** The map guards
  `Model::build_blobs` holds are now `edgefirst_tensor::HostView<'static, u8>`,
  which implements the same `TensorMapTrait`.
- **`ara2`'s `camera` feature now also enables `edgefirst-image/decode`.**
  `ImageProcessor::draw_masks` — the fused decode-and-composite call
  `yolov8_live` uses — moved behind that feature upstream so a build that only
  converts images does not link the model decoder.

### Added

- **`EndpointStatistics` is exported from the crate root.** `Endpoint::statistics()`
  has been public since 0.15.0, but `mod endpoint` is private and the type was
  missing from the `pub use` list — so the return type of a public method could
  not be named downstream. Callable, but impossible to store in a struct field
  or write in a signature.

- **`OutputSpec.normalized` and `OutputSpec.dshape` on the Python bindings.**
  The Rust `dvm_metadata::OutputSpec` has carried both since the fields were
  added; `edgefirst_ara2` exposed neither, so a Python caller could not tell a
  spec-conforming export (normalized box coordinates, named axes) from an older
  one. `dshape` is returned as `(name, extent)` string pairs — the metadata
  spellings, which map onto `edgefirst.decoder.DimName` without `edgefirst-ara2`
  depending on the decoder wheel.

### Changed

- **Python examples migrated to the `edgefirst.*` namespace packages.**
  `edgefirst_hal` is replaced by `edgefirst.tensor`, `edgefirst.image`,
  `edgefirst.decoder` and `edgefirst.codec`, imported as `ef_tensor`,
  `ef_image`, `ef_decoder` and `ef_codec` so each name says which library it
  comes from. The rewrite also corrects call sites that had been stale since
  HAL 0.28 or earlier:
  - `hal.Rect` no longer exists. `ImageProcessor.convert(src, dst,
    letterbox=(r, g, b, a))` performs the aspect-preserving fit itself, so the
    destination rectangle is gone; `compute_letterbox` now returns only the
    normalized rect the mask calls need, computed to match the HAL's own
    `letterbox_rect` (long axis exact, short axis rounded half away from zero,
    centred) so the overlay registers with the pixels the GPU rendered.
  - `hal.Tensor.load_from_bytes` / `hal.load_image` are replaced by the
    allocate-then-decode pattern: `edgefirst.codec.Tensor.peek_image_info_file`
    for the native geometry, `ImageProcessor.create_image` to allocate, and
    `edgefirst.codec.decode_file_into` to decode in place.
  - `ImageProcessor.draw_masks(decoder=..., ...)` is now
    `Decoder.draw_onto(processor, model_output, dst, ...)`, returning the same
    `(boxes, scores, classes)` tuple — `edgefirst-image` no longer depends on
    `edgefirst-decoder`.
  - Cross-package objects travel by the `__edgefirst_tensor__` capsule
    protocol, so a tensor allocated by one `edgefirst.*` package is accepted by
    another without a copy.
- Documented the actual `ara2` feature table in `ARCHITECTURE.md`: it listed a
  `hal` feature that the crate has never had.

### Fixed

- **Python examples produced wrong detections**, found by running them against
  the EdgeFirst Model Zoo `ara240` DVMs on an i.MX 95 board. All three were
  pre-existing; the Rust examples were already correct in each case, and the
  Python ones now match them box-for-box.
  - *Boxes were `input_dim` times too small.* `build_decoder` divided the box
    quantization scale by the model input dimension unconditionally. That is
    only right for an older export emitting int-encoded pixel coordinates; a
    spec-conforming DVM declares `normalized: true` and already emits `[0, 1]`.
    The examples now read the flag (via the new `OutputSpec.normalized`) and
    divide only when the metadata is silent, as `yolov8.rs` does.
  - *Segmentation could not be decoded at all.* Outputs were declared with an
    anonymous `shape=`, and the decoder read the rank-4 proto tensor as
    trailing-channel: `materialize_masks` rejected the coefficients with
    "`[N, 32]` incompatible with protos `[32, 160, 160]`". Outputs are now
    declared with a named `dshape` — from the DVM metadata when present,
    otherwise a canonical Ultralytics naming mirroring `canonical_dshape` in
    `yolov8.rs`.
  - *Printed boxes did not match the drawn overlay.* The detection list scaled
    box coordinates straight by the image dimensions, ignoring the letterbox
    the decoder reports against, so `y` was off by the pad. `unletterbox()`
    now applies the same inverse transform `draw_decoded_masks` uses.
- **`yolov8.py` fed the model an extra colour conversion.** The source JPEG was
  decoded to NV12, converted to RGBA, and only then letterboxed into the model
  input. Converting the native frame straight to PlanarRGB removes a full-frame
  GPU pass *and* changes the answer — the extra chroma round trip moved boxes by
  up to ~20 px. The RGBA copy is now made once for the overlay's base layer
  only, and the example's detections match `yolov8.rs` exactly.
- Two `clippy::byte_char_slices` errors in `yolov8_live.rs` that only appear
  under `--features camera`, which CI's clippy job does not build.

### Fixed upstream (EdgeFirst HAL 0.31.0)

- **Python instance segmentation now works.** On 0.30.0 an int8 segmentation
  model raised `I8 mask_coefficients require quantization metadata` from
  `ImageProcessor.materialize_masks`: the `ProtoData` tensors did carry
  quantization, but the `__edgefirst_protodata__` capsule that hands them from
  `edgefirst.decoder` to `edgefirst.image` described each tensor with a
  `TensorDesc`, which has no quantization field, so `import_tensor_capsule`
  rebuilt them without it. `draw_proto_masks` and the fused
  `Decoder.draw_onto` crossed the same boundary and failed identically.
  Detection models were unaffected, and so was the Rust path, which has no
  capsule boundary to cross.

  0.31.0 carries quantization in the capsule payload (a `QuantDesc`, and the
  name moved to `edgefirst_tensor_v2`) and through the same-module
  `interop::reconstruct` path. Re-verified on an i.MX 95 board against the
  released wheels: `yolov8.py` with `yolov8n-seg-int16.dvm` reports the same
  four detections as `yolov8.rs`, mask overlay included. The examples needed
  no change.

### Documentation

- **The READMEs and all four `yolov8` examples now point at the EdgeFirst model
  zoo.** `README.md`, `examples/README.md`, `crates/ara2-py/README.md` and the
  module docs of `yolov8.rs`, `yolov8.py`, `yolov8_live.rs` and `yolov8_live.py`
  name the detection (<https://huggingface.co/EdgeFirst/yolov8-det>) and
  segmentation (<https://huggingface.co/EdgeFirst/yolov8-seg>) repositories,
  explain that the ARA-2 builds are the int16 `.dvm` exports under `ara240/`,
  and give `curl` lines for them. Usage samples name real zoo artifacts rather
  than placeholders like `model.dvm` or `yolov8n_640x640.dvm`, and `TESTING.md`
  says where to get a `.dvm` for the model tests. This matches the presentation
  the `tflite-rs` READMEs and examples use for the same zoo.

  The docs also record what the embedded `edgefirst.json` is *for* on this
  runtime: it carries the per-output `normalized` flag and named `dshape`,
  neither of which the NPU reports, and without which a model emitting
  pixel-space boxes is misread.

- **`README.md`'s example table was missing four of the ten examples** —
  `yolov8_live.rs`, `yolov8_live.py`, `async_multi_model.rs` and
  `async_multi_model.py`.

- **`cargo doc` is warning-free across the workspace.** Seven intra-doc links in
  `ara2` pointed at bare method names that do not resolve from a doc comment
  (`[`submit`]` → `[`Self::submit`]`, and similar). The remaining 142 came from
  `ara2-sys`, where bindgen copies the C header's Doxygen prose into `#[doc]`
  verbatim and rustdoc reads its `[in]`/`[out]`/`[unused]`/`[unsupported]`
  direction markers as links; those are allowed at the crate root rather than
  escaped in `ffi.rs`, since any edit there is lost when `update.sh`
  regenerates it.

### Migration

| 0.16.x | 0.17.0 |
|--------|--------|
| `ara2 = "0.16"` | `ara2 = "0.17"` |
| `edgefirst-hal = "0.28"` | `edgefirst-tensor`/`-image`/`-decoder`/`-codec` = `"0.31"` |
| `use edgefirst_hal::tensor::X` | `use edgefirst_tensor::X` |
| `use edgefirst_hal::image::X` | `use edgefirst_image::X` |
| `use edgefirst_hal::decoder::X` | `use edgefirst_decoder::X` |
| `use edgefirst_hal::codec::X` | `use edgefirst_codec::X` |
| `TensorMemory::Dma` | `TensorMemory::DmaBuf` (+ wildcard arm) |
| `TensorMap<u8>` | `HostView<'static, u8>` |
| `pip install edgefirst-hal` | `pip install edgefirst-codec edgefirst-decoder edgefirst-image` |
| `import edgefirst_hal as hal` | `import edgefirst.image as ef_image` (and siblings) |
| `processor.convert(s, d, dst_crop=r, dst_color=c)` | `processor.convert(s, d, letterbox=c)` |
| `processor.draw_masks(decoder=d, ...)` | `d.draw_onto(processor, ...)` |
| `Output.protos(shape=[1, 32, H, W])` | `Output.protos(dshape=[(Batch, 1), (NumProtos, 32), (Height, H), (Width, W)])` |
| `scale = qn / input_dim` (always) | divide only when `OutputSpec.normalized` is `None` |

## [0.16.0] - 2026-08-08

### Changed

- **edgefirst-hal** bumped from `0.27` to `0.28` (`0.28.0`). The HAL 0.28
  release splits the crate into published sub-crates (`edgefirst-tensor`,
  `edgefirst-image`, `edgefirst-gl`, `edgefirst-egl`, `edgefirst-decoder`,
  `edgefirst-codec`) re-exported under the same `edgefirst_hal` paths, and
  lands the crop-contract work (fused NV→planar source regions, GL planar
  heap destinations, padded destination views, tiling-path tracing spans).
  No ara2 API changes were required — the bump compiled clean with zero
  call-site fallout. Raising the floor keeps downstream consumers on a
  single HAL stack (a `^0.27` pin dual-resolves against HAL-0.28 users and
  fails type unification at the ara2 boundary).

## [0.15.0] - 2026-07-17

### Added

- **`Endpoint::statistics()`** returns live endpoint telemetry — operational
  state, system and DRAM clocks (MHz), core voltage (V), temperature (°C), and
  DRAM usage — via `dv_endpoint_get_statistics`. This exposes the dNPU's thermal
  and voltage sensors; the Ara SDK provides no power-in-watts reading, so
  temperature and core voltage are the available power-related telemetry. The
  `dv_endpoint_get_statistics`/`dv_endpoint_free_statistics` FFI and the
  `dv_endpoint_stats` struct were already bound in `ara2-sys`; this adds the
  safe high-level `EndpointStatistics` accessor.

### Fixed

- `Endpoint::statistics()` now guards against a null or empty buffer returned
  by the SDK instead of dereferencing it.

## [0.14.0] - 2026-07-16

### Changed

- **edgefirst-hal** bumped from `0.25` to `0.27` (`0.27.0`), spanning two
  upstream releases:
  - **`0.26.0` (breaking)**: image constructors now require an explicit
    `CpuAccess` declaration (`None` / `Read` / `Write` / `ReadWrite`)
    describing how the CPU touches the buffer. `Tensor::image`,
    `ImageProcessor::create_image`, and related constructors gain a trailing
    `CpuAccess` argument; hardware-only (`None`) buffers become eligible for
    vendor tile compression, and precise declarations pick up cheaper
    mappings than the previous implicit `ReadWrite` behaviour.
  - **`0.27.0`**: adds zero-copy SAHI-style input tiling for small-object
    detection (new `edgefirst-image`/`edgefirst-decoder` tiling APIs) —
    additive, no source changes required on top of the `0.26.0` migration.
  - Updated call sites in `yolov8.rs`, `yolov8_live.rs`, and
    `model_benchmark.rs`: `CpuAccess::Write` for JPEG/image decode targets,
    `CpuAccess::Read` for the static-image render canvas (read back via
    `save_jpeg`), and `CpuAccess::None` for the live-camera canvas, which is
    only ever GPU-drawn and handed to Wayland as a DMA-BUF.
  - `yolov8_live.rs` also picks up the `Crop::letterbox` /
    `with_letterbox_crop` / `import_image` colorimetry migration introduced
    in HAL `0.25` (see `0.13.0` below) that had not yet been applied there —
    the `camera` feature isn't built in CI, so it had drifted since.
- **Dependencies**: refreshed all other workspace dependencies to their
  latest compatible versions via `cargo update` (clap, criterion, ndarray,
  regex, tokio, wasm-bindgen, zerocopy, and their transitive graph). No
  further source changes required.

### Known limitations

- The `camera` feature (`yolov8_live.rs`) could not be compiled on the
  development host used for this release (no `libcamera-dev` installed) and
  is not covered by CI (`build.yml` only builds default features). Its HAL
  migration was applied by manual review mirroring the already-verified
  `yolov8.rs` pattern, not confirmed by a compiler; verify on a host with
  `libcamera-dev` before relying on it.

## [0.13.1] - 2026-06-23

### Changed

- **Dependencies**: refreshed all workspace dependencies to their latest
  versions. The Python-bindings crate (`ara2-py`) moves its PyO3 stack from
  `0.28` to `0.29`:
  - `pyo3` `0.28` → `0.29`
  - `pyo3-build-config` `0.28` → `0.29`
  - `numpy` `0.28` → `0.29`

  No source changes were required; the published Rust crates (`ara2`,
  `ara2-sys`) are unaffected and their public API is unchanged. Python wheel
  consumers should rebuild against the PyO3 0.29 ABI.

## [0.13.0] - 2026-06-17

### Changed

- **edgefirst-hal** bumped from `0.24` to `0.25` (`0.25.1`). The HAL 0.25
  release refactors codec and image-processing APIs:
  - `DecodeOptions` is removed — `peek_info` and `load_image` no longer accept
    decode options; the destination tensor's pre-allocated pixel format drives
    conversion automatically.
  - `Crop::letterbox(pad)` replaces the manual `with_dst_rect` / `with_dst_color`
    builder pattern for letterbox placement; geometry is now resolved internally
    via `Crop::resolve`.
  - `MaskOverlay::with_letterbox_crop` takes two additional arguments (`src_w`,
    `src_h`) so the library can resolve placement without requiring callers to
    pre-compute the letterbox rectangle.
  - `ImageProcessor::import_image` gains a trailing
    `colorimetry: Option<Colorimetry>` parameter for BT.709/BT.601 colour-space
    tagging; pass `None` to preserve the previous implicit behaviour.
  - The `Rect` placement type is now crate-private; public callers use `Region`
    from `edgefirst-tensor` instead.

  The `yolov8` example and the `model_benchmark` bench are updated accordingly.

## [0.12.0] - 2026-06-16

### Added

- **I/O rebind pool**: new `InputSet` and `OutputSet` types plus four new
  `Model` methods for continuous full-pipeline NPU saturation without a transit
  reserve:
  - `allocate_output_set(memory)` → `OutputSet`: allocates and DMA-registers
    one buffer per model output. Pass to `submit_with_output_set` so a decoder
    can hold some sets while the NPU infers into others.
  - `allocate_input_set(memory)` → `InputSet`: symmetric analogue for inputs;
    tensors are shaped `[C, H, W]` to match `allocate_tensors` and are
    compatible with HAL's `TensorImageRef` for zero-copy GPU preprocessing.
  - `submit_with_output_set(output_set)`: infers using the model's own input
    tensors and writes outputs into an `OutputSet`.
  - `submit_with_io_set(input_set, output_set)`: fully decoupled path — both
    input and output bypass the model's `allocate_tensors` buffers.

  Both types are `Send`; raw DMA descriptor pointers are valid for the session
  lifetime enforced by an internal `Arc` keep-alive.

### Fixed

- `submit_with_output_set` and `submit_with_io_set` now validate session
  identity, tensor count, and per-tensor byte sizes before calling
  `dv_infer_async`, returning a typed error instead of undefined behaviour
  on a mismatched set.
- `allocate_input_set` now allocates with `input_shape()` (`[C, H, W]`)
  instead of a flat `[size]` slice, matching the layout `allocate_tensors`
  produces and enabling `TensorImageRef` compatibility.

## [0.11.2] - 2026-05-28

### Changed

- **edgefirst-hal** bumped from 0.24.0 to 0.24.2 (transitive update via
  the workspace's `edgefirst-hal = "0.24"` semver constraint). The 0.24.2
  release fixes a `GL_TEXTURE_SWIZZLE_R` leak in the GL backend's
  RGBA → PlanarRgb conversion path
  ([EdgeFirstAI/hal#84](https://github.com/EdgeFirstAI/hal/pull/84)) —
  the swizzle was left at `GL_BLUE` after the last iteration of the
  per-channel loop, then inherited by the next `draw_decoded_masks`
  call's bg blit and observed as `canvas.R := src.B` across the entire
  overlay on NXP Vivante GC7000 and Mali Valhall targets. The yolov8
  example's saved overlay JPEG now renders in natural colours on
  imx8mp-frdm and imx95-frdm.

### Added

- The `yolov8` example now synthesises a canonical Ultralytics `dshape`
  per output role (`Detection`, `Boxes`, `Scores`, `MaskCoefficients`,
  `Protos`) when the `.dvm` file does not ship an `edgefirst.json`
  metadata block — for example the official Kinara 1.2.1 exports
  (`yolov8n-kinara-1.2.1.dvm`, `yolov8n-seg-kinara-1.2.1.dvm`) which
  have no zip footer. Without this fallback, the seg path errored at
  mask materialisation with
  `mask_coefficients [N, 32] incompatible with protos [32, 160, 160]
  (expected [N, 160])` because the HAL decoder fell back to "shape is
  already canonical" and read the NCHW proto tensor as if it were
  NHWC. Verified end-to-end with both detection and segmentation Kinara
  exports on imx8mp-frdm and imx95-frdm.

## [0.11.1] - 2026-05-28

### Added

- **`ara2::dvm_metadata::OutputSpec`** now surfaces the per-output
  fields the EdgeFirst metadata spec (`metadata.md §Output
  Specification`) already requires producers to emit but `ara2`
  previously ignored:
  - `dshape: Vec<(DimName, usize)>` — physical axis names in memory
    order, parsed via `edgefirst_decoder::configs::deserialize_dshape`.
    Lets the HAL decoder stride-swap an NCHW physical tensor into its
    canonical NHWC view without copying bytes.
  - `normalized: Option<bool>` — `true` when box coords are in
    `[0, 1]`, `false` for pixel space; per the spec, meaningful only
    on `boxes` / `detections` outputs.
  - `encoding: Option<String>` — `direct`, `dfl`, `anchor` on
    `boxes` outputs.
  - `score_format: Option<String>` — `per_class` or `obj_x_class` on
    `scores` outputs.
  - `quantization: Option<QuantizationSpec>` — per-tensor
    `(scale, zero_point, dtype)`. Older `.dvm` files without these
    fields deserialize with `None` / empty defaults.
- **`ara2::dvm_metadata::QuantizationSpec`** — new public struct
  (`scale: f32`, `zero_point: i32`, `dtype: Option<String>`) carrying
  per-tensor quantization parameters from the JSON.

### Changed

- The `yolov8` example is now metadata-driven. It indexes
  `dvm_metadata::OutputSpec` entries by their (trailing-`1`-stripped)
  shape, then passes the JSON-declared `dshape` and `normalized`
  through to the HAL decoder's `Boxes`, `Scores`,
  `MaskCoefficients`, and `Protos` configs. Legacy `.dvm` files
  without these fields still work: an empty `dshape` is treated by
  the decoder as "shape is already in canonical order", and a missing
  `normalized` flag falls back to the prior `qn / input_dim`
  heuristic for the box quant scale.
- The example carries a tactical substitution
  (`DimName::NumFeatures → DimName::NumProtos` for outputs whose
  `output_type == "mask_coefs"`) for a converter regression seen in
  ara2 1.7.0–1.7.3 exports where the channel axis was misnamed.
  Spec-compliant exports (ara2 ≥ 1.7.4) declare `num_protos`
  directly, so the substitution is a no-op on current `.dvm` files.

### Fixed

- (example) Detection bounding boxes from spec-compliant `.dvm`
  files collapsing to a sub-pixel region near the origin. Root
  cause: the example unconditionally divided the box quantization
  scale by `input_dim` on the assumption that the model emitted
  pixel-space coords, but spec-compliant exports already emit
  normalized coords. The pre-divide is now gated on the metadata's
  `boxes.normalized` flag — applied only when the field is absent
  (legacy behaviour).
- (example) Segmentation pipeline erroring at `materialize_masks`
  with `mask_coefficients [N, 32] incompatible with protos
  [32, 160, 160] (expected [N, 160])`. Root cause: the example did
  not declare `dshape` for the proto / mask-coeff configs, so the
  HAL materializer interpreted the NCHW physical layout
  `[batch, num_protos, height, width]` as NHWC and selected the
  wrong axis as `num_protos`. With the metadata-driven `dshape`,
  the decoder stride-swaps into its canonical NHWC view (no byte
  copy) before mask materialisation.

## [0.11.0] - 2026-05-26

### Changed (BREAKING)

- **edgefirst-hal** upgraded from 0.23.0 to 0.24.1. HAL types
  (`Tensor<u8>`, `TensorMemory`, `edgefirst_hal::tensor::Error`,
  `edgefirst_hal::codec::CodecError`) appear in the public API surface
  of `ara2`, so this is a transitive ABI break: downstream crates
  pinning `edgefirst-hal = "0.23"` must bump in lockstep. The HAL
  release brings an optimized `edgefirst-codec` with full DMA and
  strided loading support for `Tensor::load_image`; no source changes
  were required in `ara2` or its examples — the workspace builds
  cleanly against the new HAL with the pin bump alone.
- **zip** upgraded from 2.4 to 8.6 (six major versions). The
  `Error::Zip` variant wraps `zip::result::ZipError`, which is part
  of the public API surface, so downstream `match` arms on the
  `ZipError` variants may need review. The subset of the zip API
  used by `ara2` (`ZipArchive::new`, `by_name`, the
  `InvalidArchive` and `FileNotFound` variants, `ZipWriter`,
  `SimpleFileOptions`, `CompressionMethod::Stored`) is stable across
  the bump — `ara2` itself required no source changes and all
  `dvm_metadata` tests pass against the new version.

### Changed

- Refreshed `Cargo.lock` to pick up semver-compatible updates across
  the dependency tree (`log` 0.4.29 → 0.4.30, `tokio` 1.52.1 → 1.52.3,
  `serde_json` 1.0.149 → 1.0.150, `jiff` 0.2.24 → 0.2.27, `nalgebra`
  0.34 → 0.35, `simba` 0.9 → 0.10, `wasm-bindgen` 0.2.118 → 0.2.122,
  and others). Collapsed 15 transitive `glam` versions into a single
  0.33.0.

### Migration

| 0.10.x | 0.11.0 |
|--------|--------|
| `ara2 = "0.10"` | `ara2 = "0.11"` |
| `edgefirst-hal = "0.23"` (downstream pin) | `edgefirst-hal = "0.24"` |
| `zip = "2"` (if matched on `Error::Zip(_)`) | `zip = "8"` |

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

[Unreleased]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.18.0...HEAD
[0.18.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.14.0...v0.15.0
[0.14.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.13.1...v0.14.0
[0.13.1]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.13.0...v0.13.1
[0.13.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.11.2...v0.12.0
[0.11.2]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/EdgeFirstAI/ara2-rs/compare/v0.10.0...v0.11.0
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
