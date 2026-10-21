# ARA-2 Client Library — Development Instructions

## Project Overview

Rust client library for the ARA-2 neural network accelerator (Kinara hardware).
Communicates with NPU devices via the ARA-2 proxy system service
(`rt-sdk-ara2.service` on EdgeFirst Yocto images) through FFI
bindings to `libaraclient`, dlopen'd by trying each name in
`ara2::LIBRARY_NAMES` (NXP has shipped it as `libaraclient_aarch64.so`,
`libaraclient_x86_64.so`, and `libaraclient.so.1` across SDK drops).

Three DVAPI versions are supported: 1.1 (Kinara runtime 1.2.1, packaged by meta-kinara), 1.3 (NXP rt-sdk-ara2 2.1.1) and 3.0 (arart r3.0). Their struct layouts differ (3.0 inserts `hw_cycles` mid-`dv_model`), so the loaded library's version is probed once per session as an `ara2::Abi` and every version-specific struct is read at that version's layout.

### DVAPI layouts

- `ara2-sys` declares the API itself in `crates/ara2-sys/include/`: `dvapi_common.h` (crate root; version-specific structs opaque) and `dvapi_v1_1.h`, `dvapi_v1_3.h`, `dvapi_v3_0.h` (modules `v1_1`, `v1_3`, `v3_0`, each struct in full). No SDK header is committed; our headers carry the Au-Zone Apache-2.0 SPDX header.
- In our headers: every enumerator has an explicit `= value`; every prototype is on one line; inside `dvapi_v*.h`, version-specific structs are referenced by `struct tag`, never by their `_t` typedef (which names the opaque common type).
- **Only `crates/ara2/src/raw.rs` reads version-specific structs**, through `with_layout!`. Nowhere else may dereference a `dv_model`, `dv_endpoint_statistics` or similar.
- After editing a header, run `crates/ara2-sys/update.sh`, then `update.sh --verify <1.1|1.3|3.0> <sdk-include-dir>` for every affected version (see TESTING.md), and update the pins in `crates/ara2-sys/src/lib.rs`.
- `ARA2_ABI` is refused (`Error::AbiOverrideRejected`, and the library search stops there) when it contradicts the reported version: a different recognised version, or an unrecognised version of another major. Forcing the recognised version itself, forcing when the probe failed, and forcing the same major over an unrecognised minor are accepted.

### Connection

`Session::connect()` resolves one endpoint and makes one attempt (a failed attempt can poison the 1.1 client): `ARA2_SOCKET` (`/path`, `unix:/path`, `tcp:IPV4:PORT`), then the first existing `SOCKET_PATHS` entry, then `discover()` preferring UNIX over TCP. `ara2::proxy_endpoint()` reports the choice without connecting. TCP sessions cannot pass file descriptors: `supports_fd_passing()` is false, `None` memory allocates heap, and explicit `Shm`/`DmaBuf` fails with `Error::FdPassingUnavailable`.

### Workspace Structure

| Crate | Purpose | Publish Target |
|-------|---------|---------------|
| `ara2` | Core client library (Session, Endpoint, Model) | crates.io |
| `ara2-sys` | FFI bindings to libaraclient.so via libloading | crates.io |
| `ara2-py` | Python bindings via PyO3 | PyPI |

### Key Types

```
Session → Endpoint → Model → run() → InferenceStats
```

- **Session**: Connection to the ARA-2 proxy via UNIX or TCP socket (fd-backed tensors need UNIX). Arc-based, cheaply cloneable.
- **Endpoint**: A single ARA-2 NPU device. Check status, DRAM stats, load models.
- **Model**: A loaded DVM neural network. Allocate tensors, set inputs, run inference, read outputs.

### Telemetry

`telemetry.rs` holds the telemetry types. A counter is exposed only if at least one runtime populates it on hardware, and a field the loaded DVAPI version lacks is `None`, never zero. Fields that exist in only some layouts are read through the `VersionFields` trait in `raw.rs` (defaults return `None`; the layouts that have the field override). Calls that exist in only some versions (`is_busy`, `set_log_level`) return `Error::Unsupported` elsewhere. Per-model statistics, queue statistics, the endpoint power state and power-state control are not exposed, because the runtime does not yet report or apply them reliably. Inference timestamps are `Some` only on DVAPI 3.0, the one runtime that stamps `CLOCK_MONOTONIC`. See ARCHITECTURE.md.

### Ownership Model

All types use `Arc<SessionInner>` for shared ownership instead of lifetimes.
This enables Python bindings and cross-thread usage without lifetime issues.

## Build

### Prerequisites

- Rust **stable** toolchain (edition 2024)
- For on-target: `libaraclient` (Kinara ARA-2 SDK, see `ara2::LIBRARY_NAMES` for the names tried)
- For Python: `maturin`, Python 3.11+

### Native Build

```bash
cargo build --release
```

### Cross-Compile for aarch64 (NXP i.MX)

```bash
cargo zigbuild --release --target aarch64-unknown-linux-gnu
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `codec` | no | Adds `Error::Codec` plus `From<edgefirst_codec::CodecError>` |
| `decoder` | no | Adds `OutputSpec::dshape_typed()`, returning `edgefirst_decoder::configs::DimName` pairs. `dshape` itself is always `Vec<(String, usize)>` — a feature must not reshape a public type, because Cargo unifies features graph-wide |
| `camera` | no | Build the libcamera-based live-inference example (`yolov8_live`); implies `decoder` and turns on `edgefirst-image/decode` for its fused `draw_masks` call |

`edgefirst-tensor` and `edgefirst-image` are required dependencies — the
`Model` API exposes `Tensor<u8>` / `TensorMemory` in its public surface, so
there is no FFI-only build mode. `edgefirst-decoder` and `edgefirst-codec`
are not: they belong to the examples, and `cargo tree -e normal -p ara2`
must not reach either. Keep it that way — `edgefirst-image` is deliberately
taken with `default-features = false` because its default set turns on
`codec`.

Because both features are off by default, lint and check with them on as
well, or the gated code and the `yolov8` example (which
`required-features = ["decoder"]` would skip) never get compiled:

```bash
cargo clippy --workspace --exclude ara2-py --all-targets --features ara2/codec,ara2/decoder -- -D warnings
cargo clippy --workspace --exclude ara2-py --all-targets -- -D warnings
```

### Python Wheel

```bash
cd crates/ara2-py
maturin build --release --features pyo3/abi3-py311
```

## Dependencies

### Runtime (on-target)

- `libaraclient` — Kinara client library, under one of `ara2::LIBRARY_NAMES`. Must be on `LD_LIBRARY_PATH` or in system lib dirs.
- ARA-2 proxy — System service providing NPU access, `rt-sdk-ara2.service` on
  EdgeFirst Yocto images. Must be running.

### Crate Dependencies

| Dependency | Scope | Purpose |
|-----------|-------|---------|
| `edgefirst-tensor` | library | Tensor memory management (DMA/SHM/heap) |
| `edgefirst-image` | library | Hardware-accelerated image conversion and overlay rendering |
| `edgefirst-decoder` | examples, `decoder` feature | YOLO/ModelPack output decoding, NMS, segmentation masks |
| `edgefirst-codec` | examples, benches, `codec` feature | JPEG/PNG decode into a pre-allocated tensor |
| `libloading` | library | Dynamic loading of libaraclient (name varies by SDK drop, see `ara2::LIBRARY_NAMES`) |
| `ndarray` | N-dimensional array operations for tensor data |
| `serde` / `serde_json` | DVM metadata parsing |
| `zip` | Reading embedded metadata from DVM files |

## Testing

### On-Target Test Requirements

All tests require an NXP i.MX + ARA-2 PCIe system with:

1. `libaraclient` installed and accessible under one of `ara2::LIBRARY_NAMES`
2. Proxy service running (`rt-sdk-ara2`, `ara2`, or the board's r3.0 unit)
3. ARA-2 device visible: `lspci | grep -i kinara`
4. Proxy endpoint reachable: `Session::connect()` finds it, or set `ARA2_SOCKET`

### Board matrix

| Board and runtime | DVAPI | Library | Proxy |
|-------------------|-------|---------|-------|
| i.MX 8M Plus FRDM with Kinara runtime 1.2.1 | 1.1 | system | UNIX `/var/run/ara2.sock` |
| i.MX 95 board with NXP rt-sdk-ara2 2.1.1 | 1.3 | `/usr/lib/libaraclient_aarch64.so` | UNIX `/var/run/proxy.sock` |
| i.MX 95 FRDM with arart r3.0 | 3.0 | `LD_LIBRARY_PATH=/opt/ara2/art/linux/aarch64/lib` | TCP `127.0.0.1:5000` by default |

Ask the user before using a shared test board, and before changing a board's proxy service or configuration; restore any such change afterwards. On an r3.0 install whose unit runs chip bring-up before the proxy, restarting the unit or warm-rebooting the board fails on an already-initialised chip; a proxy configuration change needs a power cycle. Run the hardware suite with `--test-threads=1`. Install Python wheels on boards into a virtual environment, never the system Python.

### Running Tests

```bash
# Run all tests (must be on-target with hardware)
cargo test -p ara2

# Run only metadata tests (no hardware needed)
cargo test -p ara2 dvm_metadata

# Run model tests (needs a .dvm file)
ARA2_TEST_MODEL=/path/to/model.dvm cargo test -p ara2 model

# Run with nextest
cargo nextest run -p ara2
```

### Test Categories

| Category | Hardware | Env Vars | Notes |
|----------|----------|----------|-------|
| `dvm_metadata` | No | None | Pure data parsing tests |
| `offline` | No | None | Hardware-free logic; name new host-only tests `offline` (or put them in `offline_tests`) so CI runs them |
| `session` | Yes | None | Needs the proxy running |
| `endpoint` | Yes | None | Needs the proxy + NPU |
| `model` | Yes | `ARA2_TEST_MODEL` | Needs a compiled .dvm file |
| `telemetry` | Yes | `ARA2_TEST_MODEL` | Statistics, info, busyness and per-version `None` handling; Python: `TestTelemetry` in `tests/test.py` |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARA2_SOCKET` | Pins the proxy endpoint for `Session::connect()`: a socket path, `unix:/path` or `tcp:IPV4:PORT` |
| `ARA2_ABI` | Forces a layout (`1.1`, `1.3`, `3.0`) under the same-major rule |
| `ARA2_SDK_V1_1` / `_V1_3` / `_V3_0`, `ARA2_SDK_REQUIRE` | SDK include directories for `ara2-sys`'s `verify_sdk` test |
| `ARA2_TEST_MODEL` | Path to a `.dvm` model file for model tests |
| `ARA2_TEST_FOREIGN_MODEL` | Path to a `.dvm` built for the other runtime generation, for the load-failure test |
| `ARA2_TEST_FORMAT_FILES` | `path=format` pairs for the offline real-file format test |
| `RUST_LOG` | Log level: `debug`, `info`, `warn`, `error` |

## On-Target Debugging

### Verifying Hardware Setup

```bash
# Check PCIe device
lspci | grep -i kinara

# Check proxy service
systemctl status rt-sdk-ara2
journalctl -u rt-sdk-ara2 --no-pager -n 50

# Check socket
ls -la /var/run/proxy.sock
```

### Debug Logging

```bash
RUST_LOG=debug cargo test -p ara2 -- --nocapture
```

### Common Error Codes (dv_status_code)

| Code | Name | Meaning |
|------|------|---------|
| 0 | `DV_SUCCESS` | Operation succeeded |
| 1 | `DV_FAILURE_UNKNOWN` | Unknown failure |
| 100 | `DV_ENDPOINT_OUT_OF_MEMORY` | NPU DRAM full, unload models |
| 200 | `DV_RESOURCE_NOT_FOUND` | Invalid handle or missing resource |
| 220 | `DV_SESSION_UNIX_SOCKET_FILE_TOO_LONG` | Socket path exceeds limit |
| 230 | `DV_ENDPOINT_INVALID_HANDLE` | Stale endpoint reference |
| 240+ | `DV_MODEL_*` | Model loading/inference errors |
| 300+ | `DV_ERROR_CATEGORY_SW_CLIENT_FATAL` | Client library crash |
| 400+ | `DV_ERROR_CATEGORY_SW_SERVER_FATAL` | Proxy crash |
| 500+ | `DV_ERROR_CATEGORY_HW_FATAL` | Hardware failure |

### SSH to Target

```bash
ssh root@<target-ip>
# Deploy binary
scp target/aarch64-unknown-linux-gnu/release/libara2.so root@<target-ip>:/usr/lib/
```

## Code Conventions

### Rust Style

- **Edition 2024** on stable Rust
- **Formatting**: `cargo fmt --all` (config in `rustfmt.toml`)
- **Linting**: `cargo clippy --workspace --all-targets -- -D warnings`

### Error Handling

The `Error` enum in `crates/ara2/src/error.rs` covers all error types. Follow the pattern:
- Add a variant to `Error`
- Add a `From<T>` impl
- Add a `Display` match arm
- A non-zero DVAPI status becomes `Error::Ara2` through `SessionInner::status_error` (or `status::status_error_from` before a session exists), which names it with the library's `dv_stringify_status_code`. There is no `From<dv_status_code>` conversion.

### Unsafe Code

All unsafe blocks are in the FFI layer. When adding new FFI calls:
1. Declare the C prototype in `crates/ara2-sys/include/` (common header if identical in every version, else each `dvapi_v*.h`), regenerate with `update.sh`, and verify against the SDKs
2. Wrap in a safe Rust function in the appropriate module
3. Check return codes and convert to `Error`
4. Document the safety invariant in a `// Safety:` comment

### Commits

- Sign all commits with `-s` (DCO)
- Conventional commits: `feat:`, `fix:`, `test:`, `chore:`, `docs:`

## CI/CD

CI is tiered, and the workflows are callers of reusable workflows in
[`EdgeFirstAI/.github`](https://github.com/EdgeFirstAI/.github) pinned by SHA.
Local files declare inputs, never steps. The design is
[CICD Pipelines](https://au-zone.atlassian.net/wiki/spaces/EAM/pages/2750906369/CICD+Pipelines)
in the EAM space; read it before changing a workflow.

### Workflows

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | PR push, `merge_group`, push to `main`, dispatch | Quick on every non-draft push; Full only on a `ci:full` label, a dispatch, or a merge-queue batch. One required check, `ci-gate`. |
| `nightly.yml` | 03:17 daily, dispatch | `cargo audit` every night regardless; Full plus the feature-combination lane only when `main` has moved since the last nightly that reached a verdict. |
| `release.yml` | push to `release/*.*.*` | **Builds** every artifact: version check, CHANGELOG check, full SBOM, `.crate` packages, wheels. Publishes nothing. |
| `tag-release.yml` | `release/*.*.*` PR merged to `main` | **Tags.** Creates the annotated `vX.Y.Z` tag, and refuses if `release.yml` was not green for that commit. |
| `publish.yml` | push of a `v*.*.*` tag, dispatch | **Publishes** what `release.yml` already built: crates.io, PyPI, GitHub Release. Builds nothing. |

Three rules are not negotiable, because the shared workflows assume them:

- **`Cargo.lock` is committed.** Every shared lane runs `--locked` — clippy,
  cross-clippy, nextest and `cargo publish`. A missing lockfile fails the lane
  before it runs anything.
- **The toolchain is pinned** in `rust-toolchain.toml` (1.94.0). The lint set
  is a property of the compiler, so an unpinned toolchain turns CI red on a
  release with no change to this repository.
- **A tag deploys; it never builds.** See the Publishing section.

### Quick tier notes specific to this repository

- **`PYO3_CROSS_PYTHON_VERSION`** is set through the `pre-command` hook. The
  shared cross-clippy lane lints the whole workspace with hardcoded arguments,
  so `ara2-py` reaches it, and pyo3 refuses to configure a cross build without
  an interpreter for the target. Nothing is executed; naming the version is
  enough.
- **`nextest-args` carries `-E 'test(dvm_metadata) | test(offline) | binary(verify_sdk)'`.** `verify_sdk` skips each SDK check when no SDK is set, so only its parser tests run. Every other test needs
  an ARA-2 NPU and a running proxy. They are deliberately not
  `#[ignore]`d, because on a target they are the point of the suite.
- **No `clippy-args` override.** `codec` and `decoder` are off by default, so
  the stock pass covers the default build; the cfg-gated arms and the `yolov8`
  example are reached by the nightly feature-combination lane
  (`cargo hack check --feature-powerset --depth 2 --exclude-features camera`).
- **`ruff` is pinned to 0.16.7** and `ruff.toml` declares only this
  repository's exceptions to ruff's default rule set.

### Test Gap

Hardware-dependent tests (session, endpoint, model) need an NXP i.MX host with
an ARA-2 PCIe card. No such runner exists in the fleet, so `ci.yml` passes
`lanes: host` with no `boards` input and the `ci:hardware` label escalates to
the same host lanes as `ci:full`. When [EDGEAI-1577](https://au-zone.atlassian.net/browse/EDGEAI-1577)
provisions one, add its label to `boards` and switch `lanes` to the template's
conditional `all`/`hardware`.

## Publishing

**A tag deploys; it never builds.** A tag-triggered workflow cannot be run by
a pull request, so anything it builds is built at the one point in the process
where nothing can test it first, and its failures are found only after the tag
exists. The release is therefore three workflows, each owning one action, and
the only build sits on a branch a pull request can see.

### Release Checklist

All workspace crates share a single version.

1. **Branch**: `git switch -c release/X.Y.Z` (or `release/X.Y.Z-rcN`).
2. **Update version** in `Cargo.toml` (root):
   - `workspace.package.version`
   - `workspace.dependencies.ara2.version`
   - `workspace.dependencies.ara2-sys.version`
   - then `cargo check` so `Cargo.lock` picks the new version up, and commit it.
3. **Update `CHANGELOG.md`**: add `## [X.Y.Z] - YYYY-MM-DD` under
   `[Unreleased]`, add the comparison link, update the `[Unreleased]` link.
   `release.yml` fails without this section, and `publish.yml` turns it into
   the GitHub Release body.
4. **Push the branch.** `release.yml` runs on the push and builds everything
   the release will ship. Open the PR into `main` and label it `ci:full`.
5. **Rehearse the publish** once per repository, and again whenever
   `publish.yml` changes: dispatch `publish.yml` with the tag name. It
   resolves the build, verifies the tree and version, downloads the artifacts
   and publishes nothing.
6. **Merge the release PR** once Full and `release.yml` are both green.
   `tag-release.yml` creates the annotated `vX.Y.Z` tag at the merge commit.
7. `publish.yml` fires on the tag and publishes what step 4 built.

Do not create release tags by hand. The tag is the deployment trigger, and a
hand-made tag points at artifacts that do not exist.

### Pre-releases

`release/X.Y.Z-rcN` runs the same machinery. If the manifests carry the base
version (`X.Y.Z`) rather than the pre-release spelling, `publish.yml` attaches
the artifacts to a GitHub pre-release and **skips crates.io and PyPI**, so the
final release can still claim that version. Give the manifests the
pre-release version if the candidate should reach the registries — note that
PEP 440 wants `1.0.0rc1` and crates.io wants `1.0.0-rc.1`.

### Python Wheel

The Python version is derived from `Cargo.toml` via `dynamic = ["version"]`
in `pyproject.toml` — no manual sync needed. Wheels are built with maturin+zig
for manylinux2014 compatibility.

```bash
# Development install
cd crates/ara2-py
maturin develop --release

# Build release wheel, as release.yml builds it
maturin build --release --locked -m crates/ara2-py/Cargo.toml \
    --zig --compatibility manylinux2014
```

### Trusted Publishing

Publishing uses OIDC trusted publishing, with no API tokens. Both publishers
match on the workflow **filename**, so the release-chain split moved the
identity: they authenticate `publish.yml`, not `release.yml`.

| Registry | Environment | Trusted publisher workflow |
|----------|-------------|---------------------------|
| crates.io | `crates-io` | EdgeFirstAI/ara2-rs, `publish.yml` |
| PyPI | `pypi` | EdgeFirstAI/ara2-rs, `publish.yml` |

PyPI matches `job_workflow_ref` in the publishing repository, so a reusable
workflow in `EdgeFirstAI/.github` cannot be the Trusted Publisher. The wheel
upload therefore stays in this repository's `publish.yml` as the
`publish-pypi` job, while crates.io — which matches the *caller's*
`workflow_ref` — publishes from inside the shared workflow.

`publish.yml` runs `cargo publish --no-verify`. `--no-verify` skips the
verification compile, which is the only part of `cargo publish` that would be
a build on the tag path; the same tree was compiled and tested by Full and
packaged by `release.yml`. Crates go out in dependency order — `ara2-sys`,
then `ara2` — because crates.io is the one target where a partial publish
cannot be undone by re-running the job.

### Trusted Publishing Setup

1. On crates.io: each crate → Settings → Trusted Publishers → repository
   `EdgeFirstAI/ara2-rs`, workflow `publish.yml`, environment `crates-io`.
2. On PyPI: `edgefirst-ara2` → Settings → Publishing → repository
   `EdgeFirstAI/ara2-rs`, workflow `publish.yml`, environment `pypi`.
3. On GitHub: Settings → Environments → `crates-io` and `pypi` must exist.
4. Organisation secret `RELEASE_TAG_TOKEN` must be readable by this
   repository, or `tag-release.yml` cannot create the tag.

**If a publisher still names `release.yml`, the publish fails at the upload
step with a claim mismatch, after the release PR has already merged and the
tag exists.** The dispatch rehearsal does not catch it, because a rehearsal
skips the upload. Re-point first, rehearse second, tag third.
