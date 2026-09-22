# ARA-2 Client Library — Development Instructions

## Project Overview

Rust client library for the ARA-2 neural network accelerator (Kinara hardware).
Communicates with NPU devices via the ARA-2 proxy system service
(`rt-sdk-ara2.service` on EdgeFirst Yocto images) through FFI
bindings to `libaraclient`, dlopen'd by trying each name in
`ara2::LIBRARY_NAMES` (NXP has shipped it as `libaraclient_aarch64.so`,
`libaraclient_x86_64.so`, and `libaraclient.so.1` across SDK drops).

### Workspace Structure

| Crate | Purpose | Publish Target |
|-------|---------|---------------|
| `ara2` | Core client library (Session, Endpoint, Model) | crates.io |
| `ara2-sys` | FFI bindings to libaraclient.so via libloading | crates.io |
| `ara2-py` | Python bindings via PyO3 | PyPI |

### Key Types

```
Session → Endpoint → Model → run() → ModelTiming
```

- **Session**: Connection to the ARA-2 proxy via UNIX or TCP socket. Arc-based, cheaply cloneable.
- **Endpoint**: A single ARA-2 NPU device. Check status, DRAM stats, load models.
- **Model**: A loaded DVM neural network. Allocate tensors, set inputs, run inference, read outputs.

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
2. Proxy service running: `systemctl status rt-sdk-ara2`
3. ARA-2 device visible: `lspci | grep -i kinara`
4. Proxy socket available: `ls -la /var/run/proxy.sock`

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
| `session` | Yes | None | Needs the proxy running |
| `endpoint` | Yes | None | Needs the proxy + NPU |
| `model` | Yes | `ARA2_TEST_MODEL` | Needs a compiled .dvm file |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARA2_SOCKET` | Proxy socket path; overrides `DEFAULT_SOCKET` for `Session::connect()` |
| `ARA2_TEST_MODEL` | Path to a `.dvm` model file for model tests |
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
- FFI errors use `From<dv_status_code>` converting non-zero return codes

### Unsafe Code

All unsafe blocks are in the FFI layer. When adding new FFI calls:
1. Add the C function signature to `crates/ara2-sys/src/ffi.rs` (generated by bindgen)
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
- **`nextest-args` carries `-E 'test(dvm_metadata)'`.** Every other test needs
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
