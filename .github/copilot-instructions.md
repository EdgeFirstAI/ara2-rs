# ARA-2 Client Library — Development Instructions

## Project Overview

Rust client library for the ARA-2 neural network accelerator (Kinara hardware).
Communicates with NPU devices via the `ara2-proxy` system service through FFI
bindings to `libaraclient.so.1`.

### Workspace Structure

| Crate | Purpose | Publish Target |
|-------|---------|---------------|
| `ara2` | Core client library (Session, Endpoint, Model) | crates.io |
| `ara2-sys` | FFI bindings to libaraclient.so via libloading | crates.io |
| `ara2-py` | Python bindings via PyO3 (excluded from workspace) | PyPI |

### Key Types

```
Session → Endpoint → Model → run() → ModelTiming
```

- **Session**: Connection to ara2-proxy via UNIX or TCP socket. Arc-based, cheaply cloneable.
- **Endpoint**: A single ARA-2 NPU device. Check status, DRAM stats, load models.
- **Model**: A loaded DVM neural network. Allocate tensors, set inputs, run inference, read outputs.

### Ownership Model

All types use `Arc<SessionInner>` for shared ownership instead of lifetimes.
This enables Python bindings and cross-thread usage without lifetime issues.

## Build

### Prerequisites

- Rust **stable** toolchain (edition 2024)
- For on-target: `libaraclient.so.1` (Kinara ARA-2 SDK)
- For Python: `maturin`, Python 3.11+

### Native Build

```bash
cargo build --release
```

### Cross-Compile for aarch64 (NXP i.MX)

```bash
# Install cross-compilation toolchain
sudo apt install gcc-aarch64-linux-gnu

# Build
cargo build --release --target aarch64-unknown-linux-gnu
```

The cross-linker is configured in `.cargo/config.toml`.

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `hal` | yes | Enable edgefirst-hal for tensor/image operations |

Build without HAL (FFI-only):
```bash
cargo build --no-default-features
```

### Python Wheel

```bash
cd crates/ara2-py
maturin build --release --features pyo3/abi3-py311
```

## Dependencies

### Runtime (on-target)

- `libaraclient.so.1` — Kinara client library. Must be on `LD_LIBRARY_PATH` or in system lib dirs.
- `ara2-proxy` — System service providing NPU access. Must be running.

### Crate Dependencies

| Dependency | Purpose |
|-----------|---------|
| `edgefirst-hal` | Tensor memory management, image processing (optional, feature `hal`) |
| `libloading` | Dynamic loading of libaraclient.so.1 |
| `ndarray` | N-dimensional array operations for tensor data |
| `serde` / `serde_json` | DVM metadata parsing |
| `zip` | Reading embedded metadata from DVM files |

## Testing

### On-Target Test Requirements

All tests require an NXP i.MX + ARA-2 PCIe system with:

1. `libaraclient.so.1` installed and accessible
2. `ara2-proxy` service running: `systemctl status ara2-proxy`
3. ARA-2 device visible: `lspci | grep -i kinara`
4. Proxy socket available: `ls -la /var/run/ara2.sock`

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
| `session` | Yes | None | Needs ara2-proxy running |
| `endpoint` | Yes | None | Needs ara2-proxy + NPU |
| `model` | Yes | `ARA2_TEST_MODEL` | Needs a compiled .dvm file |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `ARA2_TEST_MODEL` | Path to a `.dvm` model file for model tests |
| `RUST_LOG` | Log level: `debug`, `info`, `warn`, `error` |

## On-Target Debugging

### Verifying Hardware Setup

```bash
# Check PCIe device
lspci | grep -i kinara

# Check proxy service
systemctl status ara2-proxy
journalctl -u ara2-proxy --no-pager -n 50

# Check socket
ls -la /var/run/ara2.sock
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

### Workflows

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `test.yml` | Push/PR to main, develop | Lint (fmt, clippy), build check (default + no-default-features), test (dvm_metadata only) |
| `build.yml` | Push/PR to main, develop | Release build (x86_64, aarch64) |
| `sbom.yml` | Push/PR to main | SBOM generation, license compliance |
| `release.yml` | Tag (v*) | Trusted publish to crates.io (ara2-sys → ara2), GitHub Release with SBOM |

### Test Gap

Hardware-dependent tests (session, endpoint, model) require a self-hosted runner
with ARA-2 hardware. Only `dvm_metadata` tests run in CI.

## Publishing

### crates.io (ara2, ara2-sys)

Releases are published automatically via trusted publishing when a `v*` tag is pushed.
The release workflow publishes `ara2-sys` first, then `ara2`.

For manual publishing (e.g., initial release):

```bash
cargo publish -p ara2-sys
cargo publish -p ara2
```

### Trusted Publishing Setup

1. Manually publish the initial release of each crate
2. On crates.io: each crate → Settings → Trusted Publishers → Add `EdgeFirstAI/ara2-rs`
3. On GitHub: Settings → Environments → Create `crates-io` environment
