# ara2-sys

FFI bindings to `libaraclient` for the Kinara ARA-2 neural network accelerator.

This crate provides low-level, unsafe Rust bindings to the ARA-2 C API (DVAPI). It uses [`libloading`](https://crates.io/crates/libloading) for dynamic library loading at runtime rather than link-time binding.

## DVAPI versions

`libaraclient` is in circulation in three interface versions, and all are bound here. The version that matters is the DVAPI version the library reports through `dv_get_client_lib_version`, not the version of the SDK it was packaged in:

| DVAPI | Packaging | `dv_model` | `dv_model_output_param` |
|-------|-----------|------------|-------------------------|
| 1.1.2 | Kinara ARA-2 runtime, packaged by [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara) as 1.2.1 | 128 bytes | 112 bytes |
| 1.3.2 | NXP `imx-nxp-ara2`, rt-sdk-ara2 SDK 2.1.1 | 128 bytes | 120 bytes |
| 3.0.0 | Kinara arart r3.0 | 152 bytes, `hw_cycles` inserted at offset 40 | 120 bytes |

3.0 inserts a field in the middle of `dv_model`, so no version's layout is a prefix of another's. Each version-specific struct is therefore declared in full, at that version's layout, in its own module.

## Bindings

The declarations are our own, written from each SDK's headers and covering what `ara2` uses; no SDK header is shipped. They live in `include/`:

| Header | Generated into | Contents |
|--------|----------------|----------|
| `dvapi_common.h` | crate root (`src/ffi.rs`) | Handles, enums, the structs whose layout is identical in every version, the version-specific structs as incomplete (opaque) types, and the functions whose signature is identical in 1.1, 1.3 and 3.0 |
| `dvapi_v1_1.h` | `v1_1` | The version-specific structs at the 1.1 layout, plus the functions whose signature differs between versions (`dv_get_endpoint_busyness`) |
| `dvapi_v1_3.h` | `v1_3` | The same struct names at the 1.3 layout, plus `dv_get_endpoint_busyness`, `dv_client_set_log_level` and the `_s` detailed-status variants |
| `dvapi_v3_0.h` | `v3_0` | The same struct names at the 3.0 layout, plus `dv_get_endpoint_busyness` (which gains an `active_load` parameter), `dv_client_set_log_level` and the `_e` detailed-status variants |

Version-specific structs keep their DVAPI name inside each module (`v1_3::dv_model`, `v3_0::dv_model`). The crate root's `dv_model` is a zero-sized opaque type: pointers to it are passed through the shared function table, and the safe crate casts them to the module its probed version selects. `ara2-sys` exposes only what is declared here.

Each version module has its own dynamic-loading function table, loaded from the same library handle as the shared one. A symbol a library lacks resolves to `Err` in its table rather than failing the load.

Conventions the verifier relies on, for anyone editing a header:

- Every enumerator has an explicit `= value`.
- Every function prototype is on one line.
- Inside `dvapi_v*.h`, a version-specific struct is referenced by `struct tag`, never by its `_t` typedef. The typedefs live in the common header and name the opaque common type, so using one in a version header would silently bind the opaque type.

`src/lib.rs` pins, per version, the size of every version-specific struct and the offset of every field `ara2` reads, so an accidental edit to a declaration is a build error.

## Regenerating and verifying

`update.sh` regenerates `src/ffi.rs` and `src/v*.rs` from `include/` with bindgen, then adapts the output to `libloading` 0.9.

`update.sh --verify <1.1|1.3|3.0> <sdk-include-dir>` checks our declarations for that version against the real SDK headers. It runs `tests/verify_sdk.rs`, which compiles a C translation unit against the SDK's `dvapi.h` asserting every declared size, alignment, field offset and enumerator value, each enum's size and signedness, and every function prototype (a mismatched signature is a "conflicting types" error). It then prints, for information, the SDK functions and structs `ara2-sys` does not declare. The test can also be run directly:

```bash
ARA2_SDK_V1_1=<dir> ARA2_SDK_V1_3=<dir> ARA2_SDK_V3_0=<dir> ARA2_SDK_REQUIRE=all \
  cargo test -p ara2-sys --test verify_sdk
```

A version whose variable is unset is skipped, which is the normal case in CI where no SDK is available; `ARA2_SDK_REQUIRE` (comma-separated versions, or `all`) turns a missing one into a failure. All three versions pass against their SDKs: Kinara runtime 1.2.1 (DVAPI 1.1), NXP rt-sdk-ara2 2.1.1 (DVAPI 1.3) and arart r3.0 (DVAPI 3.0).

When a new SDK drops, run `--verify` against it before anything else. A new DVAPI version gets its own `dvapi_v*.h` and module, and an `ara2::Abi` variant.

## Usage

Most users should use the [`ara2`](https://crates.io/crates/ara2) crate instead,
which provides safe, high-level Rust APIs built on top of these bindings.

## Runtime Requirements

- `libaraclient` must be installed on the target system, implementing DVAPI 1.1, 1.3 or 3.0. `ara2::open_library()` tries each name in `ara2::LIBRARY_NAMES` — the architecture-suffixed name for the target (`libaraclient_aarch64.so` or `libaraclient_x86_64.so`), then `libaraclient.so` and `libaraclient.so.1` — since the name has differed in every packaging so far and there is no soname to rely on.
- The ARA-2 proxy service must be running (`rt-sdk-ara2.service` on EdgeFirst Yocto images with NXP's packaging).

## Supported Platforms

- NXP i.MX 8M Plus with ARA-2 PCIe accelerator
- NXP i.MX 95 with ARA-2 PCIe accelerator

## License

Licensed under the Apache License 2.0. See [LICENSE](../../LICENSE) for details.

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=67b03702-12df-456b-86f9-f246395421b5" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
