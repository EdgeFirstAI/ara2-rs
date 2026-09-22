# ara2-sys

FFI bindings to `libaraclient` for the Kinara ARA-2 neural network accelerator.

This crate provides low-level, unsafe Rust bindings generated from the ARA-2 C API header (`dvapi.h`). It uses [`libloading`](https://crates.io/crates/libloading) for dynamic library loading at runtime rather than link-time binding.

## DVAPI generations

`libaraclient` is in circulation in two interface generations, and both are bound here. The version that matters is the DVAPI version the library reports through `dv_get_client_lib_version`, not the version of the SDK it was packaged in:

| DVAPI | Packaging | `dv_model_output_param` |
|-------|-----------|-------------------------|
| 1.1.2 | Kinara ARA-2 runtime, packaged by [meta-kinara](https://github.com/EdgeFirstAI/meta-kinara) as 1.2.1 | 112 bytes |
| 1.3.2 | NXP `imx-nxp-ara2`, rt-sdk-ara2 SDK 2.1.1 | 120 bytes |

The bindings are additive. `dvapi.h` and `dv_status_codes.h` are the vendored 1.1.2 headers and supply the base; `dvapi-1.3.h` declares what 1.3 adds — functions, types, enums, status codes, and the extended structs under `_1_3` names. Building on the older layout is what makes a single binary safe against either library: every 1.3 change is a field appended to the end of a struct, so the 1.1 layout stays valid for reading whatever both generations share.

The function table needs no versioning. `--dynamic-loading` resolves each symbol into its own `Result`, so a 1.1 library leaves the 1.3 entries `Err` and the caller sees that directly.

Two things do not follow from the shared layout and are selected by `ara2::Abi` in the safe crate: the stride of an array the client indexes, and the meaning of endpoint states 4 and 5, which 1.3 reuses for different names.

## Regenerating

Run `update.sh` after refreshing a vendored header. It generates from `dvapi-1.3.h`, which includes `dvapi.h`, so one pass covers both generations.

When a new DVAPI version appears, derive its delta into a new additions header rather than replacing the base. Diff the generated `src/ffi.rs` for struct size and layout changes, not just new symbols: a struct gaining or losing a field shifts every later field's offset and silently breaks any code doing pointer arithmetic over arrays of it. The `assert_appended!` invocations in `src/lib.rs` hold new generations to the append-only shape and turn a reorder or retype into a build error.

## Usage

Most users should use the [`ara2`](https://crates.io/crates/ara2) crate instead,
which provides safe, high-level Rust APIs built on top of these bindings.

## Runtime Requirements

- `libaraclient` must be installed on the target system, implementing DVAPI 1.1 or 1.3. `ara2::open_library()` tries each name in `ara2::LIBRARY_NAMES` — the architecture-suffixed name for the target (`libaraclient_aarch64.so` or `libaraclient_x86_64.so`), then `libaraclient.so` and `libaraclient.so.1` — since the name has differed in every packaging so far and there is no soname to rely on.
- The proxy service (`rt-sdk-ara2.service` on EdgeFirst Yocto images) must be
  running

## Supported Platforms

- NXP i.MX 8M Plus with ARA-2 PCIe accelerator
- NXP i.MX 95 with ARA-2 PCIe accelerator

## License

Licensed under the Apache License 2.0. See [LICENSE](../../LICENSE) for details.

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=67b03702-12df-456b-86f9-f246395421b5" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
