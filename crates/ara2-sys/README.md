# ara2-sys

FFI bindings to `libaraclient` for the Kinara ARA-2 neural network accelerator.

This crate provides low-level, unsafe Rust bindings generated from the ARA-2 C
API header (`dvapi.h`), vendored from NXP's rt-sdk-ara2 SDK 2.1.1. It uses
[`libloading`](https://crates.io/crates/libloading) for dynamic library
loading at runtime rather than link-time binding. Run `update.sh` to
regenerate the bindings after refreshing the vendored `dvapi.h` /
`dv_status_codes.h` from a newer SDK drop -- always diff the generated
`src/ffi.rs` for struct size/layout changes, not just new symbols; a struct
gaining or losing a field shifts every later field's offset and silently
breaks any code doing pointer arithmetic over arrays of it.

## Usage

Most users should use the [`ara2`](https://crates.io/crates/ara2) crate instead,
which provides safe, high-level Rust APIs built on top of these bindings.

## Runtime Requirements

- `libaraclient` must be installed on the target system (provided by NXP's
  `imx-nxp-ara2` package, rt-sdk-ara2 SDK 2.1.1+). NXP has renamed the
  shared library in every SDK drop so far, so `ara2::open_library()` tries
  each name in `ara2::LIBRARY_NAMES` (the architecture-suffixed name for the
  target -- `libaraclient_aarch64.so` or `libaraclient_x86_64.so` -- then
  `libaraclient.so` and `libaraclient.so.1`) rather than hardcoding one.
- The proxy service (`rt-sdk-ara2.service` on EdgeFirst Yocto images) must be
  running

## Supported Platforms

- NXP i.MX 8M Plus with ARA-2 PCIe accelerator
- NXP i.MX 95 with ARA-2 PCIe accelerator

## License

Licensed under the Apache License 2.0. See [LICENSE](../../LICENSE) for details.

<img referrerpolicy="no-referrer" src="https://px.edgefirst.ai/a.png?x-pxid=67b03702-12df-456b-86f9-f246395421b5" alt="" width="1" height="1" style="position:absolute; width:1px; height:1px; opacity:0; pointer-events:none;" />
