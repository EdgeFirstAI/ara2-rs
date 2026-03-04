# ara2-sys

FFI bindings to `libaraclient` for the Kinara ARA-2 neural network accelerator.

This crate provides low-level, unsafe Rust bindings generated from the ARA-2 C
API header (`dvapi.h`). It uses [`libloading`](https://crates.io/crates/libloading)
for dynamic library loading at runtime rather than link-time binding.

## Usage

Most users should use the [`ara2`](https://crates.io/crates/ara2) crate instead,
which provides safe, high-level Rust APIs built on top of these bindings.

## Runtime Requirements

- `libaraclient.so.1` must be installed on the target system (provided by the
  Kinara ARA-2 SDK)
- The `ara2-proxy` system service must be running

## Supported Platforms

- NXP i.MX 8M Plus with ARA-2 PCIe accelerator
- NXP i.MX 95 with ARA-2 PCIe accelerator

## License

Licensed under the Apache License 2.0. See [LICENSE](../../LICENSE) for details.
