#!/bin/sh

bindgen --dynamic-loading araclient --allowlist-item 'dv_.*' --allowlist-item 'DV_.*' dvapi.h > src/ffi.rs
