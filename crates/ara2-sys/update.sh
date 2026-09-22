#!/bin/sh
set -e

# libclang doesn't bundle its own freestanding headers (stdbool.h etc.) on
# every system; point it at gcc's, which are always present alongside it.
gcc_include="$(gcc -print-file-name=include)"

# dvapi-1.3.h includes dvapi.h, so one pass yields the 1.1 base plus the
# 1.3 additions. The function table is additive for free: --dynamic-loading
# resolves each symbol into its own Result, so a 1.1 library simply leaves
# the 1.3 entries Err.
bindgen --dynamic-loading araclient --allowlist-item 'dv_.*' --allowlist-item 'DV_.*' \
    dvapi-1.3.h -o src/ffi.rs -- -I"$gcc_include"
