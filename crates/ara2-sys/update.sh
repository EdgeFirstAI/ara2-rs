#!/bin/sh
set -e

# libclang doesn't bundle its own freestanding headers (stdbool.h etc.) on
# every system; point it at gcc's, which are always present alongside it.
gcc_include="$(gcc -print-file-name=include)"

bindgen --dynamic-loading araclient --allowlist-item 'dv_.*' --allowlist-item 'DV_.*' \
    dvapi.h -o src/ffi.rs -- -I"$gcc_include"
