#!/bin/sh
# Regenerates the bindings from include/. The common header goes to the
# crate root; each version header goes to its own module, restricted to
# the items that header itself declares.
set -e
cd "$(dirname "$0")"

# libclang does not bundle its freestanding headers (stdbool.h etc.) on
# every system; gcc's are always present alongside it.
gcc_include="$(gcc -print-file-name=include)"

bindgen --dynamic-loading araclient \
    --allowlist-file '.*/dvapi_common\.h' \
    include/dvapi_common.h -o src/ffi.rs -- -I"$gcc_include" -Iinclude

for v in 1_1 1_3 3_0; do
    bindgen --dynamic-loading "araclient_v$v" \
        --allowlist-file ".*/dvapi_v$v\.h" --no-recursive-allowlist \
        --raw-line 'use super::*;' \
        "include/dvapi_v$v.h" -o "src/v$v.rs" -- -I"$gcc_include" -Iinclude
done

# bindgen passes `path` straight to libloading::Library::new, whose 0.9
# bound (AsFilename) a generic AsRef<OsStr> does not satisfy; &OsStr does.
sed -i 's/::libloading::Library::new(path)?/::libloading::Library::new(path.as_ref())?/' \
    src/ffi.rs src/v1_1.rs src/v1_3.rs src/v3_0.rs
