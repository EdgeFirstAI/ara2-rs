#!/bin/sh
# Regenerates the bindings from include/. The common header goes to the
# crate root; each version header goes to its own module, restricted to
# the items that header itself declares.
set -e
export LC_ALL=C
# Resolve the SDK path before changing directory so it may be relative to
# the caller's working directory.
if [ "$1" = "--verify" ] && [ -d "$3" ]; then
    sdk="$(cd "$3" && pwd)"
fi
cd "$(dirname "$0")"

# libclang does not bundle its freestanding headers (stdbool.h etc.) on
# every system; gcc's are always present alongside it.
gcc_include="$(gcc -print-file-name=include)"

if [ "$1" = "--verify" ]; then
    version="$2"
    case "$version" in
        1.1|1.3|3.0) ;;
        *) echo "usage: $0 --verify <1.1|1.3|3.0> <sdk-include-dir>" >&2; exit 2 ;;
    esac
    [ -n "$sdk" ] || { echo "$0: SDK include directory not found: $3" >&2; exit 2; }
    var="ARA2_SDK_V$(echo "$version" | tr . _)"
    env "$var=$sdk" ARA2_SDK_REQUIRE="$version" cargo test -p ara2-sys --test verify_sdk -- --nocapture

    # Informational: what the SDK has that ara2-sys does not declare.
    out="../../target/dvapi-inventory-$version"
    mkdir -p "$out"
    bindgen --allowlist-item 'dv_.*' --allowlist-item 'DV_.*' "$sdk/dvapi.h" \
        -o "$out/sdk.rs" -- -I"$gcc_include" -I"$sdk"
    v="$(echo "$version" | tr . _)"
    cat src/ffi.rs "src/v$v.rs" > "$out/ours.rs"
    for kind in fn struct; do
        grep -oE "pub $kind dv_[a-z0-9_]+" "$out/sdk.rs" | sort -u > "$out/sdk.$kind"
        if [ "$kind" = fn ]; then
            grep -oE "pub dv_[a-z0-9_]+:" "$out/ours.rs" | sed 's/^pub /pub fn /; s/:$//' | sort -u > "$out/ours.$kind"
        else
            grep -oE "pub struct dv_[a-z0-9_]+" "$out/ours.rs" | sort -u > "$out/ours.$kind"
        fi
        echo "== SDK ${kind}s not declared by ara2-sys:"
        comm -23 "$out/sdk.$kind" "$out/ours.$kind" | sed 's/^pub [a-z]* /  /'
    done
else
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
fi
