#!/usr/bin/env bash
# Build tools the lane needs that the runner image does not provide.
#
# Two gaps, both discovered by a real run rather than by reading an image
# manifest:
#
#   - `ubuntu-24.04-arm-xlarge` has no C compiler, so every build script in
#     the tree fails with `linker "cc" not found` before a single test runs.
#     The GitHub-hosted `ubuntu-24.04-arm` image does provide one; the larger
#     runner is a different image.
#   - Quick cross-compiles to aarch64 from an x86_64 host. That pass is
#     check-only for Rust, but a dependency with a build script still invokes
#     cc-rs, which needs a compiler for the target triple.
#
# Honours the shared setup-hook contract: SKIP_PACKAGES=1 means do not install
# system packages, for runners with no package manager or no network path to
# one.
set -euo pipefail

[[ "$(uname -s)" == Linux ]] || exit 0

need=()
command -v cc >/dev/null 2>&1 || need+=(build-essential)
if [[ "$(uname -m)" == x86_64 ]] && ! command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
  need+=(gcc-aarch64-linux-gnu)
fi

if [[ ${#need[@]} -eq 0 ]]; then
  echo "ok  build toolchain already present"
  exit 0
fi

if [[ "${SKIP_PACKAGES:-0}" == 1 ]]; then
  echo "::notice::SKIP_PACKAGES=1; not installing ${need[*]}"
  exit 0
fi

echo "::notice::installing ${need[*]} (absent from this runner image)"
sudo apt-get update
sudo apt-get install -y "${need[@]}"
