#!/usr/bin/env bash
# Build the rsymbolic2 WebAssembly module (Ubuntu / CI).
#
# Requires an activated emsdk on PATH (emcmake / em++). Typically:
#   source /path/to/emsdk/emsdk_env.sh
#   web/wasm/build.sh
#
# Output: web/app/vendor/rsymbolic2.js + rsymbolic2.wasm
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${here}/build"

emcmake cmake -S "${here}" -B "${build}" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build}"
echo "OK: web/app/vendor/rsymbolic2.js + rsymbolic2.wasm"
