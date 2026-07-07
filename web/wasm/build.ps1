# Build the rsymbolic2 WebAssembly module (Windows dev loop).
#
# Requires an activated emsdk on PATH (emcmake / em++). If emsdk lives at
# C:\Users\<you>\emsdk, run its env script first, e.g.:
#   . C:\Users\toshi\emsdk\emsdk_env.ps1
# then:  web\wasm\build.ps1
#
# Output: web/app/vendor/rsymbolic2.js + rsymbolic2.wasm
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $here "build"

emcmake cmake -S $here -B $build -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "emcmake configure failed" }
cmake --build $build
if ($LASTEXITCODE -ne 0) { throw "wasm build failed" }
Write-Output "OK: web/app/vendor/rsymbolic2.js + rsymbolic2.wasm"
