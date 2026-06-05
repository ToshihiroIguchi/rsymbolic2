# Build and test the rsymbolic2 core with Rtools45 (MinGW/GCC/UCRT).
# Run from the repository root:
#   .\scripts\build-mingw.ps1
#
# Optional parameters:
#   -RtoolsRoot  path to Rtools installation (default: C:\rtools45)
#   -BuildDir    build directory              (default: build-mingw)
#   -Config      cmake build type            (default: Release)

param(
    [string]$RtoolsRoot = "C:\rtools45",
    [string]$BuildDir   = "build-mingw",
    [string]$Config     = "Release"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# Locate cmake and ninja from VS18 (already known to work on this machine).
$vsBase = "C:\Program Files\Microsoft Visual Studio\18\Community"
$cmakeExe = "$vsBase\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninjaExe = "$vsBase\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"

if (-not (Test-Path $cmakeExe)) { throw "cmake not found at $cmakeExe" }
if (-not (Test-Path $ninjaExe)) { throw "ninja not found at $ninjaExe" }

# Rtools usr\bin must be in PATH so that the GCC toolchain can find its
# own utilities (as, ld, dlltool, etc.) when cmake test-compiles.
$rtoolsUsr = "$RtoolsRoot\usr\bin"
$rtoolsTc  = "$RtoolsRoot\x86_64-w64-mingw32.static.posix\bin"
$env:PATH  = "$rtoolsUsr;$rtoolsTc;$env:PATH"

$toolchain = Join-Path $repo "cmake\toolchain-rtools.cmake"

Write-Host "=== configure ===" -ForegroundColor Cyan
& $cmakeExe -B $BuildDir -G Ninja `
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain" `
    "-DRTOOLS_ROOT=$RtoolsRoot" `
    "-DCMAKE_BUILD_TYPE=$Config" `
    "-DCMAKE_MAKE_PROGRAM=$ninjaExe" `
    "-DRSYMBOLIC2_BUILD_TESTS=ON" `
    "-DRSYMBOLIC2_BUILD_BENCHMARKS=OFF"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "=== build ===" -ForegroundColor Cyan
& $cmakeExe --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

Write-Host "=== test ===" -ForegroundColor Cyan
& $cmakeExe --build $BuildDir --target test -- ARGS="--output-on-failure -C $Config"
$testExit = $LASTEXITCODE

Write-Host ""
if ($testExit -eq 0) {
    Write-Host "All tests passed (MinGW/GCC/UCRT)" -ForegroundColor Green
} else {
    Write-Host "Some tests FAILED" -ForegroundColor Red
    exit $testExit
}
