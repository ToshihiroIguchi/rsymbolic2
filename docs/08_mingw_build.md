# MinGW Build (Rtools45 / GCC / UCRT)

## Purpose

The C++ core must build and pass all tests under the same toolchain that R uses
on Windows: Rtools (`x86_64-w64-mingw32.static.posix`, GCC, UCRT target).
This document records the verified procedure and results.

## Verified configuration

| Item | Value |
|---|---|
| Rtools version | Rtools45 |
| Compiler | GCC 14.3.0 (`x86_64-w64-mingw32.static.posix`) |
| Target ABI | UCRT (same as R 4.2+ on Windows) |
| R version | R 4.6.0 |
| CMake | VS18 bundled (cmake 4.1) |
| Ninja | VS18 bundled |
| Host OS | Windows 11 |
| Tests | 12 / 12 passed, 0 warnings |

## Quick start

From the repository root, run the helper script:

```powershell
.\scripts\build-mingw.ps1
```

Or configure/build/test manually:

```powershell
$vsBase  = "C:\Program Files\Microsoft Visual Studio\18\Community"
$cmake   = "$vsBase\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja   = "$vsBase\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$rtools  = "C:\rtools45"
$env:PATH = "$rtools\usr\bin;$rtools\x86_64-w64-mingw32.static.posix\bin;$env:PATH"

& $cmake -B build-mingw -G Ninja `
    -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-rtools.cmake `
    -DRTOOLS_ROOT=$rtools `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_MAKE_PROGRAM=$ninja `
    -DRSYMBOLIC2_BUILD_TESTS=ON `
    -DRSYMBOLIC2_BUILD_BENCHMARKS=OFF

& $cmake --build build-mingw
Set-Location build-mingw
& $cmake -E env CTEST_OUTPUT_ON_FAILURE=1 -- $cmake --build . --target test
```

## Notes

- `build-mingw/` is in `.gitignore`; it is a separate directory from the other
  local `build*/` trees so several configurations can coexist without
  interference. (All of them are in fact Rtools/MinGW configurations — the
  standalone harness has never been built with MSVC. MSVC *is* verified for the
  Python package; see `docs/58_windows_python_toolchain.md`.)
- The toolchain file (`cmake/toolchain-rtools.cmake`) accepts a `RTOOLS_ROOT`
  cache variable for non-default installation paths.
- The Rtools `usr\bin` directory must appear in PATH before configure so that
  GCC can find its own linker utilities (`as`, `ld`, `dlltool`, etc.).
- Ubuntu / GCC verification is deferred until a WSL distribution is installed
  (currently no distro is available on this machine).
