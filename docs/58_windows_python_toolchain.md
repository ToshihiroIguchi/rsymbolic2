# Windows C++ Toolchain for the Python Package (MSVC vs. Rtools)

**Date:** 2026-07-23
**Status:** measured; documentation corrected, one MSVC-only build flag added
**Scope:** build/documentation only. No engine source change, no default change.

---

## 1. Why

`README.md` listed **Rtools** as *the* Windows prerequisite for both packages, with the
note "The R package *requires* Rtools anyway; the Python package reuses the same
compiler". That is accurate for R and misleading for Python:

- `python/CMakeLists.txt` requires only a C++17 compiler. When no suitable CMake is on
  `PATH`, `scikit-build-core` adds the PyPI `cmake` wheel to the build requirements itself
  (`builder/get_requires.py`), so the user supplies a compiler and nothing else.
- The core (`r-package/rsymbolic2/src/`) contains no compiler-specific constructs — no
  `__GNUC__`, `_MSC_VER`, `__builtin_*`, `#pragma GCC` or `__attribute__`. The only
  OpenMP use is two `#pragma omp parallel for schedule(dynamic) num_threads(...) if(n > 1)`
  loops in `evolutionary_search.cpp` (signed `std::int64_t` index), which is within
  OpenMP 2.0 — the level MSVC implements with `/openmp`.
- **A Windows user without Rtools builds with MSVC, and that path had never been tested.**
  On Windows `scikit-build-core` keeps CMake's own default generator
  (`builder/generator.py: get_default()`), so the *CMake that gets found* decides the
  toolchain:

  | CMake used | Its default generator | Compiler |
  |---|---|---|
  | Rtools' `cmake` (on `PATH`) | `Unix Makefiles` → rewritten to `Ninja` by `get_default()` | GCC found on `PATH` |
  | VS-bundled or PyPI `cmake` | `Visual Studio 18 2026` | MSVC |

  So a machine with Rtools on `PATH` builds with GCC (which is why every CMake cache in
  this repository — `build/`, `build-noeigen/`, `standalone/build-win/` — was configured
  with `C:/rtools45/x86_64-w64-mingw32.static.posix/bin/g++.exe`), while a Python-only
  machine goes through MSVC. The README documented only the first case and the second was
  untested.

## 2. Measured result (2026-07-23, this repository at HEAD)

Both toolchains build the Python package from the same tree and pass the same suite.
Each was built in a clean venv with `pip install ./python`; the MSVC run was made with
Rtools removed from `PATH` so the toolchain could not leak.

| | **MSVC** | **Rtools / MinGW** | **Ubuntu (WSL)** |
|---|---|---|---|
| Compiler | MSVC 14.50.35717 (VS 18 Community) | GCC 14.3.0 (`x86_64-w64-mingw32.static.posix`, UCRT) | GCC (`x86_64-linux-gnu-g++`) |
| CMake generator | Visual Studio 18 2026 | Ninja | Ninja |
| OpenMP | found, **2.0** (`-openmp`) | found, **4.5** (`-fopenmp`) | found, 4.5 (`-fopenmp`) |
| Compile errors | none | none | none |
| `pytest python/tests` | **47 passed, 3 skipped** | **47 passed, 3 skipped** | **47 passed, 3 skipped** |
| Wheel tag | `cp313-cp313-win_amd64` | `cp313-cp313-win_amd64` | — |

(The 3 skips are the optional `pandas` / `matplotlib` extras, absent in the test venvs.)

### 2.1 Runtime DLL dependencies differ — and that matters on Windows

`dumpbin /dependents` on the built `_core.cp313-win_amd64.pyd`:

| Toolchain | Non-system dependencies |
|---|---|
| Rtools / MinGW | **none** — only `python313.dll`, `KERNEL32.dll` and `api-ms-win-crt-*` (UCRT) |
| MSVC | `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, `VCOMP140.DLL` (+ UCRT) |

Rtools45's `static.posix` toolchain links libstdc++, libgcc **and libgomp statically**, so
the MinGW-built extension is self-contained. This is worth knowing because **Python 3.8+
no longer searches `PATH` when resolving an extension module's DLL dependencies**: a
MinGW build that dynamically linked `libstdc++-6.dll` would fail to import once Rtools left
the search path. It does not, because of the static toolchain.

The MSVC build needs the Microsoft Visual C++ runtime (`MSVCP140`/`VCOMP140`). In practice
that is present on any machine with the Build Tools installed — which is exactly the
machine that just compiled it — so the source-install story is fine. It would matter for
*redistributing* wheels (they would need `delvewheel`, or the VC++ Redistributable as a
documented requirement); see §4.

### 2.2 Results are not bit-identical across toolchains

Same seed, same settings, MSVC vs. MinGW, `population_size=50, generations=30, seed=1`:
both converge to a numerically exact fit (loss ≈ 1e-29) but by different expressions, and
shared Pareto members differ in the last ULP (`30.013102441059864` vs
`30.013102441059857`). This is the already-documented libm/ULP situation, identical in kind
to the WASM build (`docs/51`): floating-point differences in transcendental functions
change which candidate wins a comparison and the search diverges from there. Quality
parity holds; bit parity across *toolchains* is not claimed and never was — the
reproducibility guarantee is same-build, same-seed, thread-count independent.

### 2.3 UTF-8 warning (fixed)

MSVC on a non-UTF-8 locale (here CP932) reported **C4819** for every core header: the
sources are UTF-8 without a BOM and MSVC otherwise reads them in the system code page.
Non-ASCII appears only in comments — the two UTF-8 unit aliases (`µ`, `Ω`) are matched on
raw bytes (`unit_parser.hpp`: `0xC2 0xB5`, `0xCE 0xA9`), not through string literals, and
a probe confirmed `X_units=["µm"]` and `X_units=["um"]` give identical results in both
builds. The warnings were therefore cosmetic but appeared on every file, so
`python/CMakeLists.txt` now adds `/utf-8` under `if(MSVC)`. GCC and Clang already assume
UTF-8, so the R, Ubuntu and WASM builds are untouched. After the change: 0 × C4819, tests
still 47 passed / 3 skipped on all three platforms.

## 3. What the documentation now says

- `README.md` prerequisites table: split into **Windows (Python)** — MSVC Build Tools *or*
  Rtools, both verified — and **Windows (R)** — Rtools required, because R on Windows is
  built with MinGW/UCRT and not MSVC (CLAUDE.md, Platform Constraints).
- The Windows tip covers both paths (Developer Command Prompt for MSVC, `PATH` for
  Rtools) and states that CMake picks MSVC when both are available.
- `docs/08_mingw_build.md` no longer claims `build/` is an MSVC configuration; it never
  was.

Recommendation kept in the README: if you use the R package too, install Rtools and let
one toolchain serve both. Otherwise MSVC is the ordinary Windows Python route.

## 4. Known limitation (recorded, not addressed here)

There are **no binary wheels**: every user compiles from source, and the only CI workflow
is `deploy-pages.yml` (web GUI). Publishing to PyPI would also require solving a layout
problem — the shared core lives outside `python/` (`r-package/rsymbolic2/src/`), which
`scikit-build-core` cannot pull into an sdist, so an sdist built from `python/` alone would
not compile. The two supported source installs are unaffected and both work:
`pip install "git+…#subdirectory=python"` (pip clones the whole repository) and
`pip install ./python` from a clone. Adding CI and/or `cibuildwheel` wheels is out of scope
for this change.

## 5. Reproducing

```powershell
# MSVC (Rtools must not be on PATH, or CMake may pick g++)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
python -m venv venv-msvc
venv-msvc\Scripts\pip install pytest numpy
venv-msvc\Scripts\pip install .\python
venv-msvc\Scripts\python -m pytest python\tests -q

# Rtools (PATH contains C:\rtools45\...\bin)
python -m venv venv-rtools
venv-rtools\Scripts\pip install pytest numpy
venv-rtools\Scripts\pip install .\python
venv-rtools\Scripts\python -m pytest python\tests -q
```

```bash
# Ubuntu (WSL)
python3 -m venv ~/venv-rsym
~/venv-rsym/bin/pip install pytest numpy ./python
~/venv-rsym/bin/python -m pytest python/tests -q
```
