# Dependency Analysis: rsymbolic2

**Date:** 2026-06-03

---

## 1. Core C++ Library Dependencies

**Guiding rule (from CLAUDE.md):** the default answer to "add a dependency?" is no.
Portability and simplicity outrank performance. A dependency that raises Windows
maintenance cost is a major penalty. This analysis is revised accordingly: the
earlier draft listed Ceres Solver and Intel TBB as required; both are now demoted
to conditional fallbacks adopted only on measured evidence.

### 1.1 Required (Minimal, dependency-light core)

#### Eigen 3.4+
- **Purpose:** Dense linear algebra for the self-implemented Levenberg-Marquardt
  constant optimizer and for batch evaluation matrix operations
- **Why acceptable:** header-only, no build step, widely packaged on both Windows
  and Ubuntu, stable ABI — minimal Windows maintenance cost
- **License:** MPL2

#### OpenMP (compiler feature, not a library dependency)
- **Purpose:** Island-parallel evolution via `#pragma omp parallel for`
- **Why:** built into GCC, Clang, and MSVC; adds no external dependency and no
  Windows toolchain work. Always degradable to serial (compile without `-fopenmp`).
- **License:** N/A (compiler runtime)

#### PCG Random (pcg-cpp)
- **Purpose:** Thread-safe, high-quality pseudo-random number generation
- **Why over std::mt19937:**
  - Faster than Mersenne Twister for small state operations
  - Supports independent streams per thread (seeded by master + thread_id)
  - Header-only, single file — negligible maintenance cost
- **License:** Apache 2.0 / MIT
- **Note:** `std::mt19937_64` from the standard library is an acceptable
  zero-dependency fallback if pcg-cpp ever becomes a burden.

### 1.2 Conditional fallbacks (adopt only on measured evidence)

These are NOT adopted up front. Each is justified only if the dependency-light
default is *measured* to be insufficient. Record the measurement before adopting.

#### Ceres Solver 2.x — fallback for constant optimization
- **Adopt only if:** the self-implemented LM solver shows poor convergence or
  robustness on degenerate (rank-deficient) Jacobians that cannot be fixed simply.
- **Strengths:** NIST nonlinear-regression benchmark winner; built-in AD; robust
- **Cost:** pulls in Eigen + glog + gflags; raises Windows build/maintenance cost
- **License:** BSD 3-Clause

#### Intel TBB (oneTBB) 2021+ — fallback for parallelism
- **Adopt only if:** OpenMP is measured to scale below the 0.7 efficiency target
  due to island load imbalance (considered unlikely for a coarse-grained, roughly
  uniform island model).
- **Strengths:** work-stealing for uneven loads; nested parallelism
- **Cost:** external dependency; additional Windows build configuration
- **License:** Apache 2.0

### 1.3 Optional (Phase-Gated)

#### NLopt 2.7+
- **Purpose:** Alternative optimizer for non-sum-of-squares loss functions (MAE, Huber, custom)
- **Phase:** Phase 4 (custom loss support). Whether NLopt earns its dependency cost
  is itself deferred until custom losses are actually built; a derivative-free
  method on top of the existing solver may suffice.
- **Algorithms used:** L-BFGS-B (gradient-based), Nelder-Mead (derivative-free fallback)
- **License:** MIT/LGPL (algorithm-dependent)
- **Build:** CMake, or `nlopt-dev` system package

#### egglog (via C FFI)
- **Purpose:** E-graph-based symbolic simplification
- **Phase:** Phase 4
- **Language:** Rust; exposed via C ABI (`extern "C"`)
- **Build complexity:** HIGH — requires Rust toolchain, `cbindgen` for header generation
- **License:** MIT
- **Alternative if egglog is too complex:** Custom port of key e-graph rules in pure C++

#### Adept-2
- **Purpose:** Tape-based reverse-mode AD for expressions with many constants (k > 10)
- **Phase:** Phase 4
- **License:** Apache 2.0
- **When needed:** Rarely — most SR expressions have ≤5 constants

---

## 2. R Package Dependencies

### 2.1 Build-Time (C++ Compilation)

| Package    | Version  | Purpose                                 | License |
|------------|----------|-----------------------------------------|---------|
| cpp11      | ≥0.4.0   | R/C++ bridge; wraps C++ types for R     | MIT     |
| BH         | ≥1.81    | Boost headers (used by Ceres indirectly)| BSL-1.0 |

**Note:** The dependency-light default (Eigen header-only + OpenMP + pcg-cpp) keeps
the R build simple on both platforms. Heavy fallbacks (Ceres, TBB) are only a build
concern if measurement forces their adoption; deferring them is itself a
portability decision.

### 2.2 Runtime R Dependencies

| Package   | Version  | Purpose                                    |
|-----------|----------|--------------------------------------------|
| R6        | ≥2.5.0   | Reference classes for mutable search state |
| cli       | ≥3.0     | Progress bar and formatted output          |
| ggplot2   | ≥3.4     | Pareto front visualization (optional)      |
| tibble    | ≥3.0     | Structured results table                   |

### 2.3 Suggests (Testing and Benchmarking)

| Package   | Purpose                              |
|-----------|--------------------------------------|
| testthat  | Unit tests (≥3rd edition)            |
| bench     | Micro-benchmarks vs. PySR            |
| reticulate| Compare against PySR in R            |

---

## 3. Build System

### 3.1 C++ Core: CMake 3.20+

```cmake
cmake_minimum_required(VERSION 3.20)
project(rsymbolic2_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch dependencies (default core: header-only + OpenMP)
include(FetchContent)

FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)
FetchContent_Declare(
    pcg-cpp
    GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
    GIT_TAG        master
)
find_package(OpenMP)  # optional; serial build if absent

add_library(rsymbolic2_core STATIC src/...)
target_link_libraries(rsymbolic2_core PUBLIC Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rsymbolic2_core PUBLIC OpenMP::OpenMP_CXX)
endif()
# Ceres / TBB are NOT linked by default. They appear only behind a CMake option
# (e.g. -DRSYMBOLIC2_USE_CERES=ON) enabled after measurement justifies them.
```

### 3.2 R Package: R CMD build with configure.ac

With the dependency-light default, the R package compiles header-only Eigen +
pcg-cpp and uses the compiler's OpenMP. No external system library is strictly
required, which is the simplest cross-platform path and the most CRAN-friendly.
A `configure` script (or `Makevars.win` / `Makevars`) handles OpenMP flag
detection per platform.

**If a heavy fallback (Ceres/TBB) is later adopted**, it complicates distribution:

| Option | Description | CRAN Compatible |
|--------|-------------|-----------------|
| Bundle source | Include fallback source in `src/` | Yes (slow build) |
| System library | Require user to install via apt/brew | No |
| r-universe only | Distribute via r-universe, not CRAN | Yes |

**Recommendation for Phase 3:** Ship the dependency-light core. This keeps both
Windows and Ubuntu builds simple and defers the CRAN-distribution complications
that heavy fallbacks would introduce.

### 3.3 Python Bindings (Phase 5): pybind11

```cmake
find_package(pybind11 REQUIRED)
pybind11_add_module(rsymbolic2_python python/bindings.cpp)
target_link_libraries(rsymbolic2_python PRIVATE rsymbolic2_core)
```

---

## 4. System Requirements Summary

### Minimum (Phase 1–3)
- C++17 compiler with OpenMP: GCC 9+, Clang 10+, MSVC 2019+
- CMake 3.20+
- Eigen 3.4+ (header-only)
- pcg-cpp (header-only)
- R 4.1+ (for R package)
- cpp11 0.4+ (for R package)

(Ceres Solver and Intel TBB are explicitly NOT in the minimum set; they are
conditional fallbacks behind CMake options, adopted only on measured evidence.)

### Extended (Phase 4–5)
- Rust toolchain 1.70+ (for egglog integration)
- Python 3.10+ with pybind11
- Adept-2 2.1+ (for high-constant-count AD)

---

## 5. Dependency Risk Assessment

| Dependency   | Status      | Risk Level | Mitigation / Notes                              |
|--------------|-------------|------------|-------------------------------------------------|
| Eigen        | Required    | LOW        | Widely used; headers only; stable ABI           |
| OpenMP       | Required    | LOW        | Compiler built-in; degradable to serial         |
| pcg-cpp      | Required    | LOW        | Header-only; fallback std::mt19937_64           |
| cpp11        | Required    | LOW        | RStudio/Posit-maintained                        |
| Ceres Solver | Fallback    | MEDIUM     | Adopt only on measured need; Windows build cost |
| Intel TBB    | Fallback    | MEDIUM     | Adopt only if OpenMP scaling measured poor      |
| NLopt        | Optional    | MEDIUM     | Less active; fallback: derivative-free on own LM|
| egglog       | Optional    | HIGH       | Rust FFI complexity; API stability uncertain    |
| Adept-2      | Optional    | MEDIUM     | Last release 2020; fallback: forward-mode dual  |
| pybind11     | Optional    | LOW        | De facto standard; stable                       |

**Note:** The egglog dependency is the highest risk item due to the Rust FFI requirement. An alternative is to implement a minimal C++ e-graph from scratch (union-find + rule engine), which is feasible at ~2000 lines of C++ but requires careful implementation.

**Windows + R operational risk:** The "Fallback" rating for Ceres and TBB above is
backed by primary-source evidence in `06_windows_dependency_risk.md`. The key facts:
R on Windows builds with Rtools (MinGW/GCC + UCRT), not MSVC; Ceres officially
targets only MSVC on Windows and now requires the ABI-unstable Abseil library, with
no CRAN distribution precedent; TBB is shippable on CRAN/Windows but only via the
sustained maintenance embodied in RcppParallel; OpenMP is R's official,
dependency-free parallelism mechanism (`SHLIB_OPENMP_CXXFLAGS`) and works on Windows
via Rtools' libgomp. These are maintainability/portability findings, not performance
claims.
