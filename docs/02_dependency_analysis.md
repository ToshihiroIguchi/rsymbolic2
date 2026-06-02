# Dependency Analysis: rsymbolic2

**Date:** 2026-06-03

---

## 1. Core C++ Library Dependencies

### 1.1 Required (No Alternative)

#### Ceres Solver 2.x
- **Purpose:** Nonlinear least-squares optimization for constant fitting
- **Algorithm used:** Levenberg-Marquardt (DENSE_QR or DENSE_NORMAL_CHOLESKY for small parameter counts)
- **Why Ceres over alternatives:**
  - Benchmarked as best solver on NIST nonlinear regression problems
  - Built-in automatic differentiation (AutoDiff cost functions) as backup to our dual-number approach
  - Handles rank-deficient Jacobians robustly (common when tree has redundant constants)
  - Production-grade: maintained by Google, used in Google Maps, ARCore
- **Dependencies of Ceres:** Eigen 3.3+, glog (optional), gflags (optional)
- **License:** BSD 3-Clause
- **Build:** CMake `FetchContent` or system package

#### Eigen 3.4+
- **Purpose:** Linear algebra primitives used by Ceres; also used for batch evaluation matrix operations
- **License:** MPL2
- **Header-only:** Yes (no separate build step)
- **Note:** Already required by Ceres; no additional cost

#### Intel TBB (oneTBB) 2021+
- **Purpose:** Work-stealing thread pool for island-parallel evolution
- **Why TBB over OpenMP:**
  - Dynamic load balancing handles uneven tree evaluation times
  - Nested parallelism without deadlock risk
  - `parallel_for` with `blocked_range` maps naturally to island populations
  - OpenMP has no work-stealing; thread imbalance degrades scaling efficiency
- **License:** Apache 2.0
- **Build:** CMake `find_package(TBB)` or `FetchContent`

#### PCG Random (pcg-cpp)
- **Purpose:** Thread-safe, high-quality pseudo-random number generation
- **Why over std::mt19937:**
  - Faster than Mersenne Twister for small state operations
  - Supports independent streams per thread (seeded by master + thread_id)
  - Header-only, single file
- **License:** Apache 2.0 / MIT

### 1.2 Optional (Phase-Gated)

#### NLopt 2.7+
- **Purpose:** Alternative optimizer for non-sum-of-squares loss functions (MAE, Huber, custom)
- **Phase:** Phase 3 (custom loss support)
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

**Note:** Ceres and TBB must be available as system libraries or bundled as source. Bundling source is recommended for CRAN compatibility, but increases build time significantly.

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

# Fetch dependencies
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
# Ceres and TBB: prefer system packages, fallback to FetchContent
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)

add_library(rsymbolic2_core STATIC src/...)
target_link_libraries(rsymbolic2_core PUBLIC Ceres::ceres TBB::tbb)
```

### 3.2 R Package: R CMD build with configure.ac

For cross-platform compilation of C++ with external dependencies in an R package, a `configure` script (or `Makevars.win` / `Makevars`) is needed. Key challenge: Ceres and TBB are not available via CRAN.

**Strategy options:**

| Option | Description | CRAN Compatible |
|--------|-------------|-----------------|
| Bundle source | Include Ceres+TBB source in `src/` | Yes (slow build) |
| System library | Require user to install via apt/brew | No |
| CMake + wrapper | Build C++ via CMake, wrap as shared lib | Partially |
| r-universe only | Distribute via r-universe, not CRAN | Yes |

**Recommendation for Phase 3:** Distribute via r-universe with system library requirement (Ceres, TBB). Provide install instructions for Ubuntu/macOS/Windows. CRAN submission is a Phase 5 goal requiring full source bundling.

### 3.3 Python Bindings (Phase 5): pybind11

```cmake
find_package(pybind11 REQUIRED)
pybind11_add_module(rsymbolic2_python python/bindings.cpp)
target_link_libraries(rsymbolic2_python PRIVATE rsymbolic2_core)
```

---

## 4. System Requirements Summary

### Minimum (Phase 1–3)
- C++17 compiler: GCC 9+, Clang 10+, MSVC 2019+
- CMake 3.20+
- Eigen 3.4+
- Ceres Solver 2.1+
- Intel TBB 2021.3+
- R 4.1+ (for R package)
- cpp11 0.4+ (for R package)

### Extended (Phase 4–5)
- Rust toolchain 1.70+ (for egglog integration)
- Python 3.10+ with pybind11
- Adept-2 2.1+ (for high-constant-count AD)

---

## 5. Dependency Risk Assessment

| Dependency   | Maintenance | Risk Level | Mitigation                                      |
|--------------|-------------|------------|-------------------------------------------------|
| Eigen        | Active      | LOW        | Widely used; headers only; stable ABI           |
| Ceres Solver | Active      | LOW        | Google-maintained; stable API; fallback to NLopt|
| Intel TBB    | Active      | LOW        | Open-source since 2020; std::execution backend  |
| cpp11        | Active      | LOW        | RStudio/Posit-maintained                        |
| NLopt        | Moderate    | MEDIUM     | Less active; fallback: custom gradient descent  |
| egglog       | Active      | HIGH       | Rust FFI complexity; API stability uncertain    |
| Adept-2      | Low         | MEDIUM     | Last release 2020; fallback: forward-mode dual  |
| pybind11     | Active      | LOW        | De facto standard; stable                       |

**Note:** The egglog dependency is the highest risk item due to the Rust FFI requirement. An alternative is to implement a minimal C++ e-graph from scratch (union-find + rule engine), which is feasible at ~2000 lines of C++ but requires careful implementation.
