# 40. Eigen dependency removed (EigenLM backend dropped)

**Date:** 2026-06-27

## Summary

The library now has **no third-party C++ dependency**. The Eigen-based constant
optimizer (`EigenLMOptimizer`) — the last consumer of Eigen anywhere in the shipped
code — has been removed, and with it every `find_package`/`FetchContent` of Eigen and
the R package's `LinkingTo: RcppEigen`. The default and only least-squares backend is
the self-contained `SelfLMOptimizer` (pure C++/STL, `self_lm_optimizer.cpp`).

## Why

`SelfLMOptimizer` had already replaced `EigenLMOptimizer` as the default (docs/25): it
matched EigenLM's recovery rate problem-for-problem while being ~7–8× faster per fit and
performing zero large per-fit heap allocations, which restored multi-island scaling.
EigenLM was kept only as a comparison backend. Eigen was therefore a build-time
dependency that the shipped runtime never used. Removing it:

- aligns with the project's dependency policy (the default answer to "keep a dependency
  we don't need?" is no) and with "no Julia / minimal dependencies" intent;
- removes a Windows/Rtools build-surface (RcppEigen `LinkingTo`, Eigen header fetch);
- simplifies all three builds (R, standalone, Python) to "needs a C++17 compiler".

The trade-off (accepted by the user): the EigenLM comparison baseline is no longer
reproducible from the current tree. Its measurements remain on record in docs/23 and
docs/25, and the `OptimizerType`/`OptimizerFactory` seam is unchanged, so a backend can
still be re-added later behind the same interface if a comparison is needed again.

## What changed

- **Deleted:** `src/eigen_lm_optimizer.cpp`,
  `src/rsymbolic/optimization/eigen_lm_optimizer.hpp`,
  `standalone/tests/test_eigen_lm_optimizer.cpp`, and the Eigen-specific benchmarks
  `bench_eigen_reuse.cpp`, `bench_alloc.cpp`, `bench_heap.cpp` (their reason for being
  was the EigenLM-vs-SelfLM allocation comparison).
- **Core:** removed `OptimizerType::EigenLM`, its `OptimizerFactory::create` case, and
  the `"eigen_lm"` `from_string` mapping (`optimizer_factory.{hpp,cpp}`).
- **Tests:** `test_constant_fitting`, `test_optimizer_stop`, `test_multivariate`, and the
  `test_random_restart_optimizer` factory test now exercise `SelfLM` instead of EigenLM.
- **Build:** removed Eigen from `CMakeLists.txt`, `standalone/CMakeLists.txt`,
  `python/CMakeLists.txt`; dropped `RcppEigen` from `DESCRIPTION` `LinkingTo` and switched
  `rsymbolic2_r.cpp` / `RcppExports.cpp` from `RcppEigen.h` to `Rcpp.h`.

## Verification (2026-06-27)

- **Windows (Rtools45 MinGW/UCRT):** standalone CMake build + `ctest` 18/18 passed;
  `R CMD INSTALL` clean (no RcppEigen) + testthat **82 PASS / 0 FAIL** (incl. recovery,
  `NOT_CRAN=true`); Python `pip install ./python` + recovery loss=0 + pytest 5 passed.
- **Ubuntu 24.04 (gcc 13.3):** see the build verification run recorded alongside this
  change (standalone + Python rebuilt with no Eigen).

Historical docs (06, 07, 20–25, etc.) still describe EigenLM as it existed when written;
they are left as the record of that period. This file is the authoritative note that the
dependency has since been removed.
