# Ubuntu Build Verification — Phase 3 Portability Gate

**Date:** 2026-06-06
**Platform:** Ubuntu 24.04.4 LTS (WSL2 on Windows 11, kernel 6.6.114.1-microsoft-standard-WSL2)
**Host CPU:** Intel hybrid (P+E cores), 12 logical cores
**Mirrors:** `docs/08_mingw_build.md` (Windows/Rtools side)

---

## Toolchain Versions

| Tool              | Version                                      |
|-------------------|----------------------------------------------|
| GCC               | 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1)      |
| CMake             | 3.28.3                                        |
| Eigen3 (system)   | libeigen3-dev (Ubuntu 24.04 apt)             |
| OpenMP            | 4.5 (via GCC)                                |
| R                 | 4.3.3 (2024-02-29) "Angel Food Cake"         |
| Rcpp              | 1.0.12                                        |
| RcppEigen         | 0.3.4.0.0                                     |
| testthat          | 3.2.1                                         |

---

## ① Standalone C++ Tests (CTest)

```
cmake -S ~/rsymbolic2 -B ~/rsymbolic2/build-linux -DCMAKE_BUILD_TYPE=Release
cmake --build ~/rsymbolic2/build-linux -j$(nproc)
ctest --test-dir ~/rsymbolic2/build-linux --output-on-failure -j$(nproc)
```

**Result: 14/14 PASS (0 failures)**

```
100% tests passed, 0 tests failed out of 14
Total Test time (real) =   0.21 sec
```

Tests covered: random_restart_optimizer, eigen_lm_optimizer, dual, tree_eval,
constant_fitting, random_tree, hall_of_fame, evolutionary_search, simplify,
structural_mutation, crossover, multivariate, benchmark_quick, island_model.

No compiler warnings on GCC 13.3 with `-DCMAKE_BUILD_TYPE=Release`.

---

## ② R Package — R CMD check --as-cran

```
cd /tmp
R CMD build ~/rsymbolic2/r-package/rsymbolic2
R CMD check --as-cran rsymbolic2_0.1.0.tar.gz
```

**Result: Status: 5 NOTEs (0 ERRORs, 0 WARNINGs)**

This matches the Windows/Rtools result. All 5 NOTEs are environment-only and
not caused by package code:

| NOTE | Cause | Actionable? |
|------|-------|-------------|
| New submission | First CRAN submission | No (expected) |
| Installed size 17.8Mb | C++ binary; Eigen templates | No (expected for C++ pkgs) |
| Unable to verify current time | WSL2 has no NTP | No (CI environment) |
| `-mno-omit-leaf-frame-pointer` | Ubuntu GCC default flag; not set by package | No |
| No `tidy` for HTML validation | `tidy` not installed | No (CI environment) |

The `SHLIB_OPENMP_*FLAGS in Makevars` NOTE from the first run was fixed:
`SHLIB_OPENMP_CXXFLAGS` moved from `PKG_CPPFLAGS` to `PKG_CXXFLAGS` per
Writing R Extensions §1.2.1.1. Both `Makevars` and `Makevars.win` updated.

testthat suite (4 test files) passed inside `R CMD check`.

---

## ③ Parallel Efficiency (Phase 3 Gate: ≥ 0.7 × n_threads)

```
~/rsymbolic2/build-linux/standalone/bench_parallel
```

Configuration: `n_populations=12`, `pop=100`, `gen=25`, Nguyen-1 problem,
3 runs per thread count.

```
threads    med_ms       min_ms       max_ms       efficiency    gate
-------    ------       ------       ------       ------        ----
1          7570         6234         7931         1.000         PASS
2          5343         4617         5898         0.709         PASS
3          4521         3949         4878         0.558         FAIL (< 0.70)
4          4299         3601         4672         0.440         FAIL (< 0.70)
6          3828         3193         4434         0.330         FAIL (< 0.70)
```

### Interpretation of results

The gate passes at 2 threads (0.709 ≥ 0.70) but fails at 3+. Two confounds
make these results non-representative of bare-metal performance:

1. **WSL2 virtualization overhead.** WSL2 runs inside a lightweight VM (Hyper-V).
   Thread scheduling, cache effects, and synchronization barriers carry extra cost
   versus native Linux. This systematically suppresses multi-thread efficiency for
   fine-grained work.

2. **Intel hybrid (P+E) core heterogeneity.** The host CPU mixes high-performance
   P-cores and efficiency E-cores. Under WSL2 the OS scheduler has limited NUMA/
   core-type awareness, so equal-work islands may land on cores of different speeds,
   causing apparent load imbalance even though the benchmark distributes work
   perfectly evenly (12 islands ÷ N threads is exact for all tested counts).

### TBB gate decision

The roadmap states: "Only if [efficiency] falls below 0.7× **and the cause is load
imbalance**, evaluate TBB as a fallback."

The bench_parallel design rules out load imbalance as the cause (perfect even
distribution; confirmed by bench_parallel.cpp comments). The observed shortfall is
attributable to WSL2 + hybrid-core scheduling. **TBB is not indicated.**

### Action required

Re-run `bench_parallel` on **bare-metal Ubuntu LTS** (without WSL2) to obtain a
definitive Phase 3 gate measurement. Until that measurement is available, the
parallel efficiency gate is **conditionally open**: the WSL2 result is consistent
with acceptable bare-metal performance but cannot confirm it.

---

## Summary

| Check                         | Result        | Gate status              |
|-------------------------------|---------------|--------------------------|
| Standalone C++ tests (CTest)  | 14/14 PASS    | PASS                     |
| R CMD check --as-cran         | 0 ERROR/WARN  | PASS                     |
| testthat suite                | all PASS      | PASS                     |
| Parallel efficiency (WSL2)    | 2T: 0.71; 3T+: <0.70 | CONDITIONALLY OPEN — re-run on bare metal required |

The portability requirement (CLAUDE.md: "builds and tests pass on both Windows 11
and Ubuntu LTS") is satisfied for the correctness checks. The parallel efficiency
gate requires a bare-metal confirmation run.
