# 68. Closing the Windows libm gap: it was never the compiler

**Date:** 2026-07-30 (§6.1, the PySR head-to-head follow-up, added 2026-08-01)
**Status:** implemented and verified on both platforms. Supersedes `docs/67` §6's decision
("C — do nothing") and corrects two of its statements (§9).
**Change:** `rsymbolic/platform/libm.hpp` + `platform_libm.cpp`; call sites in `tree.hpp`,
`dual.hpp`, `multi_dual.hpp`, `soa_eval.hpp`. **Windows/MinGW only** — every other
toolchain compiles byte-identical code to before, proved in §7.

## 1. The question, and why the answer is "not the compiler"

The question asked was **what to do about the compiler**, given `docs/67`'s finding that
Windows pays a ~2.5x end-to-end penalty concentrated in `exp`/`log`/`sin`/`pow`.

`docs/67` §3.4 had already ruled the compiler out and nobody noticed the implication. Its
control was that a *pure-arithmetic* tree costs 1344 ns on Windows and 1348 ns on Linux —
identical to within noise, on a MinGW GCC that is in fact the *newer* of the two compilers
(14.3.0 vs 13.3.0). Codegen was never the variable. The variable is the C library.

So the compiler answer is: **change nothing**. R on Windows must be built with Rtools/MinGW
(CLAUDE.md, Platform Constraints), Linux's glibc is already at the frontier, and WASM is on
the fast side (`docs/67` §3.5). What needed changing was *which libm the MinGW build calls*.

## 2. The measurement that reopened the decision

`docs/67` closed with C (do nothing) because the only remaining candidate, A, required
vendoring a third-party transcendental library — an unverified licence, a build-integration
cost, and the Dependency Policy's standing "no". A third Windows arm was never measured:
**MSVC**, which `docs/58` had already established as a supported Python-on-Windows
toolchain.

`bench_libm`, same machine, same source, interleaved, **medians of 5** (net ns/elem):

| op | MinGW (Rtools45) | MSVC 19.50 | MinGW/MSVC | glibc (`docs/67` §3.1) |
|---|---:|---:|---:|---:|
| `exp` | 32.56 | 3.59 | **9.1x** | 2.64 |
| `log` | 18.66 | 3.28 | **5.7x** | 2.51 |
| `sin` | 32.02 | 4.06 | **7.9x** | 4.91 ← MSVC beats glibc |
| `pow` | 62.13 | 8.73 | **7.1x** | 7.77 |
| `sqrt` | 1.22 | 0.62 | 2.0x | 1.11 |

The control that makes this attributable is the one `docs/67` §3.2 used across platforms,
applied here *within* one platform: our own hand-written replacements cost the same under
both compilers (`exp deg10 guarded` 4.47 vs 4.48; `log s17 guarded` 4.09 vs 4.63), and MSVC
is actually **slower** on the cheapest of them (`exp deg6` 1.57 vs 2.94). MSVC is not the
better compiler here. Its libm is simply a different, faster implementation — UCRT's.

The production evaluator agreed (`bench_soa_eval`, batch, medians of 5, ns/1000-point pass):
MinGW 76,702 / MSVC 11,687 on `trig`; MinGW 106,369 / MSVC 16,829 on `transc`. MSVC lands at
Linux's level (11,794 / 12,056) and is *faster* than glibc on `rel_mass` (2,264 vs 3,962).

**Windows is not slow. mingw-w64's `libmingwex` is slow, and the UCRT sitting next to it on
the same machine is glibc-class.**

## 3. The mechanism: UCRT is reachable from a MinGW build

mingw-w64 links the UCRT already (`docs/58` §2.1: the built extension's only non-system
dependencies are `api-ms-win-crt-*`). It nevertheless overrides these six functions with its
own implementations. UCRT's are exported and can be called directly. Measured from a MinGW
build, **through a function pointer, with the indirect call included in the timing**
(medians of 5):

| op | `std::` (libmingwex) | UCRT via `GetProcAddress` | speedup |
|---|---:|---:|---:|
| `exp` | 28.80 | 3.10 | **9.28x** |
| `log` | 16.82 | 3.13 | **5.38x** |
| `sin` | 28.70 | 3.96 | **7.25x** |
| `pow` | 53.99 | 9.63 | **5.61x** |

This lands on MSVC's numbers, which is the confirmation that MSVC's advantage *is* UCRT's
libm and nothing else. **No new dependency**: UCRT is an OS component that every R-on-Windows
process has already loaded.

### 3.1 The simpler binding was measured and rejected

ucrtbase also exports `_o_`-prefixed entry points, which link directly with no runtime
lookup — a far simpler implementation (`extern "C" double _o_exp(double);` and done). It was
tried first and **measured 2.2-3.8x slower** than the apiset names, while returning
bit-identical results (0 disagreements in 10^6 samples per op): `exp` 10.31 vs 3.04 ns, `pow`
14.03 vs 8.93. The `_o_` entries reach UCRT's legacy implementations. The simpler binding
loses on evidence, so the runtime lookup stays — and this paragraph exists so nobody
"simplifies" it back.

## 4. What is redirected, and what deliberately is not

All nine libm functions the evaluator calls were priced before any were touched:

| op | mingw ns | UCRT ns | speedup | results differ? | redirected |
|---|---:|---:|---:|---|---|
| `exp` | 28.6 | 3.3 | 7.3-9.4x | 0.48 % of samples, ≤1 ulp | **yes** |
| `log` | 16.3 | 3.3 | 4.5-5.3x | 0.03 %, ≤1 ulp | **yes** |
| `sin` | 28.5 | 3.7 | 7.6-7.9x | 2.98 %, ≤1.7e-15 | **yes** |
| `cos` | 28.4 | 4.0 | 6.6-7.7x | 3.10 %, ≤2.2e-14 | **yes** |
| `erf` | 55.0 | 6.9 | 7.9-8.2x | 15.9 %, ≤4.6e-16 | **yes** |
| `pow` | 54.0 | 9.6 | 5.6x | 0.44 %, ≤1 ulp | **yes** |
| `tanh` | 5.9 | 5.6 | 1.04-1.09x | **0.000 %** | no |
| `sinh` | 4.9 | 4.4 | 1.07-1.17x | **0.000 %** | no |
| `cosh` | 4.2 | 3.9 | 1.06-1.14x | **0.000 %** | no |
| `sqrt` | 1.2 | 1.3 | 0.79-0.96x (slower) | **0.000 %** | no |

The bottom four are the tell that closes the story: they are **already bit-identical and
already fast**, because mingw-w64 forwards them to the UCRT rather than implementing them.
The six it implements itself are exactly the six that are 4.5-9.4x slow. Redirecting the
bottom four would buy nothing and spend a bit-identity break to do it, so they stay on
`std::`.

The libm calls in the *search mechanics* (annealing in `evolutionary_search.cpp`, the score
in `hall_of_fame.cpp`, the perturbation factor in `mutation.cpp`) also stay on `std::`: they
run once per event rather than once per data point.

## 5. Implementation

`rsymbolic/platform/libm.hpp` declares six functions in `rsymbolic::libm`. Off MinGW each is
a one-line forward to `std::` and the header compiles away. On MinGW they read a table of
function pointers.

- **The table lives in `platform_libm.cpp`**, so `<windows.h>` never enters the evaluator
  headers (which the cpp11/R translation unit includes).
- **Constant-initialised to the `std::` implementations**, then overwritten by a dynamic
  initialiser. Static initialisation always precedes dynamic initialisation, so there is no
  ordering hazard: a caller that somehow ran first would get today's behaviour, never a null
  call. This is also the named fallback the Dependency Policy asks for.
- **`GetModuleHandle` only, never `LoadLibrary`.** A DLL's dynamic initialisers run under the
  Windows loader lock, where `LoadLibrary` can deadlock.
- **All-or-nothing resolution.** A partially resolved table would make the build's numerics
  depend on which symbols happened to be present.
- **`-DRSYMBOLIC2_NO_UCRT_LIBM`** reverts everything to `std::`. Every A/B number below was
  produced with it.

Under CLAUDE.md's PySR-parity rule this is an *implementation-method* change of the kind the
rule explicitly permits: it changes how a transcendental is computed, never which settings
define the search.

## 6. Measured effect

**Evaluator, identical production code, both arms `-O3 -DNDEBUG`, 12 reps** (ns/1000-point
pass, median [min-max]):

| tree | redirect OFF | redirect ON | speedup |
|---|---:|---:|---:|
| `trig` (sin*cos) | 68,992 [58,555-127,695] | 12,420 [10,225-12,873] | **5.55x** |
| `transc` (exp+log+sin) | 90,927 [76,946-260,231] | 14,276 [11,730-16,454] | **6.37x** |
| `rel_mass` (sqrt+div) | 3,352 | 3,399 | 0.99x |
| `poly` (pure arith) | 1,759 | 1,710 | 1.03x |

The bottom two rows are the control: trees containing none of the redirected six are
unchanged, so the redirect costs nothing where it does not apply.

**End to end** (`bench_profile`, PySR-default gate config, n=1000, 2800 generations,
`OMP_NUM_THREADS=4`, both arms Release, run adjacent per seed). `cpu/wall` was 3.45-3.78 in
every cell, so no run was CPU-starved:

| problem | seed | OFF wall | ON wall | wall | cpu | forward evals OFF/ON |
|---|---:|---:|---:|---:|---:|---|
| spring_pe | 1 | 18.99 s | 2.78 s | 6.83x | 6.50x | 248,557 / 199,043 |
| spring_pe | 2 | 4.84 s | 1.09 s | 4.44x | 4.63x | 75,152 / 75,148 ← **same work** |
| spring_pe | 3 | 12.17 s | 2.78 s | 4.38x | 4.30x | 149,032 / 199,046 (ON did *more*) |
| rel_mass | 1 | 124.91 s | 39.76 s | 3.14x | 2.94x | within 0.07 % |
| rel_mass | 2 | 73.52 s | 32.40 s | 2.27x | 2.18x | within 0.03 % |
| rel_mass | 3 | 130.48 s | 28.09 s | 4.65x | 4.69x | within 0.06 % |

**Median 4.41x wall, 4.47x cpu.** The trajectories diverge (§7), so the load-bearing cell is
spring_pe seed 2, where the two arms performed forward evaluation counts within 0.005 % of
each other: **4.44x wall / 4.63x cpu on equal work.**

**The shipped R package**, same protocol, first run after each install discarded (see §8):

| n | OFF (s) | ON (s) | speedup |
|---:|---|---|---|
| 300 | 6.33 / 5.44 / 5.63 | 1.13 / 1.22 / 1.25 | 4.5-5.6x |
| 1000 | 18.05 / 17.06 / 17.39 | 3.03 / 3.12 / 3.04 | 5.5-6.0x |

This is the number that matters: it is measured on the artefact users install, and it proves
the redirect activates inside a DLL loaded by `R.exe` rather than silently falling back.

### 6.1 Follow-up (2026-08-01): what it did to the PySR head-to-head

The §6 speedups are OFF-vs-ON against our own prior code. They say nothing on their own about
the reference tool. This subsection closes that gap for the one Feynman problem where the
wall-clock comparison against SR.jl was *methodologically clean*.

Why `rel_mass` is the right problem to re-measure. On most of the Feynman set a wall-clock
comparison is confounded by the asymmetric stopping criterion `docs/15` §5 records
(rsymbolic2 early-stops at `target_loss`, SR.jl consumes its full budget). `rel_mass` is
the exception in rsymbolic2's favour-free direction: it **recovers** (NMSE ~1e-9) but never
reaches `target_loss = 1e-10`, so every seed runs the full 2800 generations — the same
full-budget behaviour as SR.jl's default `early_stop_condition=None`. Both tools are
therefore timed doing all of the work, and the ratio is pure throughput. (This is the same
property that made `rel_mass` `docs/37`'s thread-scaling test case.)

Gate config, `OMP_NUM_THREADS=4`, 5 seeds, HEAD `d130211` reinstalled from source
(`benchmarks/results/feynman_gate_diag_20260801.csv`):

| | median wall | vs SR.jl |
|---|---:|---:|
| rsymbolic2, redirect OFF (`feynman_gate_20260627.csv`, 5 seeds) | 85.4 s | 2.93x slower |
| **rsymbolic2, redirect ON (2026-08-01, 5 seeds)** | **32 s** | **1.10x** |
| SR.jl / PySR (`sr_comparison_feynman_20260621.csv`, 3 seeds: 20.4 / 29.2 / 51.0) | 29.1 s | — |

**2.67x against our own prior gate number**, inside the 2.27-4.65x this document's §6 measured
for `rel_mass` under a controlled A/B — so the two agree, and the head-to-head deficit is gone:
1.10x sits well inside SR.jl's own 20-51 s seed spread. Recovery is unchanged at 5/5; the NMSE
values do not reproduce the 2026-06-27 run, which is §7's documented consequence on Windows,
not a regression.

Two limits on this result, stated so it is not over-read:

- **One problem, not the set.** The other problems where rsymbolic2 trailed (planck,
  boltzmann_dist, bose_einstein, doppler_rel) are transcendental-heavy in the same way and are
  *expected* to have moved similarly, so the 25-problem totals (1191 s vs 1264 s at the time)
  have probably shifted in rsymbolic2's favour. **That is an inference, not a measurement.**
  The full 25x5 re-run has not been done.
- **The SR.jl arm is the 2026-06-21 CSV**, not a fresh run, and its times exclude Julia JIT
  compilation (warmed up untimed, `benchmarks/05_feynman_pysr_comparison.jl:212`) — a choice
  that favours the reference tool. Both are unchanged from the earlier comparison, so the
  *delta* recorded here is attributable to the redirect.

## 7. The bit-identity break, and its exact blast radius

**Windows results change.** `diag_search_digest` (33 fixed-seed searches, hex-float digests):
**74 of 656 lines differ.** That is the expected consequence of 1-ulp differences steering an
evolutionary search, identical in kind to the Win/Linux divergence `docs/67` §3.3 recorded
and the WASM/native divergence `docs/51` recorded. It is a one-time, audited event against
pre-existing Windows builds, not ongoing nondeterminism: same build, same seed still gives
the same answer.

**Nothing else changes, and this was verified rather than argued:**

- **Linux: `diag_search_digest` is byte-for-byte identical, all 656 lines**, comparing a
  build of the parent commit against a build of this one.
- **WASM: numerically identical to the parent commit** — the parity gate's best expression,
  best loss and all eight `erf` values match exactly. The committed `rsymbolic2.wasm` is 242
  bytes smaller purely from inlining/layout; the parent build reproduces the previously
  committed byte count exactly, so the size change is attributable to this commit and the
  numeric check above is what rules out any effect from it.
- **MSVC** does not compile the redirect (it is already on UCRT).

No stored baselines needed regenerating: `diag_search_digest` is a before/after tool, not a
committed golden file, and the WASM parity gate asserts outcome equivalence rather than bit
equality by design.

## 8. Verification performed

| | Windows | Ubuntu 24.04 (WSL) |
|---|---|---|
| standalone `ctest` | 29/29 | 29/29 |
| R `testthat` (`NOT_CRAN=true`) | 327 pass, 0 fail | 327 pass, 0 fail |
| `pytest` | 66 passed | 57 passed, 9 skipped |
| WASM parity gate | PASSED (emsdk 6.0.2) | — |
| digest vs parent commit | 74/656 differ (expected, §7) | **0/656 differ** |

**Two measurement traps were hit and corrected during this work; both are worth remembering.**

1. **The first end-to-end A/B was invalid.** The OFF arm was configured without
   `CMAKE_BUILD_TYPE`, so it built unoptimised while the ON arm was Release. It reported
   7.56x where the corrected matched-Release comparison reports 4.44x. Always print both
   caches before believing an A/B.
2. **Timing a build immediately after installing it is contaminated.** An R-level A/B run
   right after `R CMD INSTALL` reported 1.1-1.4x; the same binaries, measured after a
   discarded warm-up run, reported 4.5-6.0x. This is the failure mode
   `reference-benchmark-cpu-health` warns about, and it produced a *four-fold* error.

## 9. Corrections to `docs/67`

- **§3.4 and §8 describe `bench_soa_eval` as running "the production SoA evaluator". It did
  not.** Its batch arm was a local prototype written to price the design before it was
  productionised, and it had drifted: it called `std::` directly (so it never saw this
  change) and used an unguarded `sqrt` where production guards it. The driver now calls
  `evaluate_soa_residual`. Its `bit-exact NO` output was the harness correctly reporting its
  own drift — the numbers in `docs/67` §3.4's batch column are the prototype's, not the
  shipped evaluator's.
- **§4 point 2 and §5 estimate the available Windows win as "bounded above by the observed
  2.4-2.7x platform gap and certainly less than it".** The direct A/B measures 4.4x median
  end to end. That estimate came from an n=1 cross-platform comparison with diverging
  trajectories which §3.3 itself labelled indicative; a same-machine, same-session,
  matched-build A/B supersedes it.
- **§6.1's decision (C) and §7's question 1 (vendoring licence) are moot.** The win was
  available with no dependency at all, which is why the cost side of the argument collapsed.

## 10. Incidental finding, not fixed here

`tree.hpp::apply_unary<double>` computes `Sqrt` as unguarded `std::sqrt` (negative → NaN),
while the SoA evaluator and `dual.hpp` guard it (negative → 0). `soa_eval.hpp`'s header
comment claims the SoA path is bit-identical to `apply_unary`; for a negative `sqrt`
argument it is not. This surfaced only because the benchmark was pointed at production
(§9), and it is **pre-existing and not user-visible**: the shipped search and `predict()`
both go through `evaluate_soa_residual`, and `evaluate<double>` now has no production caller
at all — only tests and benchmarks. The existing SoA test never exercises a negative `sqrt`
argument, which is why it was not caught. Left for a separate decision, because closing it
either way changes what `evaluate<double>` returns.

## 11. Reproduction

```
cmake -S . -B build-win     -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-win-off -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DRSYMBOLIC2_NO_UCRT_LIBM"
cmake --build build-win     --target bench_soa_eval bench_profile diag_search_digest -j 8
cmake --build build-win-off --target bench_soa_eval bench_profile diag_search_digest -j 8

./build-win-off/standalone/bench_soa_eval.exe   # repeat >= 5x; V8-free but still noisy
./build-win/standalone/bench_soa_eval.exe
OMP_NUM_THREADS=4 ./build-win-off/standalone/bench_profile.exe spring_pe 300 2
OMP_NUM_THREADS=4 ./build-win/standalone/bench_profile.exe     spring_pe 300 2
diff <(./build-win-off/standalone/diag_search_digest.exe) <(./build-win/standalone/diag_search_digest.exe)
```

For the R-package arm, append `PKG_CPPFLAGS += -DRSYMBOLIC2_NO_UCRT_LIBM` to
`r-package/rsymbolic2/src/Makevars.win`, reinstall, and **discard the first timed run** (§8).
