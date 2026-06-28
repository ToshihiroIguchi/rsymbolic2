# docs/25 — Allocation-free Levenberg-Marquardt backend (SelfLMOptimizer)

## Context

The island-parallel health probe (`benchmarks/diag_omp_check.R`) showed broken
multi-island scaling on Windows: four islands took ~5.9× the wall time of one (ideal
~1×) with `cpu/wall ≈ 1.5`, i.e. threads **blocked**, not throttled (docs/23 §4). This
starves the search budget on the hardest problems.

Allocation counting localized the cause to the constant optimizer. `EigenLMOptimizer`
delegates the Levenberg-Marquardt loop to Eigen's unsupported
`NonLinearOptimization` module, which allocates heavily per solve. On Windows/MinGW
those allocations serialize on the process-wide heap lock, so adding islands adds
contention instead of throughput.

This document records the diagnosis evidence and the fix: a hand-written,
allocation-free LM backend (`SelfLMOptimizer`) that reuses persistent scratch across
fits. It is now the default backend; `EigenLM` remains selectable for comparison.

## Evidence (standalone diagnostic benches)

All on Intel Core 7 150U (2 P-core + 8 E-core, 15 W), Windows 11, Rtools45 g++,
`-O3`. Each bench is bounded; `cpu/wall` was sanity-checked against a pure-FP control
that scales to 3.97 at 4 threads (hardware is healthy).

- **`bench_alloc.cpp`** (malloc counted via `--wrap`), m=1000, per fit:
  - `make_least_squares_problem`: 5 op-new / 5 malloc.
  - **EigenLM** `optimize`: 8 op-new, **43 malloc** → **35 raw (Eigen) allocations not
    seen by `operator new`** (its dynamic matrices use `std::malloc` directly).
  - **SelfLM** `optimize`: 30 op-new, **30 malloc → 0 raw Eigen-style allocations.**
- **`bench_eigen_reuse.cpp`**: reusing the Eigen LM object still leaves ~33 malloc/fit
  — the allocations are intrinsic to Eigen's algorithm, not removable by object reuse.
- **`bench_heap.cpp`** (N independent threads, identical fit work, no islands/OpenMP):

  | workload | wall(1) | wall(4) | cpu/wall(4) |
  |---|---|---|---|
  | EigenLM optimize | 15.0 s | 109.7 s | **1.41** (blocked) |
  | **SelfLM optimize** | **1.98 s** | **12.4 s** | **2.92** |
  | residual-only (no Eigen; ceiling) | 1.80 s | 3.86 s | 3.74 |
  | pure-FP control | — | — | 3.97 |

  SelfLM is ~7.6× faster per fit and lifts 4-thread `cpu/wall` from 1.41 to 2.92
  (≈8.8× the throughput at 4 threads). The remaining gap to the residual-only ceiling
  (3.74) was hypothesized to be the per-evaluation temporaries the residual/Jacobian
  closures still allocate (~30 small vectors/fit in `least_squares_problem.hpp`).
  That hypothesis was later tested and **disproven** — see "Closure scratch reuse"
  below: removing those allocations did not move `cpu/wall`.

## Design

`SelfLMOptimizer` (`r-package/rsymbolic2/src/rsymbolic/optimization/self_lm_optimizer.hpp`,
`.../src/self_lm_optimizer.cpp`) implements `ConstantOptimizer`, so it is a drop-in
behind the existing factory/abstraction. It **reuses the verified AD machinery
unchanged** — the dual-number residual/Jacobian closures from
`make_least_squares_problem` — and replaces only the linear algebra and iteration.

Algorithm: normal-equations Levenberg-Marquardt with Marquardt diagonal scaling.
Each iteration:

1. residuals `r` (m) and Jacobian `J` (m×k) from the existing closures; non-finite
   values clamped to `1e10` (matching the Eigen backend's `fill_residuals`). When no
   analytic Jacobian is supplied, a forward-difference fallback fills `J`.
2. form `A = JᵀJ` (k×k) and `g = Jᵀr` (k) by point loops (no temporaries);
3. solve `(A + λ·diag(A)) δ = -g` with a hand-written in-place Cholesky (k is small,
   typically ≤ 5; ≤ ~10 is safe). The per-column damping is floored so a null column
   stays positive-definite. A non-positive pivot signals "increase λ".
4. accept the trial `p+δ` if SSE decreases (then λ ×= 0.1), else increase λ (×10) and
   re-solve **reusing A, g** — a bounded damping search per iteration;
5. stop on small `‖g‖∞`, small `‖δ‖/‖p‖`, small relative SSE reduction, the iteration
   cap (`OptimizerConfig::max_iterations`), `stop_requested()`, or the abort flag.

**Allocation-free across fits:** every workspace (`params/trial/rbuf/trial_rbuf/jbuf/
A/aug/g/delta`) is a persistent `mutable` member, sized once with `resize()` (grows
only). `mutable` is thread-safe under the same contract as the Eigen backend: each
island owns its optimizer instance and never calls `optimize()` on it concurrently.

**Determinism** is structural: only floating-point arithmetic, no RNG. A fit is
reproducible and independent of thread count — required for the island-model `np=1`
regression and thread-count determinism gates (both green).

Semantics preserved from the Eigen backend: null residuals throw; `k==0` reports a
single SSE evaluation; the abort flag is reset per `optimize()`; `success` ⇔ finite
loss. Numerical note: normal equations square the condition number, acceptable for the
small k here (verified on `y=a·exp(b·x)`); an allocation-free QR could replace the
solve if an ill-conditioned case ever demands it, without changing the interface.

## Validation

- **Unit tests** (`standalone/tests/test_self_lm_optimizer.cpp`, in CTest): linear and
  exponential recovery via the real AD Jacobian, finite-difference fallback agreement,
  factory reachability (`self_lm`), reproducibility, null-throws, `k==0`, immediate
  stop. Standalone suite **16/16 green on Windows and Ubuntu (WSL)**.
- **Recovery (no regression)** — standard 6-problem suite, pop=600, gen=100, 5 runs,
  identical seeds, `standalone/benchmarks/benchmark_main.cpp <pop> <gen> <runs> <opt>`:

  | problem | SelfLM | EigenLM |
  |---|---|---|
  | linear, exponential, Nguyen-1, multivar | 5/5 each | 5/5 each |
  | Nguyen-5 | 0/5 | 0/5 |
  | Nguyen-7 | 1/5 | 1/5 |
  | **total** | **21/30 (70%)** | **21/30 (70%)** |

  Identical recovery problem-for-problem, with machine-precision test error on the
  recovered cases — the speedup is real efficiency, not under-convergence. SelfLM was
  faster on every problem.

- **End-to-end island scaling** (`benchmarks/diag_omp_check.R`, center_mass, fixed
  work), SelfLM as default vs the EigenLM baseline recorded in docs/23 §4:

  | | np=1 wall | np=4 wall | cpu/wall(4) | wall(4)/wall(1) |
  |---|---|---|---|---|
  | EigenLM (baseline) | 14.7 s | 86.9 s | 1.58 | 5.9× |
  | **SelfLM** | **3.2 s** | **14.5 s** | **1.81** | **4.5×** |

  Large absolute win (np=1 4.6× faster, np=4 6× faster). The *scaling ratio* gain in
  the full search is smaller than the optimizer-only microbench (1.41 → 2.92): the
  island search still pays the per-evaluation closure allocations plus migration /
  epoch-barrier overhead the microbench excludes. Closing that gap is the deferred
  closure-allocation task below.

## Closure scratch reuse — measured, did NOT move scaling

The deferred follow-up above (remove the residual/Jacobian closures' per-evaluation
allocations) was implemented and measured. The hypothesis — that those ~30 small
vectors/fit were the remaining island-scaling lever — **is not supported by the
measurements.** Recording the negative result here so it is not re-attempted.

Change (`least_squares_problem.hpp`): the residual closure's `stack`, and the Jacobian
closure's `constants` and `stack`, were moved from per-call locals into the `Model`
struct (lifetime-managed by `shared_ptr`) and reused across calls. The `Model` is built
locally per `fit()` and never shared across threads, and the two closures are called
sequentially within one `optimize()`, so the reuse is safe under the same contract as
the optimizer's own scratch. The change is **numerically identical** (same FP ops in the
same order; buffers are fully overwritten before use). Same machine/build as above.

- **`bench_alloc`** SelfLM `optimize`: **30 op-new / 30 malloc → 4 / 4** per fit (raw
  Eigen-style allocations still 0). The per-evaluation allocations are gone as intended.
- **`bench_heap`** SelfLM 4-thread `cpu/wall`: **2.92 → ~2.90** (2.89/2.90/2.93 over
  three runs; residual-only ceiling ~3.7; pure-FP control 3.9). **Unchanged.**
- **`diag_omp_check.R`** (center_mass), SelfLM default: np=4 `cpu/wall` **1.81 → 1.79**
  (np=1 wall 3.4 s, np=4 wall 13.7 s). **Unchanged.**
- **Recovery (no regression)**, standard suite pop=600 gen=100, 2 runs, identical seeds:
  SelfLM **9/12 == EigenLM 9/12**, problem-for-problem identical. Standalone **16/16 on
  Windows and Ubuntu (WSL)**.

Interpretation: `bench_heap` is exactly the same-process multithreaded scenario where a
heap-lock bottleneck would surface, and removing 26 allocations/fit there did not change
`cpu/wall`. So the residual gap (SelfLM ~2.9 vs residual-only ceiling ~3.7) is **not**
allocation contention — it is the heavier compute of the dual-number Jacobian (k passes
of dual arithmetic + the LM solve, several × the residual-only FP/libm work) hitting the
frequency/bandwidth ceiling. In the full island search (`diag_omp`) fits are only ~10%
of the work (`optimize_probability=0.1`), so its `cpu/wall ≈ 1.8` is limited by other
serial/contended sections (migration, epoch barriers, tree ops), not by fit allocations
at all.

The change is **kept** regardless: it is a genuine heap-churn reduction (lower malloc
pressure, removes a latent malloc-lock risk under high island counts), numerically
exact, no regression, low complexity. It simply is not the scaling lever that was
hypothesized. Further island-scaling work must target the dual-Jacobian compute or the
search's serial sections, not closure allocation.

## Status and follow-ups

- `SelfLMOptimizer` is the **default** (`SearchOptions::optimizer_type`); `EigenLM`
  stays available via the factory for comparison.
- Closure per-evaluation allocations: **done** (above) — reduced 30 → 4/fit, but did not
  improve `cpu/wall`; the bottleneck lies elsewhere (dual-Jacobian compute / search
  serial sections).
- Deferred (separate tasks): (1) Feynman/Nguyen recovery vs PySR at documented defaults;
  (2) eventual removal of the Eigen dependency.
