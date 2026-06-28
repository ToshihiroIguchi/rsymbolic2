# 36. BFGS and Float32, implemented and measured

**Date:** 2026-06-27
**Motivation:** `docs/32` Part C *recommended against* implementing BFGS and Float32 for the
constant optimizer, on reasoning (allowed divergences; LM is the right algorithm for
least-squares; Float32 has no SIMD lever in our scalar interpreter). The user asked to stop
reasoning and **implement both and measure**. This file records the measured result so the
conclusion is now evidence-backed, not predicted.

## What was implemented

A standalone evaluation harness, `standalone/benchmarks/bench_opt_precision.cpp` (CMake
target `bench_opt_precision`), implementing **real** (not stub) templated solvers driven by
the production evaluator:

- **LM\<T\>** — Levenberg-Marquardt, mirroring `self_lm_optimizer.cpp` (normal equations
  `A=JᵀJ`, Marquardt damping, in-place Cholesky), parameterised on the scalar type `T`.
- **BFGS\<T\>** — textbook inverse-Hessian BFGS with backtracking Armijo line search (the
  `Optim.jl`/PySR default optimizer), parameterised on `T`.
- Both run the **SR.jl multi-start** (start 0 + `n_restarts` perturbed starts) and are fed
  the **same** analytic Jacobian (`k` forward-mode dual passes) via a templated `GDual<T>`
  and the production `evaluate<T>` / `evaluate<GDual<T>>`. So LM-vs-BFGS and f64-vs-f32 are
  apples-to-apples.

`T = float` exercises a genuine Float32 fit (residual, Jacobian, normal equations / Hessian,
linear solve all in single precision); `T = double` is the baseline.

**Why a harness, not an engine change.** The shipped `ConstantOptimizer` /
`OptimizationProblem` interface is hard-locked to `std::vector<double>` and the AD types
(`Dual`/`MultiDual`) are double-only; templating the whole engine + R bindings to float is
disproportionate to a measurement. The fit is ~93 % of search compute (`docs/30`), so a
faithful **fit-level** measurement answers the speed question directly.

## Settings

PySR per-fit budget (`docs/28`): `optimizer_nrestarts = 2`, `optimizer_iterations = 8`.
m = 1000 points, k = 2–3 constants, common perturbed start across all four cells, median of
400 fits. Trees ordered by transcendental density (poly → rel_mass → trig → transc).

## Result — speed (µs per fit, `-O2` generic, the flag the R package ships)

| problem  | LM f64 | LM f32 | BFGS f64 | BFGS f32 |
|----------|-------:|-------:|---------:|---------:|
| poly     |  1774  |  1747  |   3596   |   3745   |
| rel_mass |  2431  |  2712  |   3211   |   3065   |
| trig     | 14893  | 16235  |  20247   |  21218   |
| transc   | 11154  | 11430  |  23648   |  24457   |

Derived ratios:

- **BFGS / LM (f64): 1.32× – 2.12× SLOWER.** (poly 2.03×, rel_mass 1.32×, trig 1.36×,
  transc 2.12×.) BFGS needs value+gradient per line-search trial; LM's Gauss-Newton step
  extracts more progress per Jacobian.
- **Float32 / Float64 (LM): no speedup — 0.98× to 1.12× (mostly slightly SLOWER).** At
  `-O3 -march=native` Float32 is *worse* (e.g. rel_mass 1.74× slower). Same as the
  evaluator micro-benchmark (`docs/30` Optimization 2): our scalar interpreter has no
  data-parallel lane for Float32 to widen, and MinGW's `expf/logf/...` are not faster than
  the double forms, so the only net effect is extra float↔double conversion.

## Result — quality (lower is better; SSE and max constant relative error)

| problem  | LM f64 SSE | LM f32 SSE | BFGS f64 SSE | BFGS f32 SSE |
|----------|-----------:|-----------:|-------------:|-------------:|
| poly     | 7.7e-17    | 3.8e-12    | 0.0          | 4.3e-12      |
| rel_mass | 0.0        | 0.0        | 5.0e-14      | 0.0          |
| trig     | 0.0        | 3.2e-12    | 1.2e-8       | 1.2e-8       |
| transc   | 2.3e-14    | 2.9e-12    | **5.9e-3**   | **6.1e-3**   |

Two correctness costs, both measured:

1. **Float32 raises the achievable loss floor ~100–1000×** (SSE ~1e-12 vs ~1e-16…0; const
   error ~1e-7 vs ~1e-9). Against the Feynman gate's `target_loss = 1e-10` and the NMSE
   recovery thresholds, a ~1e-12 *absolute* SSE floor is uncomfortably close to the bar for
   some problems — a recovery risk for the smallest possible speed gain (which is ≤ 0 here).
2. **BFGS does not converge as well as LM at the PySR per-fit budget.** On the
   transcendental-heavy `transc` it leaves SSE = 5.9e-3 versus LM's 2.3e-14 — ten orders of
   magnitude worse — and on `trig` 1.2e-8 vs 0. To match LM's fit BFGS would need many more
   iterations, making the already-slower per-fit time worse still.

## Conclusion

The measurement **confirms `docs/32` Part C with hard numbers**:

- **BFGS:** 1.3–2.1× slower per fit *and* lower fit quality at equal budget. No case where it
  wins. LM (Gauss-Newton) is the correct solver for this least-squares problem.
- **Float32:** zero speedup (often a slight slowdown) on the Rtools/MinGW scalar evaluator,
  with a ~100–1000× worse loss floor. It would trade Correctness (priority #1) for a
  non-existent Performance (priority #5) gain.

Neither is worth integrating into the shipped engine. Both remain CLAUDE.md *allowed
divergences*, not parity requirements, so keeping self-LM + Float64 is also the
parity-correct choice. The harness is retained as reproducible evidence; the engine is
unchanged.

**Reproduce:**
```
g++ -std=c++17 -O2 -I r-package/rsymbolic2/src \
    standalone/benchmarks/bench_opt_precision.cpp -o bench_opt && ./bench_opt 1000 400
```
