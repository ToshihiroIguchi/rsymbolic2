# Benchmark Implementation Plan

**Date:** 2026-06-06
**Scope:** Phase 2 gate benchmark (Nguyen suite) — development evidence only.
This plan does NOT cover the publication-quality benchmark (Phase 5 / SRBench++).

---

## RESULTS (2026-06-06) — GATE: PASS

First run of this benchmark. rsymbolic2 0.1.0, Windows 11, Rtools45 MinGW/UCRT,
OpenMP, 4 islands. Raw data: `benchmarks/results/nguyen_gate_20260606.csv`.

| Problem | Recovered | median NMSE | median time | Notes |
|---------|-----------|-------------|-------------|-------|
| N1 | 5/5 | 3e-33 | 11s | |
| N2 | 5/5 | 2e-32 | 28s | |
| N3 | 5/5 | 1e-14 | 129s | |
| N4 | 5/5 | 2e-13 | 203s | seed5 = 2966s outlier |
| N5 | 5/5 | 8e-31 | 25s | |
| N6 | 5/5 | 1e-13 | 66s | |
| N7 | 5/5 | 5e-13 | 62s | |
| N8 | SKIP | — | — | sqrt operator not implemented |
| N9 | 5/5 | — | — | seed5 completed in probe run (1127s, recovered) |
| N10 | 5/5 | — | — | completed in probe run (`02_blowup_probe.R`) |

**Verdict: PASS (9/9 active problems recovered).** Gate threshold is 7 of 9.
N1–N7, N9, N10 all recovered on all 5 seeds. N9 seed5 and all of N10 were completed
in a follow-up probe run (`benchmarks/02_blowup_probe.R`, with the new timeout +
instrumentation) — data in `benchmarks/results/blowup_probe_20260606.csv`. All
recoveries are far below the NMSE < 1e-4 threshold. The search core demonstrably
recovers polynomial, trigonometric, and logarithmic equations.
**Phase 2 may continue to Phase 3.**

### Earlier defect found and fixed during this run

The first attempt was ~6–10× slower than estimated. Root cause: `run_evolution`'s
outer epoch loop had no global early stop — once one island reached `target_loss`,
the other islands and remaining epochs still ran their full budget. Fixed by adding
a cross-island target check that breaks the outer loop. Verified: N1 seed1 dropped
from 317s → 2.6s with identical recovery. This is a product-level efficiency fix,
not a benchmark hack; it only triggers at loss < 1e-10, far below the 1e-4 recovery
threshold, so it cannot affect recovery classification.

### Open finding: runtime blowup on some seeds (NOT a gate blocker)

Even after the early-stop fix, three runs were pathological outliers:
N4 seed5 (2966s), N9 seed1 (1786s), N9 seed4 (1269s), and N9 seed5 (DNF, killed at
~96min while still actively computing — not deadlocked, CPU was accumulating).

All of these still *recovered* (where they finished) — the issue is wall-time, not
correctness. The likely cause is **expression bloat**: when a population fills with
large trees carrying many constants, each Levenberg-Marquardt constant-optimization
call grows expensive, and per-generation cost explodes. `simplify=TRUE` was on but
did not contain it. These runs never reach `target_loss=1e-10` quickly, so the
early-stop cannot rescue them and they consume the full generation budget at a high
per-generation cost.

**This is the most important thing this benchmark surfaced.** It is the top
investigation item before Phase 3 polish:
- Confirm bloat is the cause (log median/max tree size and constant count per generation).
- Consider: a hard per-run wall-clock timeout (so a single run cannot block a suite);
  stronger parsimony pressure (Phase 2.7 adaptive parsimony, still unimplemented);
  capping `max_nodes` more aggressively; or limiting LM constants per tree.
- Re-run N9 seed5 and N10 in isolation once addressed, to complete the table.

---

## 1. Purpose

The Phase 2 gate decision requires evidence that the search recovers equations.
Without it, all further work (adaptive parsimony, more operators, R interface polish)
may be building on a broken foundation.

This plan defines the minimum viable benchmark that answers the gate question:
> "Can rsymbolic2 recover symbolic equations from synthetic data at a competitive rate?"

---

## 2. Scope of This Benchmark

### In scope
- Nguyen symbolic regression benchmark (N1–N10), a standard community-agreed suite
- 5 independent runs per problem (development gate; publication requires 30)
- Fixed hyperparameters — not tuned on this test set
- Results stored as CSV for regression tracking

### Out of scope (deferred to Phase 5)
- PySR/Operon head-to-head comparison (requires matching setup; blocked on build)
- Feynman benchmark (requires more operators: sqrt, pow, tanh; Phase 4 gate)
- SRBench++ (requires 30+ runs × 252 datasets; publication-scale work)
- Statistical tests (Wilcoxon, critical difference diagrams — too few runs here)

---

## 3. Nguyen Benchmark Problems

Standard Nguyen benchmark (Uy et al. 2011). Training data: n=20 uniform random
samples per the ranges below, generated with a fixed seed to ensure reproducibility
across runs and algorithms.

| ID  | Formula                          | Domain (x, y)        | Operators needed   | Status        |
|-----|----------------------------------|----------------------|--------------------|---------------|
| N1  | x³ + x² + x                     | x ∈ [−1, 1]          | +, *, neg          | ✓ runnable    |
| N2  | x⁴ + x³ + x² + x               | x ∈ [−1, 1]          | +, *               | ✓ runnable    |
| N3  | x⁵ + x⁴ + x³ + x² + x         | x ∈ [−1, 1]          | +, *               | ✓ runnable    |
| N4  | x⁶ + x⁵ + x⁴ + x³ + x² + x   | x ∈ [−1, 1]          | +, *               | ✓ runnable    |
| N5  | sin(x²)·cos(x) − 1             | x ∈ [−1, 1]          | sin, cos, *, neg   | ✓ runnable    |
| N6  | sin(x) + sin(x + x²)           | x ∈ [−1, 1]          | sin, *, +          | ✓ runnable    |
| N7  | log(x+1) + log(x²+1)           | x ∈ [0, 2]           | log, +, *          | ✓ runnable    |
| N8  | √x                               | x ∈ [0, 4]           | sqrt               | ✗ SKIP (sqrt not implemented) |
| N9  | sin(x) + sin(y²)               | x,y ∈ [−1, 1]        | sin, *, +          | ✓ runnable    |
| N10 | 2·sin(x)·cos(y)                | x,y ∈ [−1, 1]        | sin, cos, *, +     | ✓ runnable    |

**Active problems: N1–N7, N9–N10 (9 problems).**
N8 is recorded as SKIP; it becomes active after the sqrt operator is implemented.

---

## 4. Fixed Hyperparameters

These are set BEFORE running the benchmark and are not tuned on benchmark results.
The rationale for each value follows.

```r
BENCH_PARAMS <- list(
  population_size       = 500,   # larger than default (200); better diversity
  n_populations         = 4,     # 4 islands: meaningful parallelism at modest overhead
  generations           = 200,   # ~16s / run on Windows with 4 threads (estimated)
  tournament_size       = 5,
  unary_ops             = c("neg", "exp", "log", "sin", "cos"),
  binary_ops            = c("add", "sub", "mul", "div"),
  max_depth             = 6,
  max_nodes             = 50,
  target_loss           = 1e-10,  # early stop only if near-perfect
  simplify              = TRUE,
  crossover_probability = 0.5
)
```

**Rationale:**
- `generations=200` with `population_size=500 × n_populations=4` gives ~400k total candidate
  evaluations — enough for Nguyen-level complexity without exceeding a 60-second budget.
- `div` is included despite safe-division concerns: Nguyen problems don't require division,
  but including it tests that the optimizer handles near-zero denominators without crashing.
- `simplify=TRUE` reduces bloat and aids constant optimization on the remaining structure.

**Estimated runtime per run (based on parallel efficiency results):**
- 0.26 ms per (member × generation), single-threaded  
- 4 islands × 500 × 200 gens = 400k → ~104s single-threaded  
- With 4 threads and 0.89 efficiency at 2T, 0.64 at 3T: expect ~30–40s per run  
- Total (9 problems × 5 seeds): ~25–30 minutes

---

## 5. Protocol

### 5.1 Data Generation

- Training data seed: fixed at `data_seed = 123` (same across all runs; only the
  symbolic regression seed varies)
- n = 20 points per problem (standard Nguyen protocol)
- No separate test set for this gate benchmark — NMSE is evaluated on training data
  (consistent with exact-recovery framing: if the formula is recovered, training NMSE → 0)

### 5.2 Run Protocol

Seeds: 1, 2, 3, 4, 5 (fixed sequence; same seeds will be used for PySR comparison later).

```r
for (problem in problems) {
  for (seed in 1:5) {
    dataset <- generate_dataset(problem, data_seed = 123)
    t0 <- proc.time()["elapsed"]
    result <- symbolic_regression(dataset$X, dataset$y, seed = seed, ...BENCH_PARAMS)
    elapsed <- proc.time()["elapsed"] - t0
    nmse <- compute_nmse(result$loss, dataset$y)
    record(problem$id, seed, elapsed, result$loss, nmse, result$expression)
  }
}
```

### 5.3 NMSE Computation

The `loss` field returned by `symbolic_regression()` is the sum of squared residuals (SSR):

    loss = Σᵢ (ŷᵢ − yᵢ)²

NMSE (Normalized Mean Squared Error):

    NMSE = SSR / Σᵢ (yᵢ − ȳ)²
         = loss / ((n − 1) × var(y))

This is scale-independent and the standard threshold for Feynman/Nguyen recovery is
`NMSE < 1e-4` (approximately 4 significant digits of agreement).

### 5.4 Recovery Classification

A single run is "recovered" if `NMSE < 1e-4`.
A problem is "recovered" if at least 3 of 5 seeds recover it (majority).

---

## 6. Gate Criterion

**Pass:** 7 or more of 9 active Nguyen problems are "recovered" (majority across 5 seeds).

**What a fail means:** If fewer than 7/9 problems pass:
1. Check whether polynomial problems (N1–N4) are recovered. If not, the mutation set
   or constant optimizer is broken at a fundamental level — redesign required.
2. If N1–N4 pass but N5–N7 fail: trig/log search likely has insufficient operator
   scope or max_nodes. Tune hyperparameters conservatively and rerun.
3. If recovery is stochastic (some seeds pass, others fail): investigate whether
   the loss landscape has multiple basins. Consider increasing n_populations or
   migration frequency.

---

## 7. Output Format

Results saved as `benchmarks/results/nguyen_gate_YYYYMMDD.csv`:

```
problem_id, seed, elapsed_sec, loss, nmse, recovered, expression
N1, 1, 14.2, 3.2e-11, 5.1e-12, TRUE, "((x * x) + x)"
N1, 2, ...
```

Summary printed to console:
```
Nguyen Gate Benchmark — 2026-06-06
====================================
N1  [x^3+x^2+x]       5/5 recovered   med NMSE=2e-12   med time=13s
N2  ...
...
====================================
Gate: 8/9 PASS (threshold 7/9)
```

---

## 8. File Structure

```
benchmarks/
  nguyen_datasets.R     Dataset definitions (formulas, domains, generators)
  utils.R               Shared helpers (nmse, save_results, print_summary)
  01_nguyen_gate.R      Gate benchmark runner (sources the above two)
  results/              CSV output files (gitignored by default)
```

---

## 9. Deferred Work

The following benchmark components are deferred until the Phase 2 gate passes:

| Item | Blocked on | When |
|------|-----------|------|
| N8 (sqrt(x)) | sqrt operator implementation | After Phase 1 operator extension |
| PySR comparison | PySR Python environment setup | Phase 3 milestone |
| Feynman subset (N=10) | sqrt, pow, tanh operators | Phase 4 gate |
| 30-run statistical analysis | Too slow for development gate | Phase 5 |
| Parallel efficiency on Ubuntu | Ubuntu CI setup | Phase 3 milestone |

---

## 10. Anti-Patterns Avoided

Per `docs/04_benchmark_strategy.md`:
- Hyperparameters fixed before running (Section 4).
- 5 seeds reported (not single best run).
- Full 9-problem table reported (no cherry-picking).
- NMSE threshold 1e-4 (standard; not weakened).
- No comparison to PySR until PySR setup is independently verified.
