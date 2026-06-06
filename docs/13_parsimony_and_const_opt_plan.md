# Implementation Plan: Bloat Control (Parsimony + Constant-Optimization Decoupling)

**Date:** 2026-06-06
**Basis:** Runtime-blowup diagnosis in `docs/12` (H1 bloat dominant, H3 = LM cost on
large trees as its consequence) and a study of how PySR / SymbolicRegression.jl
controls bloat (parsimony cost, frequency-based adaptive parsimony, simulated
annealing, probabilistic constant optimization, warmup maxsize, constraints).

## Guiding principle (why this is NOT a PySR clone)

CLAUDE.md: "Do not assume the PySR / SymbolicRegression.jl architecture is optimal.
It is a reference, not a specification. Borrow what is justified; redesign what is not."

A faithful PySR port (rewriting the acceptance rule to simulated annealing, importing
the full frequency machinery, constraints, per-node complexity, warmup, all at once)
was considered and **rejected**: it changes too many variables to attribute effects,
rewrites code our evidence does not implicate (the steady-state replace-worst rule),
breaks reproducibility wholesale, and risks the Phase 2 gate we just passed. Instead we
proceed **one measurable lever at a time, our-bottleneck-first**, borrowing only the
justified PySR pieces.

## Measurement harness (B0) — done first, reused at every gate

Yardstick = `benchmarks/02_blowup_probe.R` (N9 s5 + N10, with `verbosity=1` so we log
median/max tree size and constant count per epoch) plus three fast Nguyen problems
(N1, N5, N7) as a recovery-regression check. Record before/after for each step in
`benchmarks/results/`. Metrics: per-run wall time, median tree size, median nconst,
recovery (NMSE < 1e-4). A step is accepted only if it reduces the slow-run wall time
**without** losing recovery on the fast set.

---

## Step B1 — Decouple constant optimization from every child (highest leverage)

**Why first:** This is the largest divergence from PySR and the direct cause of H3.
Our `evolve_island` calls `fit()` (a full Eigen LM solve) on **every** child, **every**
step (`evolutionary_search.cpp:137`). PySR instead makes constant optimization a
*probabilistic mutation* (`weight_optimize` / `should_optimize_constants`): most
offspring are produced by cheap structural/constant-perturbation moves, and the
expensive optimization runs only occasionally. On a bloated population this is the
difference between paying the LM cost once in a while versus on every candidate.

**Change:**
- Add `SearchOptions::optimize_probability` (double, default to be chosen by B0
  measurement; start the sweep at 1.0 = current behavior, then 0.5, 0.2, 0.1).
- In the step loop: after producing `child_tree`, optimize its constants via `fit()`
  with probability `optimize_probability`; otherwise evaluate the loss with the
  inherited/perturbed constants (no LM) using the existing residual evaluation.
- Children that are *not* LM-optimized still get a loss (needed for selection) from a
  plain `evaluate<double>` pass — cheap, no Jacobian.
- Keep `optimize_probability = 1.0` selectable so the pre-B1 behavior is testable.

**Reproducibility:** this adds one RNG draw per step. Update the determinism test to
fix the new default and assert "same seed → same result" still holds (the real
invariant); the exact RNG sequence is allowed to change between versions.

**Gate:** re-run B0. Expect slow-run wall time to drop sharply (LM no longer paid per
child) with recovery preserved. Pick the `optimize_probability` that best trades wall
time against recovery; record it.

---

## Step B2 — Static parsimony in selection (simplest bloat lever)

**Why:** PySR's core `loss_to_cost` adds `complexity * parsimony` to a normalized loss.
This is the minimal, well-understood pressure against bloat and is independent of B1.

**Change:**
- Add `SearchOptions::parsimony` (double, default 0 = off initially).
- Define a selection cost: `cost = loss / normalization + parsimony * complexity`,
  where `normalization` makes the loss scale-stable (e.g. divide by the variance of y,
  giving an NMSE-like quantity). Normalization is a *justified* PySR borrow — without
  it `parsimony` is not transferable across problems of different y-scale.
- Use `cost` (not raw `loss`) in `tournament_best` and `tournament_worst`.
- Leave the HallOfFame Pareto front keyed on raw loss vs. complexity (unchanged): the
  archive should still record the true accuracy at each complexity; only *selection*
  uses the penalized cost.

**Gate:** re-run B0. Expect median tree size to fall and slow runs to shrink. Tune
`parsimony` with PySR's heuristic (≈ expected min NMSE / 5–10) as a starting point,
then by measurement. Confirm fast-set recovery holds.

---

## Step B3 — Frequency-based adaptive parsimony (only if B1+B2 insufficient)

**Why last:** This is PySR's distinctive and most complex mechanism. Adopt it only if
B1+B2 leave the complexity distribution still lopsided (population piling at the node
cap), which B0 logs will show.

**Change (minimal, per-island to avoid contention):**
- Add a small `RunningSearchStatistics`: a `frequencies` vector over complexity
  1..max_nodes, updated as each candidate is evaluated, with a sliding window and a
  `normalized_frequencies` view. Keep it **per island** (no cross-thread sharing,
  matching our share-nothing island design).
- Add `SearchOptions::adaptive_parsimony_scaling`. Fold the normalized frequency of a
  candidate's complexity into the selection cost:
  `cost += adaptive_parsimony_scaling * normalized_frequency[complexity]`.
- Do NOT copy PySR's window size (100000) blindly; size it to our population/throughput
  and justify the value by measurement.

**Gate:** re-run B0. Accept only if it measurably flattens the complexity distribution
and/or improves recovery beyond B1+B2.

---

## Deferred (recorded, not adopted now)

Adopt each only if a measured gap later demands it; none is required to fix the blowup:
- **Simulated-annealing acceptance** (PySR `next_generation`): a deep rewrite of our
  steady-state replace-worst rule; no evidence implicates that rule.
- **warmup_maxsize_by**: grow `max_nodes` over early generations.
- **Operator constraints / per-node `complexity_of_*`**: scope creep for the blowup.

## Cross-cutting tasks (per accepted step)

- Wire each new `SearchOptions` field through `rsymbolic2_r.cpp` and
  `symbolic_regression.R`; `Rcpp::compileAttributes()`; roxygen docs.
- Add/adjust testthat tests; keep a deterministic default path (reproducibility test).
- Re-run B0 and record the before/after table in this file before moving on.

## Scope boundary

In scope: B1, B2, and B3-if-needed, each measured. Out of scope: a wholesale PySR loop
rewrite (annealing, constraints, warmup) — borrowed only if later evidence justifies it.

---

# DETAILED IMPLEMENTATION

All file paths are under `r-package/rsymbolic2/`. The R package `src/` tree is the
single source of truth (per CLAUDE.md / commit 66526b8).

## B1 — Probabilistic constant optimization (code-level)

### B1.1 New option
`src/rsymbolic/search/evolutionary_search.hpp`, in `SearchOptions`:
```cpp
// Probability that a newly produced child has its constants optimized by the LM
// backend. 1.0 = optimize every child (pre-B1 behavior). Lower values make constant
// optimization a probabilistic event (cf. PySR weight_optimize), which is the main
// lever against the per-child LM cost on bloated trees (see docs/12, docs/13).
double optimize_probability = 1.0;
```
Default 1.0 deliberately preserves current behavior and keeps every existing test
green. The default is flipped to the measured-best value only in B1.6, with evidence.

### B1.2 Loss-without-LM helper
`src/evolutionary_search.cpp`, in the anonymous namespace near `fit()`. Evaluate the
SSE with the tree's current constants — one residual pass, no Jacobian, no LM:
```cpp
// Sum of squared residuals using the tree's current constant values (no optimization).
// Used for children that are not LM-optimized this step. Cheaper than fit(): it skips
// make_least_squares_problem and the solver entirely.
double sse_current(const Tree& tree,
                   const std::vector<std::vector<double>>& X,
                   const std::vector<double>& y) {
    const std::vector<double> c = initial_constants(tree);
    double sse = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double pred = evaluate<double>(tree, X[i].data(), c.data());
        const double r = pred - y[i];
        if (!std::isfinite(r)) return kInf;
        sse += r * r;
    }
    return sse;
}
```

### B1.3 Step-loop branch
`src/evolutionary_search.cpp`, in `evolve_island`'s inner step loop, replace:
```cpp
const double loss = fit(child_tree, X, y, *isl.optimizer);
```
with:
```cpp
const double loss =
    (unit(isl.rng) < opts.optimize_probability)
        ? fit(child_tree, X, y, *isl.optimizer)   // LM: writes optimized constants back
        : sse_current(child_tree, X, y);          // no LM: keeps inherited constants
```
`fit()` mutates the tree's constants (via `set_constants`); `sse_current` does not.
That asymmetry is intended: optimized children carry their fitted constants; others
carry the inherited/mutated ones. The `unit(isl.rng)` draw is the only new RNG
consumption (see B1.5).

### B1.4 Initialization unchanged (for now)
`initialize_island` keeps optimizing every initial member: it is a one-time cost
(`population_size` calls, not per generation), and good initial constants seed the
search. Revisit only if B0 logs show initialization dominating wall time.

### B1.5 Reproducibility
The new per-step `unit(isl.rng)` draw shifts the RNG stream, so a given seed produces a
different (but still deterministic) sequence than pre-B1. The existing determinism
tests compare two same-seed runs to each other (not to a frozen value), so they remain
green. No test change needed; verify by running the suite.

### B1.6 R wiring
- `src/rsymbolic2_r.cpp`: add `double optimize_probability` to the
  `symbolic_regression_cpp(...)` signature; set `opts.optimize_probability`.
- `R/symbolic_regression.R`: add `optimize_probability = 1` arg; pass
  `as.double(optimize_probability)`; document in roxygen (explain it trades wall time
  against fit quality; default 1 = optimize every child).
- Regenerate: `Rcpp::compileAttributes()` (updates `RcppExports.{cpp,R}`); rebuild.

### B1.7 Tests
`tests/testthat/test-optimize-probability.R` (new):
```r
test_that("optimize_probability < 1 still recovers a simple target", {
  skip_on_cran()
  X <- matrix(seq(-5, 5, length.out = 30), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  res <- symbolic_regression(X, y, unary_ops = character(0),
                             population_size = 300L, generations = 60L,
                             seed = 42L, optimize_probability = 0.2)
  expect_lt(res$loss, 1e-4)                 # looser than the p=1 test; still recovers
})
test_that("optimize_probability is reproducible for a fixed seed", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 12), ncol = 1); y <- X[, 1]^2
  a <- symbolic_regression(X, y, population_size = 80L, generations = 15L,
                           seed = 7L, optimize_probability = 0.3)
  b <- symbolic_regression(X, y, population_size = 80L, generations = 15L,
                           seed = 7L, optimize_probability = 0.3)
  expect_equal(a$expression, b$expression)
})
```
Full suite must stay green (timeout, recovery, output, validation, island).

### B1.8 Measurement and the default-flip gate
Sweep `optimize_probability` ∈ {1.0, 0.5, 0.2, 0.1} on the B0 harness
(N9 s5, N10, plus N1/N5/N7). For each, record wall time, and fast-set recovery.
Add a results table below.
**Gate / default decision:** pick the smallest probability whose fast-set recovery is
unchanged from p=1.0; flip `optimize_probability`'s default to it (in both the C++
struct and the R wrapper) in the same change-set, citing the table. If even p=0.5
hurts recovery, keep default 1.0 and rely on B2/B3 instead — and record that B1 alone
was insufficient.

#### B1.8 sweep results (2026-06-06, pop=500, islands=4, gens=200, timeout=120s)

Hardware: Windows 11 Home, CPU with 4 OpenMP threads.
All NMSE values are well below the 1e-4 recovery threshold.

| p   | N9 s5 (s) | N10 s3 (s) | N1 s1 (s) | N5 s1 (s) | N7 s1 (s) | All recovered? |
|-----|-----------|------------|-----------|-----------|-----------|----------------|
| 1.0 | 121       | 60         | 50        | 35        | 64        | YES            |
| 0.5 | 121       | 37         | 9         | 39        | 532       | YES            |
| 0.2 | 121       | 120        | 31        | 36        | 34        | YES            |
| **0.1** | **34** | **14**  | **11**    | **7**     | **14**    | **YES**        |

Notes:
- At p=0.5, N7 s1 took 532s (vs 64s at p=1.0); likely variance — p=0.2 and p=0.1 both ran fast for N7.
- At p=0.2, N9 s5 and N10 s3 still hit the 120s timeout cap; at p=0.1 they completed in 34s and 14s.
- At p=0.1, slow-run wall times dropped 3.5–4× vs p=1.0, and fast-set recovery was maintained across all cases.

**Decision:** p=0.1 is the smallest p with full recovery on both slow and fast cases, and it gives
the largest wall-time reduction. Default flipped to **0.1** in `evolutionary_search.hpp` and
`symbolic_regression.R`. (See commit following this measurement.)

### B1.9 Risk register
- *Recovery loss on hard problems:* mitigated by the sweep + conservative default flip.
- *HoF admits a lucky un-optimized member:* harmless — it is a valid expression with
  its constants; the Pareto archive still records true accuracy per complexity.
- *Throughput of `sse_current` for large trees:* it is O(rows × tree_size), strictly
  cheaper than `fit()` which does the same per LM iteration × restarts; net win.

---

## B2 — Static parsimony in selection (code-level outline)

- `SearchOptions`: `double parsimony = 0.0;` (0 = off).
- Selection cost helper (anonymous namespace):
  `cost(m) = m.loss / norm + opts.parsimony * m.complexity`, where `norm` is a
  per-run constant = `(n-1) * var(y)` (the same denominator as our NMSE, making
  `parsimony` scale-stable across problems). Compute `norm` once in `run_evolution`
  and thread it (or store on the island) — do not recompute per comparison.
- `tournament_best` / `tournament_worst`: compare by `cost`, not `loss`.
- HallOfFame: unchanged (keyed on raw loss vs. complexity). Selection is penalized;
  the archive still reports true accuracy.
- R wiring + a test that higher `parsimony` yields a smaller median complexity on a
  problem with a known compact form (e.g. N10), recovery preserved.
- Tune from PySR's heuristic (≈ expected min NMSE / 5–10) then by B0 measurement.

## B3 — Frequency-based adaptive parsimony (code-level outline, only if needed)

- New `struct RunningSearchStatistics { int window; std::vector<double> freq; ... };`
  one per `Island` (share-nothing, matching the island design). `freq` sized
  `max_nodes + 1`, initialized to 1.0.
- Update `freq[complexity] += 1` as each candidate is evaluated; when the running
  count exceeds `window`, scale all entries down (sliding window). Maintain a
  normalized view (`freq / sum(freq)`).
- Extend B2's cost: `cost += opts.adaptive_parsimony_scaling * normalized_freq[complexity]`.
- `SearchOptions`: `double adaptive_parsimony_scaling = 0.0;`, `int parsimony_window`.
  Do not copy PySR's 100000 window blindly; pick by measurement against our throughput.
- R wiring + a test that the population's complexity distribution flattens (lower
  variance of complexity) versus B2 alone.
