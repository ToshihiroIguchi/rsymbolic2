# 33. Constant-optimizer multi-start (SR.jl parity)

**Date:** 2026-06-25

## Problem

PySR/SR.jl optimise each member's constants with `optimizer_nrestarts = 2` (a PySR
default, `docs/28` §A): the optimiser runs from the current constants, then from
`optimizer_nrestarts` further perturbed starts, keeping the best. This escapes poor local
minima in constant space and is the mechanism behind PySR solving several near-miss Feynman
targets (constants reachable but the single basin doesn't converge).

rsymbolic2's default optimiser is `SelfLM`. The search passed `n_restarts = 2` in
`OptimizerConfig`, but **`SelfLMOptimizer::optimize` ignored it** — it ran a single LM from
x0. So the setting was "value-matched" while the *mechanism* was absent: a parity gap
(CLAUDE.md PySR Default Parity), not an allowed implementation difference. Additionally the
config carried `perturbation_scale = 0.129`, conflating it with `perturbation_factor`
(the `mutate_constant` kernel scale) — SR.jl's restart scale is a separate hardcoded value.

## Authoritative SR.jl mechanism

`~/.julia/packages/SymbolicRegression/3nKj1/src/ConstantOptimization.jl::_optimize_constants`:

1. `baseline = f(x0)` (loss at the current constants).
2. Optimise from `x0` (start 0).
3. `for _ in 1:optimizer_nrestarts`: `xt = x0 .* (1 + (1//2)·randn)` — **multiplicative
   Gaussian perturbation, fixed 0.5 scale** (NOT `perturbation_factor`); optimise from `xt`;
   keep if `minimum < result.minimum`.
4. Accept the best iff `< baseline`, else restore `x0`.

The `0.5` is a mechanism constant (not a PySR-exposed setting); `optimizer_nrestarts = 2` →
1 + 2 = 3 starts total.

## Implementation

- `self_lm_optimizer.{hpp,cpp}`: the LM loop is factored into `run_lm_from(problem, x0, …)`.
  `optimize()` runs start 0 from `problem.initial_constants`, then `config.n_restarts`
  perturbed starts `xt[i] = x0[i]·(1 + perturbation_scale·N(0,1))`, keeping the best.
  A `mutable std::mt19937_64 rng_` (seeded from `config.seed`) drives only the perturbation;
  the LM loop stays RNG-free. `n_restarts = 0` reduces to the previous single-LM behaviour.
  Because start 0 is monotone from x0, the returned best is always ≤ f(x0), so SR.jl's
  "restore x0 unless improved" baseline guard holds automatically (no member is worsened);
  the population pass write-back is unchanged.
- `evolutionary_search.hpp`: `optimizer_config` restart scale corrected `0.129 → 0.5`
  (SR.jl's `T(1//2)`); `perturbation_factor = 0.129` stays in `const_perturb_scale`
  (`mutate_constant`), now clearly distinct.
- `evolutionary_search.cpp`: each island's optimiser is seeded deterministically per island
  (idx-only, distinct salt from the island/migration RNGs), so restarts are reproducible and
  thread-count independent (`test_island_model` thread-1 ≡ thread-4 still holds).

This is allowed to differ from PySR only in the inner solver (self-LM vs BFGS, CLAUDE.md
A#1); the *multi-start mechanism and its settings* now match.

## Verification

- Standalone C++: 19/19 tests pass on **both** Windows (Rtools/MinGW) and Ubuntu (WSL),
  including new `test_self_lm_optimizer` cases (multi-start escapes a constructed double-well
  local minimum; `n_restarts=0` backward-compat; cross-instance determinism) and
  `test_island_model` thread determinism.
- R testthat (Windows): FAIL 0, PASS 23, SKIP 24 (CRAN).
- Bounded near-miss sanity (gate config, seed 1, 150 s budget): **`rel_mass` now RECOVERED
  (NMSE 7.5e-12) in 72 s**, where single-start timed out at 300 s (median NMSE 2.7e-4,
  `docs/26` §5). `lens_eq` still misses (1.1e-3) at the reduced budget — no regression, but a
  harder near-miss. This is directional (1 seed); the authoritative measure is the multi-seed
  Feynman gate (`benchmarks/02_feynman_gate.R stage=1`), a milestone run.

## Cost / parity note

Multi-start triples the LM calls in the once-per-iteration population pass (gated by
`optimize_probability = 0.14`, so not the whole population). This is exactly PySR's cost — it
is parity-faithful, not a regression. The rel_mass result shows the recovery gain outweighs
the per-fit throughput cost on at least one timed-out near-miss.

See also `docs/28` §A (optimizer settings), `docs/29` A#1 (optimiser implementation
difference), `docs/30`/`docs/31` (throughput / cadence).
