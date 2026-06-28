# 31. Constant-optimisation cadence: aligning with PySR/SymbolicRegression.jl

## Problem

`docs/30` established that the Feynman search is throughput-limited and that **≈ 93 % of
all compute is constant fitting** (`evolve_fit` 72 % + `reopt_fit` 16 % + `init_fit` 4 %).
Investigation of the authoritative SymbolicRegression.jl source (v1.11.0,
`~/.julia/packages/SymbolicRegression/3nKj1/src/`) showed rsymbolic2 was invoking the LM
optimiser **far more often than PySR** — a parity divergence, not just an implementation
cost.

### What SR.jl actually does

- **The reg-evol mutation cycle never optimises constants.** `next_generation`
  (`Mutate.jl:161-343`) mutates a child and evaluates it with a single forward pass
  `eval_cost` (`:257`). BFGS is *never* called there. The `optimize` mutation type exists
  (`Mutate.jl:587-602`) but its weight `weight_optimize` is **0.0** by default, so it is
  never sampled.
- **Optimisation is confined to one per-iteration pass.**
  `optimize_and_simplify_population` (`SingleIteration.jl:68-92`) runs once per iteration:
  it simplifies every population member and, for the members selected by
  `do_optimization = rand(pop.n) .< optimizer_probability` (0.14), optimises constants —
  expected **0.14 × 27 ≈ 3.8 members per iteration**.
- **The initial population is not optimised.** `Population` (`Population.jl:35-61`) scores
  random members with `eval_cost` only; the first optimisation opportunity is the first
  end-of-iteration pass.
- **No structural deduplication / "skip if already optimised".** No hash or cache exists.
  The closest thing is the `baseline = f(x0)` guard in `_optimize_constants`
  (`ConstantOptimization.jl:87`): an already-converged structure re-enters BFGS but
  converges immediately and writes nothing back — re-optimisation is effectively a no-op,
  not a skip.

### What rsymbolic2 did (before this change)

- LM-optimised **every child** during the mutation step with probability
  `optimize_probability` (`evolve_island`).
- Re-optimised **every hall-of-fame member every epoch** at probability 1.0
  (`reoptimize_hof`, added in `docs/27`).
- LM-optimised **every initial member** (`initialize_island`).

### Magnitude

Per matched iteration (one epoch = `migration_interval` = 28 generations = 756 mutation
steps; epochs = 2800 / 28 = 100 = PySR `niterations`):

| | optimiser calls / iteration / island |
|---|---|
| SR.jl | 0 in the cycle + 27 × 0.14 ≈ **3.8** |
| rsymbolic2 (old) | 756 × 0.14 ≈ 106 + HOF re-opt (~15–30) ≈ **120–130** |

→ **≈ 28–33× more optimiser invocations** than PySR. This is the divergence behind the
93 %-constant-fitting profile.

### Verdict

The `optimize_probability` **value** (0.14) matched PySR, but its **application semantics**
(per-child + HOF, vs per-iteration over the population) did not. Per CLAUDE.md's PySR
Default Parity rule, optimisation cadence is *search behaviour*, not *implementation
method*, so this is a bug to fix — not a tunable trade-off.

## Change

In `r-package/rsymbolic2/src/evolutionary_search.cpp` (the single source of truth; the
standalone build references it directly):

1. **`evolve_island`** — the per-child branch `rand()<optimize_probability ? fit() :
   sse_current()` is replaced by an unconditional `sse_current()` (forward pass with
   inherited constants), matching `next_generation`'s `eval_cost`. The optimise-decision
   RNG draw is removed from the cycle.
2. **`optimize_and_simplify_population`** (renamed/extended from `simplify_population`) — a
   single per-epoch pass over the population: simplify each member, then with probability
   `optimize_probability` (gated by `should_optimize_constants`, and only when the member
   has constants) LM-optimise it; otherwise evaluate the simplified tree's SSE. Mirrors
   `SingleIteration.jl:79-92`, simplify-before-optimise. The optimise die is drawn from the
   island RNG per member.
3. **`reoptimize_hof` removed.** PySR has no whole-archive re-optimisation;
   `should_optimize_constants` now gates the per-population-member optimisation in (2),
   matching its PySR meaning.
4. **`initialize_island`** scores initial members with `sse_current` instead of `fit()`,
   matching `Population.jl`. (`sse_current` draws no island RNG, so the per-island RNG
   stream is unchanged from the old init.)

Determinism is preserved: each island is share-nothing with its own RNG; the per-member
optimise die is drawn from the island RNG, so the result is thread-count-independent
(`test_island_model` thread-1 ≡ thread-4 still passes).

## Verification

- **Standalone C++ tests** (`build-win`, Rtools/MinGW, Ninja, Release): all 19 pass,
  including `evolutionary_search` (linear/exp recovery), `island_model` (determinism),
  `simplify`.
- **R package testthat** (`NOT_CRAN=true`): `FAIL 0 | PASS 63` — including the
  `skip_on_cran` recovery, optimize-probability (incl. reproducibility), island, and
  parsimony suites.
- **Throughput** (`bench_profile rel_mass 20`, `OMP_NUM_THREADS=4`, cpu/wall 3.75):

  | phase | before (docs/30, % of compute) | after |
  |---|---|---|
  | constant fitting (LM) | **93 %** (evolve_fit + reopt_fit + init_fit) | **`popopt_fit` 10.7 %** (1 415 calls / 20 s) |
  | forward-pass child eval | `evolve_sse` (minority) | **`evolve_sse` 87.5 %** (321 515 calls / 20 s) |

  The optimiser is no longer the bottleneck; candidate evaluations per second rose roughly
  an order of magnitude (≈ 31 k → 321 k per ~20 s), exactly as the cadence analysis
  predicted.
- **Feynman recovery gate** (`02_feynman_gate.R stage=1 runs=1`, `OMP_NUM_THREADS=4`,
  300 s/problem, `feynman_gate_20260624.csv`): **18/25 recovered → PASS** (threshold 18/25).
  This is an **improvement, not a regression**:
  - prior 1-seed gate at the same config (`feynman_gate_20260621.csv`): **13/25** → +5;
  - old rsymbolic2 multi-seed baseline (docs/26 §5): **12/25**;
  - PySR (faithful, SR.jl engine) multi-seed headline (docs/26 §5): **18/25** — rsymbolic2
    now **matches PySR's recovery total**.

  Notably `rel_mass` — the transcendental-heavy target that hit the 300 s timeout at
  loss ≈ 0.41 in docs/30 — now recovers in 131 s, directly attributable to the ~10×
  throughput gain. The 7 remaining failures (interference, planck, center_mass,
  driven_osc, boltzmann_dist, bose_einstein, newtons_grav) are the known far-miss /
  exp-in-denominator hard cases (docs/26 §5), not new regressions.

  *Caveat:* this is a 1-seed run (noisy, "direction only" per docs/26); the full 5-seed
  `stage=1` (~8 h) is the higher-confidence confirmation and remains optional.

## Risk noted

`reoptimize_hof` was added in `docs/27` to push structurally-correct, constant-unpolished
candidates below the recovery threshold. Removing it could in principle lower recovery on
some problems; the hypothesis (consistent with PySR achieving recovery without it) is that
the ~10× throughput gain more than compensates, because surviving structures are polished
repeatedly by the per-iteration pass. The Feynman gate is the arbiter — if recovery
regresses, revisit (e.g. a bounded final-archive polish) rather than reverting the cadence.
