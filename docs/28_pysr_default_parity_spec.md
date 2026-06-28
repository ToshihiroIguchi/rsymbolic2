# 28. PySR default-parity specification (authoritative)

> **Attribution.** PySR and SymbolicRegression.jl are licensed under the Apache
> License 2.0 (© Miles Cranmer). rsymbolic2 is an independent re-implementation whose
> defaults and mechanisms are matched to theirs; this and the other `docs/` files
> describe that matching. Legal attribution is carried in the repository `NOTICE`
> file, not in these notes. rsymbolic2 is not affiliated with those projects.

Per CLAUDE.md ("PySR Default Parity"), rsymbolic2's **default configuration and search
behaviour must equal PySR's documented defaults; only the implementation method may
differ.** This file is the authoritative reference for *what those defaults are* and
*how the mechanisms work*, so the C++/R defaults can be aligned and kept aligned.

Read defaults from the installed source, never from memory or prose. This document
records a snapshot; re-derive it from the installed source when versions change.

For the complementary catalog of where rsymbolic2 still **differs** from PySR (allowed
implementation choices, unmatched/absent parameters, missing features, and semantic
micro-differences), see `docs/29`.

## Source (this dev environment)

- **PySR** 1.5.10 — `PySRRegressor.__init__` in
  `…/site-packages/pysr/sr.py`. PySR is a thin scikit-learn wrapper; the `__init__`
  defaults are what `PySRRegressor()` uses, and `sr.py` assembles them into a
  `SymbolicRegression.MutationWeights(...)` and `SymbolicRegression.Options(...)`
  (`sr.py` ~L1962, ~L2018) passed to the Julia engine.
- **SymbolicRegression.jl** 1.11.0 — `~/.julia/packages/SymbolicRegression/3nKj1/src/`.
  The mechanisms (`AdaptiveParsimony.jl`, `LossFunctions.jl::loss_to_cost`,
  `Population.jl::_best_of_sample`, `MutationWeights.jl`) define the behaviour the
  defaults drive.

## A. PySR `__init__` defaults (sr.py, verbatim)

| PySR parameter | default | rsymbolic2 mapping | class |
|---|---|---|---|
| `niterations` | 100 | epochs (× ncycles) | setting |
| `populations` | 31 | `n_populations` | setting |
| `population_size` | 27 | `population_size` | setting |
| `ncycles_per_iteration` | 380 | per-epoch mutation cycles (see §C) | setting |
| `maxsize` | 30 | `max_nodes` | setting |
| `maxdepth` | None → `maxsize` (30) | `max_depth` | setting |
| `warmup_maxsize_by` | None → 0.0 | (size warmup; off) | setting |
| `parsimony` | 0.0 | `parsimony` | setting |
| `use_frequency` | True | (freq histogram on) | setting |
| `use_frequency_in_tournament` | True | adaptive parsimony on | setting |
| `adaptive_parsimony_scaling` | **1040.0** | `adaptive_parsimony_scaling` | setting |
| `alpha` | 3.17 | (annealing temp; annealing off) | setting (inert) |
| `annealing` | False | (no annealing) | setting |
| `fraction_replaced` | 0.00036 | ring-migration fraction | setting |
| `fraction_replaced_hof` | 0.0614 | `fraction_replaced_hof` | setting |
| `weight_add_node` | 2.47 | `mutation_weights.add_node` | setting |
| `weight_insert_node` | 0.0112 | `mutation_weights.insert_node` | setting |
| `weight_delete_node` | 0.870 | `mutation_weights.delete_node` | setting |
| `weight_do_nothing` | 0.273 | `mutation_weights.do_nothing` | setting |
| `weight_mutate_constant` | 0.0346 | `mutation_weights.mutate_constant` | setting |
| `weight_mutate_operator` | 0.293 | `mutation_weights.mutate_operator` | setting |
| `weight_swap_operands` | 0.198 | `mutation_weights.swap_operands` | setting |
| `weight_rotate_tree` | 4.26 | `mutation_weights.rotate_tree` | setting |
| `weight_randomize` | 0.000502 | `mutation_weights.randomize` | setting |
| `weight_simplify` | 0.00209 | `mutation_weights.simplify` | setting |
| `weight_optimize` | 0.0 | (optimize-as-mutation; off) | setting |
| `crossover_probability` | 0.0259 | `crossover_probability` | setting |
| `migration` | True | ring migration on | setting |
| `hof_migration` | True | HOF migration on | setting |
| `topn` | 12 | migration top-n | setting |
| `should_simplify` | True | `simplify` | setting |
| `should_optimize_constants` | True | `should_optimize_constants` | setting |
| `optimizer_nrestarts` | 2 | optimiser restarts | setting |
| `optimize_probability` | 0.14 | `optimize_probability` | setting |
| `optimizer_iterations` | 8 | optimiser max iterations | setting |
| `perturbation_factor` | 0.129 | `const_perturb_scale` | setting |
| `probability_negate_constant` | 0.00743 | constant-negate prob | setting |
| `tournament_selection_n` | 15 | `tournament_size` | setting |
| `tournament_selection_p` | **0.982** | `tournament_selection_p` | setting |
| `skip_mutation_failures` | True | retry/discard failed mutation | setting |
| `optimizer_algorithm` | "BFGS" | self-LM | **implementation** |
| `optimizer_f_calls_limit` | None | (BFGS-specific) | implementation |
| `precision` | 32 (Float32) | Float64 core | **implementation** |
| `parallelism` | (multithreading in bench) | OpenMP islands | implementation |
| `batching` | False | implemented, default off (B5) | setting (off) |
| `batch_size` | 50 | `batch_size` (used iff batching) | setting |
| `turbo`,`bumper`,`fast_cycle` | False | off | setting (off) |

The mutation weights are **normalised to sum to 1** before sampling
(`MutationWeights.jl`, `sample_mutation`), so only the *ratios* matter. Note these
sr.py values **override** SymbolicRegression.jl's own `MutationWeights` struct defaults
(which are different, e.g. struct `insert_node=2.44`); `PySRRegressor()` uses the sr.py
values, so those are "PySR's defaults".

## B. SymbolicRegression.jl mechanisms the defaults drive

### B1. Frequency-adaptive parsimony (`AdaptiveParsimony.jl`, `Population.jl`)
- `frequencies`: a histogram over complexity, length **`maxsize`** (= 30), init all 1.0.
- `normalized_frequencies = frequencies ./ sum(frequencies)` → **sums to 1**, uniform
  baseline `1/maxsize ≈ 0.033`. Refreshed once per cycle; raw counts accumulate, with a
  sliding `window_size = 100000` (`move_window!` subtracts from the largest bins).
- Tournament cost (`_best_of_sample`), when `use_frequency_in_tournament`:
  ```
  frequency        = (0 < size <= maxsize) ? normalized_frequencies[size] : 0
  adjusted_cost[i] = member.cost * exp(adaptive_parsimony_scaling * frequency)
  ```
- **Why 1040 does not explode:** when all sizes are equally frequent every member gets
  the *same* `exp(1040·0.033)` factor, which cancels in the ranking. The penalty acts on
  **relative** frequency differences, exponentially pushing the population toward a
  uniform spread over complexities. (Overflow only if one size holds >`709/1040≈0.68`
  of the mass; SR.jl clamps nothing — it relies on the dynamics staying below that.)
- **rsymbolic2 divergence to fix:** the histogram is currently sized `max_nodes+2` (=52),
  not `maxsize` (=30); with `max_nodes=30` and a `maxsize`-exact size (drop the `+2`,
  frequency 0 outside `[1,maxsize]`), the normalisation and the 1040 value transfer
  exactly. (The earlier docs/24–27 claim that PySR's value is "~20" and that our scale
  "differs" was **wrong** — it is 1040 over a sum-to-1 / maxsize histogram.)

### B2. Cost = normalised loss + linear parsimony (`LossFunctions.jl::loss_to_cost`)
```
normalization = (baseline >= 0.01 && use_baseline) ? baseline : 0.01
cost          = loss / normalization + complexity * parsimony
```
`baseline` is the loss of the mean predictor (≈ var(y) for L2), so `loss/baseline` is an
NMSE-like ratio. rsymbolic2's `loss/y_norm + parsimony*complexity` (with
`y_norm = SSE_baseline`) is the same ratio — structurally matched. With PySR
`parsimony=0`, cost is the normalised loss alone.

### B3. Tournament (`_best_of_sample`)
Sample `tournament_selection_n` (15) members, rank by `adjusted_cost`, pick rank `r`
with probability `p(1-p)^r`, `p = tournament_selection_p` (**0.982**). rsymbolic2's
`tournament_best` already implements this; only the default `p` and `n` must change.

### B4. Migration
- Ring migration every iteration (`fraction_replaced = 0.00036`, `topn = 12`).
- HOF migration (`fraction_replaced_hof = 0.0614`, `hof_migration = True`).

### B5. Batching (`SingleIteration.jl`, `Dataset.jl::batch`, default off)
Implemented as an opt-in feature; the default stays `batching = False` (parity). When on,
SR.jl draws a fresh `SubDataset` of `batch_size` row indices **with replacement**
(`rand(rng, 1:n, batch_size)`) **once per iteration** for the reg-evol cycle, and a
**second** batch for the per-iteration constant-optimisation pass. Crucially the global
hall of fame only ever receives **full-dataset** costs: `finalize_costs` recomputes the
whole population on the full data and the per-iteration `best_seen` archive is recomputed
on the full data (`SymbolicRegression.jl:1129`) before either is merged. `loss_to_cost`
reads the **full** dataset's `baseline_loss` even for a `SubDataset`, and the loss itself is
a per-point **mean** (`normalize=true`), so the batched cost is
`mean_loss_batch / full_baseline`.

rsymbolic2 mapping (one epoch ↔ one PySR iteration, §C): per epoch, per island, draw a reg
batch and an opt batch from a dedicated per-island batch RNG (with replacement); evolve and
optimise on them, accumulating an epoch-local `best_seen` hall of fame instead of touching
`isl.hof`; then recompute the population and `best_seen` on the full dataset and merge into
`isl.hof` (`finalize_costs_and_merge`). The full-data path (`batching = false`) is byte-for-
byte unchanged. Because rsymbolic2's loss is an **SSE** (sum) not a mean, the batch is built
with each sampled point's weight rescaled by `n_full / batch_size`; paired with the full-data
`y_norm`, the batched selection cost equals `(SSE_batch / batch_size) / (SST_full / n_full)`,
which is algebraically identical to SR.jl's `mean_loss_batch / full_baseline`. Rows are
sampled with replacement; the only divergence from SR.jl is the RNG stream (an allowed
implementation difference, CLAUDE.md). `batch_size` is clamped to `n_full` (PySR caps it at
`len(X)`).

## C. Cycles vs generations mapping (resolved)

PySR runs `niterations = 100` iterations, each `ncycles_per_iteration = 380` reg-evol
cycles per population, migrating between iterations. The exact per-iteration mutation
count comes from `RegularizedEvolution.jl`:
`n_evol_cycles = ceil(population_size / tournament_selection_n)` mutations **per** cycle,
so one iteration = `ncycles_per_iteration * ceil(population_size / tournament_selection_n)`
mutations per population. For the defaults: `380 * ceil(27/15) = 380 * 2 = 760`
mutations/pop/iteration.

rsymbolic2 counts `generations`, each `= population_size` tournament-and-replace steps
(`= population_size` mutations), and migrates every `migration_interval` generations.
Equating the per-epoch mutation budget and the migration cadence:

- mutations per PySR iteration per pop = 760 → `migration_interval = round(760 / 27) = 28`
  generations (756 mutations/epoch, within 0.5 % of 760).
- one epoch ↔ one PySR iteration → total `generations = niterations * migration_interval
  = 100 * 28 = 2800`.

**Shipped defaults (implemented):** `generations = 2800`, `migration_interval = 28`.
This reproduces PySR's per-population mutation budget (≈ 75 600 vs 76 000 mutations/pop)
and migration cadence (100 migrations). The derivation depends on `population_size = 27`
and `tournament_size = 15`; if a caller overrides those, the generation/interval defaults
no longer track PySR exactly (documented in the `generations` roxygen). Benchmarks may
still pin the structure explicitly. The rsymbolic2 "generation" and the SR.jl "cycle" are
deliberately different granularities (population_size vs `ceil(pop/tournament_n)`
mutations); the mapping equates total mutations, not the labels.

## D. rsymbolic2 default alignment (implemented)

Shipped R defaults (`symbolic_regression()`) and C++ `SearchOptions`, old → **PySR**
(all applied; see the roxygen / struct comments and the builds in §E):

| field | old | **PySR (now default)** |
|---|---|---|
| `population_size` | 200 | **27** |
| `n_populations` | 1 | **31** |
| `generations` / `migration_interval` | 50 / 10 | **2800 / 28** (§C) |
| `tournament_size` | 4 | **15** |
| `tournament_selection_p` | 0.86 | **0.982** |
| `max_nodes` | 40 | **30** |
| `max_depth` | 4 | **30** (= maxsize) |
| `parsimony` | 1e-3 | **0.0** |
| `adaptive_parsimony_scaling` | 0 | **1040.0** |
| `optimize_probability` | 0.1 | **0.14** |
| `crossover_probability` | 0.5 | **0.0259** |
| `fraction_replaced_hof` | 0.035 | **0.0614** |
| `should_optimize_constants` | TRUE | True ✓ (unchanged) |
| `mutation_weights` (hpp) | insert_node=5.1, swap=0.00608, … | **sr.py set (§A)** |

C++ `SearchOptions`/core fields aligned (not all surfaced in R): histogram sized
**`maxsize`-exact** with frequency 0 outside `[1, maxsize]` (was `max_nodes+2`),
`parsimony_window=100000` ✓, **ring-migration fraction `fraction_replaced=0.00036`** with
`topn`/`migration_size=12` (injected count = `round(fraction_replaced*pop)`; 0 at the
default → inert, matching PySR), **`const_perturb_scale=0.129`**, optimiser
`optimizer_config={n_restarts=2, max_iterations=8, perturbation_scale=0.129}` (iteration
cap is implementation: self-LM, not BFGS), `ncycles_per_iteration` mapping resolved in §C.
The default operator set is a problem input, not matched here. **Ported 2026-06-22:**
`probability_negate_constant=0.00743` (now a `SearchOptions` field) together with the
multiplicative `mutate_factor` constant-mutation kernel and the Poisson/random/unconditional
migration mechanism (`docs/29` A#6, A#9, B#1, B#2, B#4).

## E. Status

CLAUDE.md policy recorded; defaults derived from installed PySR 1.5.10 / SR.jl 1.11.0.
**Code alignment implemented** (C++ `SearchOptions` + `SearchSpace` + `mutation_weights.hpp`
+ `OptimizerConfig` member + maxsize-exact histogram + fraction-based ring migration; R
`symbolic_regression()` defaults + roxygen; §C cycle mapping). **Verified:** standalone
rebuild + 17/17 C++ tests pass; R package rebuild + full testthat suite passes (the
adaptive-parsimony default-path test was updated for the new parity defaults). Remaining:
a fresh PySR-faithful Feynman re-measurement vs SR.jl (task #4). The prior
adaptive-parsimony "default OFF" decision (docs/24, docs/27) is **superseded** by this
parity rule: PySR ships it ON at 1040, so rsymbolic2 now does too.

**Search-behaviour gaps closed (2026-06-23).** Four PySR-default search behaviours that
were absent or wrong (the mutation/parsimony settings were already matched; these are the
*mechanisms* the settings drive — see `docs/29` A#9, B#6/#8/#9, D#5):
- **Regularized-evolution replacement.** Child replaces the OLDEST population member
  unconditionally (SR.jl `reg_evol_cycle`), not the tournament-worst-if-better; selection
  pressure is the parent tournament alone (`PopMember.birth`, `evolve_island`).
- **`use_frequency` acceptance test.** A mutated child is accepted with probability
  `old_freq/new_freq` over parent/child complexity frequencies (SR.jl `next_generation`),
  separate from the tournament-cost penalty (`SearchOptions.use_frequency`, default true).
- **`condition_mutate_constant!`** weight scaling `min(8,n_const)/8` in `mutate()`.
- **`skip_mutation_failures`**: a mutation that cannot apply (or yields a non-finite loss)
  is skipped as a no-op cycle, not replaced by a randomized tree.
Verified: 19/19 standalone C++ tests pass; full R testthat suite passes; bounded Feynman
sanity check recovers (e.g. gaussian NMSE≈3e-12). **Windows correctness fix shipped
alongside:** `sse_current`'s scratch was a function-local `static thread_local`, whose
per-thread semantics are unreliable for libgomp worker threads inside a loaded DLL on
Windows/MinGW — two islands then shared one buffer and raced, corrupting the heap (R
fast-fail 0xC0000409 under multi-island). The scratch is now owned by the `Island`
(inherently per-worker, reused per call), removing the data race without per-call churn.
