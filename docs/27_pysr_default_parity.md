# 27. PySR default-behaviour parity

> **Superseded in part by `docs/28` + CLAUDE.md "PySR Default Parity" (2026-06-21).**
> The policy is now: rsymbolic2's defaults must **equal** PySR's exactly (only the
> implementation differs), so each default is adopted because PySR ships it — *not*
> "only where it does not regress our recovery". Two conclusions below are reversed:
> (1) `adaptive_parsimony_scaling` is PySR's **1040** (ON), not "~20" and not "OFF";
> (2) several values stated here as already-PySR (mutation weights, `tournament_p=0.86`,
> `fraction_replaced_hof`) did **not** match the installed PySR and are corrected in
> `docs/28`. The mutation-operator/HOF/tournament *machinery* added by this change
> stands; the *default values* are governed by `docs/28`.

## Motivation

`docs/26` measured rsymbolic2 at **12/25** Feynman dev recoveries vs SymbolicRegression.jl
(PySR's engine) at **18/25**, with rsymbolic2's recovered set a strict subset of SR.jl's.
§5 there root-caused the gap as **search efficiency**, not compute: rsymbolic2 lacked the
high-weight PySR mutation operators and several default-on PySR behaviours. This change
implements those behaviours so a default-constructed search reproduces PySR's default
*mix*. PySR remains a reference, not a specification (CLAUDE.md): each default is adopted
only where it does not regress rsymbolic2's already-proven recovery.

Scope is **PySR's default-on behaviours**. Default-off features (annealing, batching,
denoising, feature selection, custom/weighted loss, dimensional analysis, parametric /
template / multi-output expressions, turbo/bumper, operator nesting constraints,
`warmup_maxsize_by`) are out of scope and remain future work.

## What was added

### Mutation operators + weighted menu (the docs/26 root cause)
- New postfix-tree operators in `evolution/mutation.{hpp,cpp}`:
  - `insert_random_op` — wrap an arbitrary subtree (not just a leaf) in a new operator.
  - `rotate_tree` — associativity-style rotation of a binary node with a binary child;
    node count is preserved, only the two operator nodes move.
  - `swap_operands` — swap a binary node's two operand subtrees.
- `evolution/mutation_weights.hpp` — `MutationWeights`, defaults = PySR's `MutationWeights`
  (high weight on `insert_node` 5.1 and `rotate_tree` 4.26). `mutate()` now samples among
  the kinds *feasible* for the tree in proportion to these weights, with `do_nothing` and
  `simplify` as menu entries. Wired through `SearchOptions.mutation_weights`.
- **Deviations from PySR's weight set** (both deliberate):
  - `optimize` mutation omitted — PySR ships it at weight 0.0, and rsymbolic2 already
    controls constant optimisation via `optimize_probability` (docs/13) plus the new
    periodic HOF pass below. A second control point would only duplicate it.
  - `form_connection` / `break_connection` omitted — graph-mode only, not the default
    tree representation.

### Probabilistic tournament
- `tournament_selection_p` (default 0.86, PySR). Parent selection picks the rank-`r` best
  of the sampled candidates with probability `p(1-p)^r`. `p = 1` reproduces the previous
  deterministic best-of-k tournament exactly (same RNG sequence).

### Periodic hall-of-fame constant optimisation — SUPERSEDED (2026-06-24, see docs/31)
- **This `reoptimize_hof` pass was removed.** It re-fitted every HOF member's constants
  every epoch at probability 1.0 — a mechanism PySR does not have. PySR's
  `should_optimize_constants` instead gates the per-population-member optimisation inside
  `optimize_and_simplify_population` (probability `optimize_probability`), which is what
  rsymbolic2 now implements. The whole-archive re-optimisation, combined with per-child LM
  in the evolution cycle, was invoking the optimiser ~28× more than PySR and accounted for
  93 % of compute (docs/30). See `docs/31` for the cadence fix and its throughput result.
  (Original rationale, retained for history: the per-epoch HOF polish targeted docs/26 §5
  "near-miss" failures where a structurally-correct candidate entered the archive with
  inherited constants.)

### Hall-of-fame migration
- `fraction_replaced_hof` (default 0.035, PySR). After each epoch the merged global elite
  archive is reinjected into every island population, replacing the worst members when the
  elite is better. Unlike ring migration it also runs for `n_populations == 1`. Deterministic.

### Recommended-equation selection
- `select_best()` in `evolution/hall_of_fame.{hpp,cpp}` implements PySR
  `model_selection="best"`: the Pareto-front member maximising the drop in log-loss per unit
  added complexity (the accuracy/complexity knee). Exposed in R as `$recommended` (the
  expression) and `$best_index` (1-based row of `pareto_front`); `$expression`/`$pareto_front`
  are unchanged (back-compatible).

## R surface
New `symbolic_regression()` arguments (PySR-faithful defaults): `tournament_selection_p`,
`should_optimize_constants`, `fraction_replaced_hof`, and `mutation_weights` (an optional
named numeric vector overriding individual weights; unknown names error). New result fields
`$recommended`, `$best_index`.

## Default-value decisions
- `tournament_selection_p = 0.86`, `should_optimize_constants = TRUE`,
  `fraction_replaced_hof = 0.035`, `mutation_weights` = PySR defaults — adopted as defaults;
  confirmed not to regress Nguyen recovery (below).
- `adaptive_parsimony_scaling`: **corrected to PySR's 1040 (ON)** — see `docs/28`. PySR
  runs the frequency penalty on at **1040** (verified in `sr.py`; the earlier "~20" here
  was wrong). Over a `maxsize`-binned sum-to-1 histogram, 1040 is well-behaved (a uniform
  histogram cancels in the ranking; the penalty acts on relative frequency differences).
  Per CLAUDE.md parity this is now ON at 1040, not a value to sweep or leave off.

## Measurement status

**Regression gate (bounded, this change):** Nguyen at the gate hyperparameters
(pop=500, islands=4, gens=200), representative spread N1/N5/N6/N9/N10, 2 seeds, 40 s cap:
**10/10 recovered**, including the historically slow multivariate N9/N10 (0–4 s). The
heavy structural-mutation weights, probabilistic tournament, HOF re-optimisation, and HOF
migration do not regress recovery.

**Unit / package tests:** 17/17 standalone C++ tests and 62/62 testthat tests pass on
Windows (R 4.6.0, Rtools45 / GCC 14). New coverage: `test_mutation_operators.cpp`
(insert/rotate/swap structural + semantic invariants, weighted-menu determinism),
`test_hall_of_fame.cpp` (`members`, `select_best`).

**Nguyen regression gate (full, run):** base **9/9** (45/45 runs) and sqrt **10/10**
(50/50) PASS at the gate hyperparameters — the new structural-mutation weights,
probabilistic tournament, HOF re-optimisation, and HOF migration do not regress recovery.

**Feynman dev 1-seed de-risk (run):** with the new operators at the existing 4×500
structure, **12/25** (same count as the docs/26 baseline) but `rel_mass` — previously an
SR.jl-only recovery — is now recovered. The dominant failure mode was timeout (11/13
failures hit 300 s): throughput-bound, not wrong-structure.

### Population structure: 4×500 is the wrong allocation (key finding)

The 4×500 benchmark structure (n_populations=4, population_size=500 → 2000 individuals)
diverges sharply from PySR's 31×27 (=837) and, under the 4-thread / 300 s budget, spends
the throughput-bound budget on breadth (wide population) instead of depth (generations).
Reproducing PySR's structure on rsymbolic2's side — **n_populations=31, population_size=27,
tournament_size=15** — over the same 4-thread budget gives, Feynman dev 1-seed:

| structure | recovered (1-seed) |
|-----------|--------------------|
| 4 × 500 (=2000) | 12/25 |
| **31 × 27 (=837, PySR)** | **14/25** |

Net +2, but the composition matters: 31×27 **gains hard problems**
(`harmonic_ke` and `larmor_rad` — both unrecovered by rsymbolic2 in docs/26;
`clausius_moss2`) while **losing near-misses** (`rel_mass` 2.5e-4,
`bohr_magneton` 2.4e-4 — both just above the 1e-4 threshold, the kind seed variance
recovers). Recoveries are also markedly faster (coulomb 300→42 s, bohr_radius 143→60 s),
confirming the budget now buys more generations. This is with adaptive parsimony **off**
and at **1 seed**. Conclusion: the wide-population benchmark setting was a poor allocation;
**align the benchmark structure to PySR's 31×27** and re-measure at full seeds. The
benchmark default in `02_feynman_gate.R` is now `n_populations=31, population_size=27,
tournament_size=15` (this change).

### adaptive_parsimony_scaling A/B at 31×27 (run, 1-seed) — decision REVERSED by parity

> **Note:** this A/B used `scaling=20` (the mistaken "PySR default") and kept
> `parsimony=1e-3` in both arms — i.e. it tested neither PySR's value (**1040**) nor
> PySR's config (`parsimony=0`). Its "keep OFF" conclusion is **void** under CLAUDE.md
> "PySR Default Parity": PySR ships the penalty **ON at 1040 with parsimony=0**, so
> rsymbolic2 matches that regardless of this 1-seed result. The data below is retained
> only as a record of the (non-faithful) `scaling=20`+`parsimony=1e-3` behaviour.

A/B on the 31×27 baseline, Feynman dev 1-seed, 300 s, 4-thread budget
(`02_feynman_gate.R stage=1 runs=1 scaling={0,20}`):

| arm | recovered (1-seed) |
|-----|--------------------|
| `scaling=0` (off) | **13/25** |
| `scaling=20` (PySR default) | **13/25** |

Identical **count**, but `scaling=20` **trades 2-for-2 and does not fix the collapse cases**:
- **gains** `rel_mass` (2.5e-4→2.2e-5) and `bohr_magneton` (2.4e-4→6.1e-32) — the two
  near-misses, where the extra diversity/polish pressure pushes them over the line;
- **loses** `coulomb` (✓54 s → 2.2e-1, timeout) and `harmonic_ke` (✓113 s → 2.0e-3, timeout)
  — both clean, fast recoveries at `scaling=0`, lost to the throughput cost of the penalty;
- the targeted **collapse cases** (`planck`, `center_mass`, `boltzmann_dist`, `bose_einstein`,
  `driven_osc`, `newtons_grav`, `lens_eq`) recover **none** under `scaling=20` (`bose_einstein`
  only improves 1.8e-2→1.7e-3). Throughput cost is visible everywhere (`gaussian` 29→199 s).

(For the record, this non-faithful arm traded `coulomb`/`harmonic_ke` for
`rel_mass`/`bohr_magneton` and fixed no collapse case — but that no longer drives the
decision.) **Under parity the default is PySR's: `adaptive_parsimony_scaling = 1040`,
`parsimony = 0`, ON.** The faithful config must be re-measured (§ Remaining, item 1).

**Remaining (heavy) measurement:**
1. Feynman dev **full 3-seed (then 5)** head-to-head at the **fully PySR-faithful config**
   (`docs/28`: 31×27, parsimony=0, scaling=1040, max_nodes=30, PySR weights, etc.) vs
   `05_feynman_pysr_comparison.jl` (SR.jl faithful, 18/25). Update docs/26 §5 and docs/05.
2. ~~Align the **shipped defaults** to PySR (`docs/28` §D), not just the benchmark.~~
   **Done:** C++ `SearchOptions`/`SearchSpace`/`mutation_weights.hpp`/optimiser config and
   the R `symbolic_regression()` defaults are now PySR-parity (maxsize-exact frequency
   histogram, fraction-based ring migration, cycle↔generation mapping resolved — `docs/28`
   §C/§D/§E). Standalone 17/17 and the full testthat suite pass.
3. ~~`adaptive_parsimony_scaling` A/B (decide ON/OFF)~~ — **moot under parity: PySR
   ships it ON at 1040, so rsymbolic2 matches (docs/28). No A/B decides this.**
4. Mixed-precision experiment (Float32 residual/Jacobian evaluation, Float64 LM solve):
   targets the measured throughput bottleneck without the normal-equations conditioning
   risk. Measure recovery + throughput before adopting.
