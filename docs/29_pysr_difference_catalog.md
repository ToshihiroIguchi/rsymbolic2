# 29. PySR difference catalog

Per CLAUDE.md "PySR Default Parity", rsymbolic2's **default settings and search
behaviour** must equal PySR's; **only the implementation method may differ**. `docs/28`
records the settings that are now matched. This file is the companion: an honest,
exhaustive catalog of where rsymbolic2 still **differs** from PySR — by allowed
implementation choice, by an unmatched/absent parameter, by a missing feature, or by a
subtle semantic difference that survives even when the nominal setting matches.

Source: PySR 1.5.10 (`pysr/sr.py`) over SymbolicRegression.jl 1.11.0
(`AdaptiveParsimony.jl`, `LossFunctions.jl`, `Population.jl`, `RegularizedEvolution.jl`,
`SingleIteration.jl`, `Migration.jl`, `MutationFunctions.jl`, `MutationWeights.jl`).
Re-derive from the installed source when versions change.

The categories are deliberately separated because they carry different obligations:
**A** and **D** are permitted (CLAUDE.md), **B** are parity gaps to close or justify,
**C** are feature gaps (all default-OFF in PySR, so no parity violation today).

## A. Implementation-method differences (same setting, different *how*) — permitted

These change *how* a result is computed, never *which* settings define the search, so
CLAUDE.md explicitly allows them.

| # | Aspect | rsymbolic2 | PySR / SR.jl |
|---|---|---|---|
| 1 | Constant optimiser | self Levenberg–Marquardt, **with multi-start now matched** (2026-06-25, `docs/33`): `optimizer_nrestarts=2` starts from `x0` plus 2 perturbed starts `x0·(1+0.5·N(0,1))`, keep best — faithful to SR.jl `_optimize_constants`. Only the inner solver (LM) differs | BFGS (Optim.jl); same multi-start (`optimizer_nrestarts`, `x0·(1+½·randn)`); different trust region, convergence, and iteration semantics |
| 2 | Numeric precision | Float64 throughout | `precision=32` (Float32 data + search) |
| 3 | Runtime / language | C++17, **no Julia** | Julia engine |
| 4 | Parallelism | OpenMP `parallel for` over islands, own team-size cap | Julia `:multithreading` / `:multiprocessing` |
| 5 | RNG | `std::mt19937_64` + own seeding scheme | Julia default RNG. **Identical settings still produce different concrete trajectories** |
| 6 | Migration mechanism | **now matched** (2026-06-22, `evolutionary_search.cpp::inject_migrants`): Poisson count (`std::poisson_distribution`, mean `pop*frac`), random destination slots, random migrant from the top-`topn` ring pool / HOF Pareto front, unconditional replace. Sole residual: ring topology is `i→(i+1)%n` rather than a globally gathered best-of-subpops pool (inert at the default fraction) | Poisson count `poisson_sample(pop*frac)`, random destination slots, random migrant from the candidate pool, unconditional replace |
| 7 | Evolution granularity | 1 generation = `population_size` mutate-and-replace steps | 1 cycle = `ceil(population_size / tournament_n)` mutations. Equated by total mutation count, not by label (`docs/28` §C) |
| 8 | Frequency-window decay | proportional down-scaling of all bins | `move_window!` subtracts from the largest bins. Differs only past `window_size = 1e5` evals |
| 9 | Constant-mutation kernel | **now matched** (2026-06-22, `mutation.cpp::mutate_constant`): multiplicative `c *= ±maxChange^U[0,1)` (or its reciprocal), `maxChange = perturbation_factor·temperature + 1.1`, temperature fixed at 1.0 (annealing off). Faithful to `mutate_factor`, including the `rand() > probability_negate_constant` sign-flip | multiplicative perturbation with temperature |
| 10 | Tree representation | postfix array (`vector<Node>`) | `Node` expression graph |
| 11 | Simplifier | **now matched** (2026-06-23, `simplify.cpp`): two passes, fold (`simplify_tree!`) then `combine_operators` — constant reassociation across same-operator nodes plus commutative canonicalisation (constant to the right). Identity elimination / double-negation / `x^2→square` were **removed** (SR.jl does none). Cadence also matched: per-iteration over the whole population (`evolutionary_search.cpp::simplify_population`, the simplify half of `optimize_and_simplify_population`) plus the weighted `simplify` mutation; the former per-child/per-init simplify was dropped | `simplify_tree!` + `combine_operators` |
| 12 | `pow` / `^` | safe_pow (returns 0 for non-positive base; avoids NaN/LM poisoning) | `^`; differs for non-positive base / non-integer exponent |
| 13 | Structural mutations (`add_node`/`insert_node`/`delete_node`/`rotate_tree`) | **now matched** (2026-06-23, `mutation.cpp`): audited 1:1 against `MutationFunctions.jl` 1.11.0 and `Mutate.jl` and six mechanism divergences fixed — (a) `add_node` is now 50% `append_random_op` + 50% **`prepend_random_op`** (root wrap), the latter previously absent; (b) `append_random_op` now **replaces** the grown leaf with fresh random leaves instead of reusing it; (c) the unary/binary growth split is `nbin/(nuna+nbin)` not a fixed 0.5; (d) `insert_random_op` always wraps the subtree as the **left** operand; (e) `rotate_tree` now includes **unary** rotation nodes via the leftmost/rightmost fallback (was binary-over-binary only); (f) `delete_random_op` uses SR.jl's parent-then-child selection with the leaf→random-leaf branch. Bit-identity is not expected (these change the trajectory by design; RNG stream differs anyway, #5); structural validity + thread determinism are retained (all standalone tests pass) | `MutationFunctions.jl` / `Mutate.jl` |
| 14 | Progress display (`verbosity`) | **default matched** (2026-06-28): the R/Python public-API default is `verbosity=1`, equal to PySR's installed `verbosity=1`/`progress=True`. The *setting* (progress on by default) is matched; only the *rendering* differs — one compact `stderr` line per epoch (`evolutionary_search.cpp`, `[epoch N t=Xs] best=… size med/max=… nconst med/max=…`) instead of PySR's live updating Pareto table. The C++ core default stays `0` (silent) so standalone benchmarks/tests are quiet; the wrappers always pass the value explicitly | `verbosity=1`, `progress=True` → live HOF table refreshed per iteration |

## B. Parameter settings not matched or not exposed (parity gaps)

To close or to justify as implementation, per CLAUDE.md.

| # | PySR parameter | Status in rsymbolic2 |
|---|---|---|
| 1 | `probability_negate_constant` = 0.00743 | **Implemented** (2026-06-22) — `mutate_constant` now flips the constant's sign with the SR.jl `rand() > probability_negate_constant` rule; `SearchOptions.probability_negate_constant = 0.00743` |
| 2 | `perturbation_factor` = 0.129 | **Mostly closed**: the constant-mutation kernel now consumes `perturbation_factor` (= `const_perturb_scale`) exactly as SR.jl does. rsymbolic2 still keeps a *separate* `optimizer_config.perturbation_scale` for optimiser restarts (also 0.129); PySR derives both from one value, so the fields are decoupled even though the defaults coincide |
| 3 | `optimizer_iterations` = 8 | Value matched, but it caps LM steps, not BFGS iterations (implementation; arguably not a true match) |
| 4 | `topn` = 12 | **Closed** (2026-06-22): `migration_size` now bounds the ring migrant pool from which migrants are sampled uniformly at random, matching PySR's "sample randomly from the topn pool" |
| 5 | `warmup_maxsize_by` = 0.0 (off) | **Closed** (2026-06-28, `docs/42`): the SR.jl `get_cur_maxsize` ramp is implemented — the per-epoch mutation/crossover size cap grows from 3 to `max_nodes` over the warmup fraction (`fraction_elapsed = done/generations`, the lockstep analog of SR.jl's per-population fraction), applied via a local `SearchSpace` inside `evolve_island`. The histogram and initial population stay at full `max_nodes` (as in SR.jl). Default 0.0 is byte-for-byte the fixed-maxsize path. Surfaced in C++/R/Python |
| 6 | `skip_mutation_failures` = True | **Closed** (2026-06-23): `mutate()` now returns whether a kind applied; on failure `evolve_island` skips the step (no replacement, no archive update) instead of falling back to a disruptive whole-tree randomization — matching SR.jl's "return the parent unchanged" no-op cycle. A non-finite child loss is likewise skipped (SR.jl's NaN-cost rejection) |
| 7 | `alpha` = 3.17 | PySR annealing temperature (inert at the default `annealing=False`); rsymbolic2 has no temperature concept at all |
| 8 | `use_frequency` = True | **Closed** (2026-06-23): SR.jl `next_generation` accepts a mutated child with probability `old_freq/new_freq` over the parent/child complexity frequencies (annealing factor off by default). rsymbolic2 previously implemented only `use_frequency_in_tournament` (the tournament-cost penalty) and omitted this acceptance test; it is now in `evolve_island` (`SearchOptions.use_frequency`, default true). The crossover path has no such test in SR.jl and likewise none here |
| 9 | Population replacement policy | **Closed** (2026-06-23): rsymbolic2 previously replaced the *tournament-worst* member and only if the child was better (greedy/elitist). SR.jl uses regularized evolution — the child replaces the **oldest** member unconditionally, selection pressure coming solely from the parent tournament. rsymbolic2 now matches: `PopMember.birth` is a per-island monotonic age stamp and `evolve_island` overwrites the smallest-birth member. The hall of fame independently retains the best-per-complexity, so the unconditional overwrite loses no quality (this was previously undocumented; it is the most consequential of the closed gaps) |

## C. PySR features absent in rsymbolic2 (all default-OFF in PySR — no parity violation today)

These are functional capabilities PySR ships but defaults off; rsymbolic2 lacks the
mechanism entirely. Listed so the gap is explicit, not to adopt speculatively
(CLAUDE.md: build what the current phase needs).

1. **Simulated annealing** (`annealing` + temperature schedule in `reg_evol_cycle`) —
   rsymbolic2's acceptance is always greedy.
2. **Batching** (`batching`, `batch_size`).
3. **turbo / bumper** (LoopVectorization-accelerated evaluation), **fast_cycle**.
4. **Denoising** (`denoise`, Gaussian-process smoothing of y).
5. **Feature selection** (`select_k_features`).
6. **Custom / weighted / non-L2 loss** (`elementwise_loss`, `loss_function`, `weights`).
7. **Dimensional analysis & units** (`X_units`, `y_units`,
   `dimensional_constraint_penalty`, `dimensionless_constants_only`) — **Implemented**
   (2026-07-04, `docs/46`) as an off-by-default opt-in feature with PySR-compatible API and
   semantics and a dependency-free C++ unit type. Still default-OFF, so no parity change.
8. **Per-operator complexity** (`complexity_of_operators`, `complexity_of_constants`,
   `complexity_of_variables`) — rsymbolic2 fixes every node at complexity 1.
9. **Constraints** (`constraints` on operator argument sizes, `nested_constraints` on
   operator nesting).
10. **Optimize-as-mutation** (`weight_optimize`; PySR keeps it at default 0.0) — omitted
    from rsymbolic2's mutation menu (constant optimisation is controlled by
    `optimize_probability` + the HOF re-optimisation pass instead; `docs/27`).
11. **Graph-mode mutations** (`form_connection` / `break_connection`) — tree only here.
12. **Parametric / template expressions and multi-output** (multiple y columns).
13. **`model_selection`** modes `"accuracy"` / `"score"` — rsymbolic2 implements only
    `"best"` (the accuracy/complexity knee).
14. **Arbitrary user operators** — rsymbolic2 has a fixed operator enum
    (`neg/exp/log/sin/cos/sqrt/tanh/abs/square`, `add/sub/mul/div/pow`); PySR accepts
    arbitrary Julia operators.
15. I/O & control surface: `early_stop_condition`, `max_evals`, `warm_start`, equation
    CSV persistence, richer progress logging. (rsymbolic2 has `timeout_seconds`.)

## D. Semantic micro-differences that survive a matched setting

Even where the nominal setting matches, the behaviour can differ subtly.

1. **Cost formula.** Both use `loss/baseline + parsimony·complexity`. rsymbolic2's
   `baseline` is the SSE of the mean predictor (floor 1.0); PySR's is the loss of the
   mean predictor under the chosen loss (floor 0.01). At `parsimony = 0` (the default)
   the tournament ranking is **identical** (baseline is a per-dataset constant). For
   `parsimony > 0` the linear-penalty scale differs.
2. **Adaptive-parsimony application.** rsymbolic2 multiplies the raw loss by
   `exp(scaling·freq)` (the `parsimony=0` cost path returns raw loss); PySR multiplies
   `loss/baseline`. Since `baseline` is constant across a tournament, the **ranking is
   identical**; only the absolute adjusted cost differs.
3. **Constant-optimisation cadence — now matched (2026-06-24, `docs/31`).** PySR optimises
   once per iteration in `optimize_and_simplify_population` (each population member with
   probability `optimize_probability`) plus the optional `weight_optimize` mutation.
   rsymbolic2 now does the same: `optimize_and_simplify_population` in
   `evolutionary_search.cpp` simplifies and (with probability `optimize_probability`, gated
   by `should_optimize_constants`) LM-optimises each population member once per epoch; the
   reg-evol cycle no longer optimises children and the initial population is scored by a
   forward pass only. The earlier per-child LM + per-epoch HOF re-optimisation (which
   invoked the optimiser ~28× more than PySR and accounted for 93 % of compute, `docs/30`)
   was removed.
4. **Matched mechanisms (for the record):** probabilistic tournament (sample n, rank by
   adjusted cost, pick rank r w.p. `p(1-p)^r`), the hall-of-fame (one best member per
   complexity level, 1..maxsize), and — since 2026-06-23 — the regularized-evolution
   replacement policy (parent tournament + replace-oldest) and the `use_frequency`
   mutation-acceptance test (B#8, B#9) are structurally equivalent.
5. **`condition_mutate_constant!` weight scaling** (2026-06-23): SR.jl scales the
   `mutate_constant` weight by `min(8, n_constants)/8` before sampling, so constant-sparse
   trees mutate constants less often. rsymbolic2 now applies the same factor in `mutate()`.

## E. The load-bearing implication

The category-A implementation differences — chiefly the LM-vs-BFGS optimiser cost,
Float64 vs Float32, the absence of turbo/batching, and per-generation overhead — mean
rsymbolic2's **throughput per unit compute is lower**. Therefore matching PySR's default
*settings* exactly does **not** reproduce PySR's *recovery* at an equal compute budget:
the search is throughput-bound (the fair 4-thread Feynman head-to-head times out on most
problems before completing the PySR-equivalent generation budget). Parity defines *what*
the search is; closing the recovery gap is a *throughput* problem in the implementation
layer, tracked separately (`docs/26` §5). This is the honest consequence of the
"same behaviour, our own implementation" split and must not be papered over.

A concrete instance (`docs/34`, 2026-06-27): with the operator set made strictly identical to
the PySR comparison (the gate's spurious 9th unary `neg` removed — A#13c's arity split is thus
the SR.jl 5/13, not 5/14), `interference`/`center_mass` remain far-misses **regardless of
mutations ④⑤⑥** — a controlled HEAD-vs-parent experiment exonerated that commit. The
authoritative neg-free gate is 18/25 (= PySR's baseline); the prior 19/25 was inflated by the
non-parity `neg`. Parity is kept over the higher score, per CLAUDE.md.
