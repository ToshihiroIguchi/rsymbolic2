// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rsymbolic/evolution/hall_of_fame.hpp"
#include "rsymbolic/evolution/mutation_weights.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

namespace rsymbolic {

// A read-only snapshot of the global Pareto front, offered to an optional progress
// callback once per epoch (docs/53). Pure observation data: nothing in the search
// reads it back, so attaching a callback cannot influence the run.
struct ProgressSnapshot {
    std::size_t epoch = 0;          // completed outer iterations
    std::vector<int> complexity;    // current global pareto front
    std::vector<double> loss;
};

// Configuration for the (minimal, steady-state) evolutionary search.
struct SearchOptions {
    SearchSpace space;
    // Defaults mirror PySR's installed __init__ defaults (docs/28 §A): populations=31,
    // population_size=27, tournament_selection_n=15. `generations` is rsymbolic2's unit
    // (population_size mutation steps each); 2800 = niterations(100) x 28, the per-epoch
    // generation count that reproduces PySR's per-iteration mutation budget — see the
    // `generations`/`migration_interval` derivation in the cycle-mapping note below and
    // docs/28 §C.
    std::size_t population_size = 27;
    std::size_t generations = 2800;
    std::size_t tournament_size = 15;
    // Probabilistic tournament (PySR tournament_selection_p). In each parent selection,
    // the rank-r best of the `tournament_size` sampled members is chosen with
    // probability p*(1-p)^r. p = 1 reproduces the deterministic best-of-k tournament
    // (identical RNG sequence). Default 0.982 = PySR installed default (docs/28 §A/B3).
    double tournament_selection_p = 0.982;
    // SelfLM is the default and only least-squares backend: it matched the former
    // Eigen-based backend's recovery rate problem-for-problem while being ~7-8x faster
    // per fit and, crucially, performing zero large per-fit heap allocations — which
    // restores multi-island scaling (4-thread cpu/wall 1.4 -> 2.9 on the heap probe).
    // The Eigen backend has since been removed, dropping the library's last third-party
    // C++ dependency. See docs/25, docs/40.
    OptimizerType optimizer_type = OptimizerType::SelfLM;
    // PySR optimiser settings (docs/28 §A): optimizer_nrestarts=2 — now actually consumed
    // by SelfLMOptimizer's multi-start loop (start 0 + 2 perturbed restarts, keep best),
    // matching SR.jl _optimize_constants. The iteration cap (8) is implementation: rsymbolic2
    // uses self-LM (PySR uses BFGS, CLAUDE.md), and LM converges in far fewer iterations per
    // restart, so 8 mirrors PySR's per-fit compute budget. The restart perturbation scale is
    // 0.5, matching SR.jl's hardcoded T(1//2) restart scale (xt = x0*(1 + 0.5*randn)); this is
    // a mechanism constant and is DISTINCT from PySR's perturbation_factor=0.129, which is the
    // mutate_constant kernel scale carried separately in const_perturb_scale below. The seed is
    // set per island in run_evolution so each island's restart stream is deterministic and
    // thread-count independent. Field order: {seed, n_restarts, max_iterations, perturbation_scale}.
    OptimizerConfig optimizer_config{0, 2, 8, 0.5};
    std::uint64_t seed = 0;
    double target_loss = 1e-10;         // early stop once the best loss is below this
    // PySR `early_stop_condition` (float form): an additional early-stop loss threshold,
    // off by default (0.0). When > 0 the search also stops once the best loss falls below
    // it, so the effective threshold is max(target_loss, early_stop_condition) — a value
    // larger than target_loss stops the run sooner, at a looser fit. The Julia-function
    // form (loss+complexity predicate) is intentionally not supported (docs/29 §C);
    // only the numeric threshold is.
    double early_stop_condition = 0.0;
    // PySR `max_evals`: cap the total number of candidate evaluations (forward-pass loss
    // evals + the residual evaluations consumed by constant-optimisation fits), summed
    // across islands. 0 = no limit (default). Enforced via a per-island fair share
    // (max_evals / n_populations) checked at generation boundaries plus a global-sum check
    // at each epoch boundary, so it is deterministic and thread-count independent (unlike
    // timeout_seconds) and adds zero hot-path synchronisation. It is a coarse budget cap:
    // a fit counts its internal residual evaluations, a forward pass counts as one, and the
    // cap may overshoot by up to one generation's worth of evals per island.
    std::size_t max_evals = 0;
    // PySR `model_selection`: which member of the Pareto front is reported as the
    // recommendation (best_index). Default Best matches PySR. See select_best.
    ModelSelection model_selection = ModelSelection::Best;
    // PySR `weights`: optional per-point weights for a weighted least-squares fit. Empty =
    // unweighted (default). When non-empty (size must equal y.size()), the loss is the
    // weighted SSE sum_i w_i (pred_i - y_i)^2 and the selection denominator y_norm uses the
    // weighted variance, so the loss/y_norm ratio is invariant to overall weight scaling.
    std::vector<double> weights;
    // Constant-perturbation scale for the mutate_constant mutation = PySR
    // perturbation_factor (docs/28 §A), a setting that must match.
    double const_perturb_scale = 0.129;
    // Probability that mutate_constant flips a constant's sign = PySR
    // probability_negate_constant (docs/28 §A; sr.py default 0.00743). The sign-flip
    // semantics are faithful to SymbolicRegression.jl (see mutation.cpp).
    double probability_negate_constant = 0.00743;
    bool simplify_expressions = true;   // algebraically simplify fitted candidates
    // Fraction of evolution steps that perform subtree crossover instead of mutation.
    // PySR crossover_probability default 0.0259 (docs/28 §A). 0.0 = mutation only.
    double crossover_probability = 0.0259;

    // Relative weights of the mutation kinds, applied when a step mutates rather than
    // crosses over. Defaults reproduce PySR's MutationWeights mix (high weight on the
    // structural insert_node / rotate_tree moves). See mutation_weights.hpp / docs/27.
    MutationWeights mutation_weights{};

    // Island-model parallelism. n_populations=1 reproduces the single-population
    // path exactly (island 0 uses seed directly; RNG sequence is identical).
    // Default 31 = PySR `populations` (docs/28 §A).
    std::size_t n_populations      = 31;  // number of parallel islands
    // OpenMP team size for the island-parallel regions (init + evolution). 0 = auto:
    // use the OpenMP default (omp_get_max_threads(), which honours OMP_NUM_THREADS and
    // otherwise defaults to the machine's core count) — this is the default and reproduces
    // the prior behaviour exactly. A positive value requests exactly that many island
    // workers. The team is always capped at n_populations because islands are the only unit
    // of parallelism (see resolve_team_size). Pure wall-clock knob: the island model is
    // bit-deterministic across thread counts (test_island_model), so n_threads changes only
    // speed, never the result (docs/37).
    int n_threads = 0;
    // Cycle mapping (docs/28 §C): PySR runs niterations=100 iterations, each
    // ncycles_per_iteration=380 reg-evol cycles, with each cycle doing
    // ceil(population_size / tournament_selection_n) mutations and migrating once per
    // iteration. One rsymbolic2 generation = population_size mutations, so one PySR
    // iteration = 380*ceil(27/15) = 760 mutations/pop = round(760/27) = 28 generations,
    // and the run is niterations*28 = 2800 generations with migration every 28.
    std::size_t migration_interval = 28;  // evolve this many generations between migrations
    // Source pool offered per ring hop (PySR `topn`). The number actually injected is
    // derived from `fraction_replaced` at the call site, so under PySR defaults
    // (round(0.00036*27)=0) ring migration is effectively inert, matching PySR.
    std::size_t migration_size     = 12;  // top-n pool sent to the next island (PySR topn)

    // Ring-migration fraction (PySR `fraction_replaced`, docs/28 §A/B4). Each epoch the
    // ring injects round(fraction_replaced * population_size) of the source island's best
    // members into the next island (replace-worst-if-better). At the PySR default
    // (0.00036) this rounds to 0 for population_size=27, so ring migration is inert by
    // default exactly as in PySR; HOF migration below carries the cross-island flow.
    double fraction_replaced       = 0.00036;

    // Hall-of-fame migration (PySR fraction_replaced_hof): after each epoch, reinject
    // the global elite archive's members into every island's population, replacing the
    // worst members when the elite is better. The injected count is
    // round(fraction_replaced_hof * population_size). Unlike ring migration this also
    // runs for n_populations == 1, reintroducing elites that the working population may
    // have since discarded. Deterministic (no RNG). 0 = off. Default 0.0614 = PySR
    // installed default (docs/28 §A).
    double fraction_replaced_hof   = 0.0614;

    // Selection cost = loss / y_norm + parsimony * complexity, where
    // y_norm = sum((y - mean(y))^2). Borrowed from PySR's loss_to_cost: dividing
    // by y_norm makes the penalty scale-stable across problems of different y-variance
    // (equivalent to weighting by NMSE rather than raw SSE). 0 = off.
    // Default 1e-3 chosen from a sweep over {0, 1e-4, 5e-4, 1e-3, 5e-3} on
    // Nguyen N9/N10/N1/N5/N7 (pop=500, n_pops=4, gen=200, timeout=120s):
    // all cases recovered at every p; p=1e-3 gave ~10× speedup on N9 s5
    // (123s→12s) and 6× on N10 s3 (48s→8s) with no fast-set regression
    // (see docs/13, B2.8). PySR default is 0.0 (docs/28 §A): the frequency-adaptive
    // term below carries the size pressure, so the fixed linear penalty is off.
    double parsimony = 0.0;

    // Opt-in dimensional analysis (PySR `dimensional_constraint_penalty`; docs/46). The
    // flat penalty added to a violating expression's loss. 0 disables the penalty; the
    // units themselves live on `space` (x_units/y_units). When no units are declared this
    // is inert and the search is byte-identical to the units-off PySR-parity default. The
    // PySR-effective default (1000.0) is resolved in the R/Python wrappers, not here.
    double dimensional_constraint_penalty = 0.0;

    // Frequency-based adaptive parsimony (borrowed from PySR / SymbolicRegression.jl).
    // In tournament selection the base cost is multiplied by
    //   exp(adaptive_parsimony_scaling * normalized_frequency[complexity]),
    // where normalized_frequency is a per-island running histogram over complexity
    // (see RunningSearchStatistics in evolutionary_search.cpp). This penalises
    // *over-represented* complexities rather than *large* ones, so when the
    // population piles up at one size that size is pushed back — a self-balancing
    // diversity pressure that counters the premature-collapse trap the fixed-scalar
    // parsimony causes on low-variance targets (docs/23, docs/24). 0 = disabled,
    // which reproduces the pre-adaptive behaviour byte-for-byte.
    //
    // The cost factor matches SR.jl: member.cost * exp(scaling * normalized_freq[size]),
    // where normalized_freq = frequencies/sum (sums to 1 over the maxsize bins). PySR's
    // installed default is 1040.0 (pysr/sr.py), well-behaved because a uniform histogram
    // cancels in the ranking and the penalty acts on relative frequency differences. The
    // histogram is now sized exactly by maxsize (= max_nodes) with frequency 0 outside
    // [1, maxsize] (RunningSearchStatistics in evolutionary_search.cpp), so 1040 transfers
    // exactly from PySR (docs/28 §B1). ON at the PySR default per CLAUDE.md parity.
    double adaptive_parsimony_scaling = 1040.0;

    // Sliding-window size for the frequency histogram: once the summed counts
    // exceed this, all bins are scaled down proportionally so the statistics track
    // recent search behaviour rather than the whole run. Matches SR.jl's default;
    // kept internal (not an R argument) — it only matters over very long runs.
    int parsimony_window = 100000;

    // Frequency-based mutation acceptance (PySR `use_frequency`, default True — distinct
    // from `use_frequency_in_tournament` / adaptive_parsimony_scaling above). SR.jl's
    // next_generation accepts a mutated child with probability
    //   probChange = old_frequency / new_frequency        (clamped to <= 1 via the
    //                                                       `probChange < rand()` test)
    // where old/new_frequency are the normalized complexity frequencies of the PARENT and
    // CHILD (1e-6 outside [1, maxsize]). This biases acceptance away from over-represented
    // complexities at mutation time, complementing the tournament-cost penalty. annealing
    // is off (PySR default), so the annealing factor is omitted. true = on (PySR default).
    bool use_frequency = true;

    // Gate for the once-per-iteration population constant-optimisation pass (PySR's
    // should_optimize_constants). When on, optimize_and_simplify_population LM-optimises
    // each population member's constants with probability optimize_probability — exactly
    // SR.jl's optimize_and_simplify_population (SingleIteration.jl). This is the ONLY place
    // constants are optimised during evolution: the reg-evol mutation cycle never optimises
    // children, and there is no separate HOF re-optimisation (the earlier per-child + HOF
    // design diverged from PySR and dominated compute — docs/30, docs/31). Deadline-guarded.
    // true = on (default, matches PySR).
    bool should_optimize_constants = true;

    // Probability that a population member is LM-optimised in the once-per-iteration
    // optimize_and_simplify_population pass (PySR optimize_probability, docs/28 §A). This is
    // applied per population member per iteration (gated by should_optimize_constants),
    // NOT per child during mutation — matching SR.jl's do_optimization = rand(pop.n) <
    // optimizer_probability. Members not optimised this iteration keep their inherited
    // constants and are evaluated with a forward pass; they remain valid for selection and
    // are re-considered next iteration. Default 0.14 = PySR installed default.
    double optimize_probability = 0.14;

    // PySR `batching` / `batch_size` (docs/28, SR.jl SingleIteration.jl). When on, the
    // evolution (reg-evol mutation/crossover scoring) and the per-iteration constant
    // optimisation are evaluated on a fresh random subsample of `batch_size` rows instead
    // of the full dataset, so each per-candidate evaluation costs O(batch_size) rather than
    // O(n) — the lever for large row counts. The hall of fame, early-stop test and final
    // result are ALWAYS decided on the FULL dataset: each batched epoch's population and
    // its per-epoch best-seen archive are recomputed on the full data before being merged
    // into the archive, exactly mirroring SR.jl's finalize_costs + best_seen recompute
    // (SymbolicRegression.jl:1129). So batching only affects *which candidates the search
    // explores*, never the accuracy attributed to a reported model. Rows are sampled WITH
    // REPLACEMENT (SR.jl batch()); two batches are drawn per epoch (one for evolution, one
    // for optimisation). batching=false (default, PySR parity) reproduces the full-data path
    // byte-for-byte: no batch is built, no best-seen archive or finalize pass runs.
    bool batching = false;
    // Subsample size used when batching is on (PySR batch_size default 50). Clamped to the
    // row count (PySR caps batch_size at len(X)); must be >= 1.
    std::size_t batch_size = 50;

    // PySR `warmup_maxsize_by` (SR.jl get_cur_maxsize, SearchUtils.jl). When > 0, the
    // size cap applied during evolution grows linearly from 3 up to max_nodes (= maxsize)
    // over the first `warmup_maxsize_by` fraction of the run, then stays at max_nodes:
    //   cur_maxsize = 3 + floor((max_nodes - 3) * f / warmup_maxsize_by)  for f <= warmup_by
    //   cur_maxsize = max_nodes                                            otherwise
    // where f = generations_elapsed / generations. This only caps the mutation/crossover
    // size limit (get_cur_maxsize); the frequency histogram, the initial population and the
    // hall of fame stay at the full max_nodes, matching SR.jl (the Population is built before
    // any cycle, so it uses maxsize, not the warmed-up size). 0.0 = off (PySR installed
    // default), reproducing the fixed-maxsize path byte-for-byte. See docs/42.
    double warmup_maxsize_by = 0.0;

    // Opt-in duplicate-evaluation cache (eval_cache.hpp). Implementation-only
    // memoisation of the forward-pass loss (sse_current): a per-island direct-mapped
    // table keyed by the tree's evaluation-relevant content returns the previously
    // computed SSE for an evaluation-identical tree instead of re-evaluating it.
    // Because sse_current is a pure function of (tree, dataset) and a cache hit is
    // charged to the evaluation counters exactly like a real evaluation, the search
    // results are bit-identical with the cache on or off — this is a speed knob, not
    // a search setting, so it does not touch PySR parity. Active only when batching
    // is off: batches change every pass, so a batched SSE is never reusable. false =
    // off (default; no table is allocated and the code path is byte-identical).
    bool eval_cache = false;

    // Opt-in Keijzer-2003 linear scaling — an "opt-in high-accuracy option" per
    // CLAUDE.md's second layer: a deliberate, behaviour-changing divergence from PySR
    // that is OFF by default (the shipped default reproduces PySR exactly). When on,
    // every forward-pass loss is the best-affine-fit SSE: the weighted least-squares
    // slope/intercept (a, b) of y on the candidate's prediction f are solved in closed
    // form and the loss is sum_i w_i (a*f_i + b - y_i)^2, so the search only has to
    // discover the SHAPE of the target, never its scale or offset (Keijzer 2003;
    // Operon does the same). The constant optimiser (LM) still fits UNSCALED residuals
    // (v1); optimize_and_simplify_population re-scores optimised members through the
    // scaled scorer so scaled and unscaled losses never mix in the archive ordering.
    // At the end of the run the fitted (a, b) are materialised into every reported
    // tree as a*f + b (skipped when they are the identity to numerical precision), so
    // expression/loss/complexity/predict stay self-consistent; the wrap may push a
    // reported tree up to 4 nodes past max_nodes (reporting-time materialisation, not
    // a search move). false = off (default): the search is byte-identical to the
    // PySR-parity path.
    bool linear_scaling = false;

    // Wall-clock timeout. 0 = no limit (fully deterministic; default). Any value > 0
    // stops the search after approximately this many seconds. A run that times out is
    // NOT reproducible across machines — document this in user-facing roxygen.
    double timeout_seconds = 0.0;

    // Verbosity. 0 = silent. 1 = one diagnostic line per epoch on stderr:
    //   [epoch N  t=Xs] best=Y  size med/max=A/B  nconst med/max=C/D
    // Higher values reserved for future use (treated same as 1).
    // This core default is intentionally 0 (silent) so standalone benchmarks /
    // tests stay quiet; the R and Python wrappers override it to 1 to match
    // PySR's installed verbosity=1 default (the public-API default that the
    // parity rule governs). Keeping the core silent while the public API
    // defaults to 1 is the deliberate asymmetry — wrappers always pass the value
    // explicitly (rsymbolic2_r.cpp / rsymbolic2_py.cpp).
    int verbosity = 0;

    // Internal hook for the Phase-0 seeding oracle (diagnostic; not exposed via the
    // R/Python bindings). When non-empty, the first seed_trees.size() members of each
    // island's initial population are copies of these trees instead of random trees.
    // Empty (the default) leaves the default search path byte-identical.
    std::vector<Tree> seed_trees;

    // Optional per-epoch progress observer (docs/53). Invoked once per outer epoch, at
    // a serial point after every island's evolution/migration for that epoch has
    // completed (all OpenMP island workers have joined and HOF/ring migration is done),
    // with a ProgressSnapshot of the current global Pareto front. Pure observation: it
    // is never read by the search, never mutates search state, and consumes no RNG —
    // attaching it cannot change the result. null by default (the default constructor
    // leaves it unset), in which case the call site is a single untaken branch and the
    // run is bit-identical to the callback-less path (PySR Default Parity, CLAUDE.md).
    // Left unset by the R and Python bindings for now (docs/53); only the WASM binding
    // wires it, for the web GUI's live Pareto chart.
    std::function<void(const ProgressSnapshot&)> progress_callback;
};

// PySR `warmup_maxsize_by` (SR.jl get_cur_maxsize, SearchUtils.jl 1.11.0). Returns the size
// cap to apply during an epoch's evolution. When warmup is on (> 0) and the run is within its
// warmup fraction, the cap grows linearly from 3 up to max_nodes; otherwise it is the full
// max_nodes. `fraction_elapsed` = generations_elapsed / generations; all islands evolve in
// lockstep, so SR.jl's per-population fraction collapses to this single value (the population
// count cancels — docs/42). The result is clamped to [1, max_nodes] so a max_nodes < 3
// configuration (SR.jl's start value) cannot exceed the absolute cap. Pure; exposed for unit
// testing and used per epoch in run_evolution.
inline int get_cur_maxsize(int max_nodes, double fraction_elapsed,
                           double warmup_maxsize_by) {
    int cur = max_nodes;
    if (warmup_maxsize_by > 0.0 && fraction_elapsed <= warmup_maxsize_by) {
        cur = 3 + static_cast<int>(std::floor(
                      (max_nodes - 3) * fraction_elapsed / warmup_maxsize_by));
    }
    if (cur < 1)         cur = 1;
    if (cur > max_nodes) cur = max_nodes;
    return cur;
}

// Resolve the OpenMP team size for the island-parallel regions (init + evolution).
// `n_threads` is the user request (SearchOptions::n_threads): 0 (or negative) = auto,
// meaning use `auto_threads` (the caller passes omp_get_max_threads(), which honours
// OMP_NUM_THREADS and otherwise defaults to the core count); a positive value is taken
// literally. The team is always capped at `n_islands` because islands are the only unit of
// parallelism — a larger team leaves surplus threads with no loop iteration, busy-waiting at
// the implicit barrier, which can balloon a node evaluation's wall time and make the timeout
// poll miss its deadline (docs/22). The result is clamped to >= 1. Pure (no OpenMP call) so
// it is unit-testable without a runtime.
inline int resolve_team_size(std::size_t n_islands, int n_threads, int auto_threads) {
    const int desired = n_threads > 0 ? n_threads : std::max(1, auto_threads);
    const std::size_t capped =
        std::min<std::size_t>(n_islands, static_cast<std::size_t>(std::max(1, desired)));
    return std::max(1, static_cast<int>(capped));
}

// The outcome of a search: the best expression found (with constants fitted) plus the
// accuracy/complexity Pareto front.
struct SearchResult {
    Tree tree;
    double loss = 0.0;
    int complexity = 0;
    std::string expression;
    // Display-only companion to `expression`, produced by display_simplify() (docs/52)
    // on a COPY of `tree` at finalization time. Never fed back into the search, never
    // used by predict() (docs/48 D2 "frozen-expression rule": `expression` alone is the
    // evaluatable round-trip source) — purely a shorter/more-readable rendering.
    std::string expression_simplified;
    std::vector<PopMember> pareto_front;
    // Index into pareto_front of the recommended ("best") accuracy/complexity trade-off
    // (PySR model_selection="best"; see select_best). 0 when the front is empty.
    int best_index = 0;
    // Evaluation accounting (summed over islands; reporting only, never a search input).
    // n_evals is the total in max_evals units — forward-pass loss evaluations plus the
    // residual evaluations consumed by constant-optimisation fits — so
    // n_evals == n_forward_evals + n_lm_resid_evals holds by construction. Jacobian
    // builds are reported separately and are never charged to n_evals (a
    // finite-difference Jacobian's residual calls already count as LM residual evals).
    std::uint64_t n_evals = 0;          // total in max_evals units (forward + LM residual)
    std::uint64_t n_forward_evals = 0;  // forward-pass loss evaluations
    std::uint64_t n_lm_resid_evals = 0; // LM residual-function evaluations
    std::uint64_t n_lm_jac_evals = 0;   // LM Jacobian builds (reported only)
    // Duplicate-evaluation cache statistics (summed over islands; 0 unless the opt-in
    // eval_cache option is on). A hit is still counted in n_evals/n_forward_evals like
    // a real evaluation (bit-identity with the cache off), so hits + misses equals the
    // number of forward passes routed through the cache, not extra work.
    std::uint64_t cache_hits = 0, cache_misses = 0;
};

// Run the search to fit y from X by discovering an expression structure and optimizing
// its constants. Deterministic for a fixed seed.
SearchResult run_evolution(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y,
                           const SearchOptions& options);

}  // namespace rsymbolic
