// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cstddef>
#include <cstdint>
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
};

// The outcome of a search: the best expression found (with constants fitted) plus the
// accuracy/complexity Pareto front.
struct SearchResult {
    Tree tree;
    double loss = 0.0;
    int complexity = 0;
    std::string expression;
    std::vector<PopMember> pareto_front;
    // Index into pareto_front of the recommended ("best") accuracy/complexity trade-off
    // (PySR model_selection="best"; see select_best). 0 when the front is empty.
    int best_index = 0;
};

// Run the search to fit y from X by discovering an expression structure and optimizing
// its constants. Deterministic for a fixed seed.
SearchResult run_evolution(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y,
                           const SearchOptions& options);

}  // namespace rsymbolic
