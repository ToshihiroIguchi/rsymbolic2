#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rsymbolic/evolution/hall_of_fame.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

namespace rsymbolic {

// Configuration for the (minimal, steady-state) evolutionary search.
struct SearchOptions {
    SearchSpace space;
    std::size_t population_size = 200;
    std::size_t generations = 20;
    std::size_t tournament_size = 4;
    // SelfLM is the default: it matches EigenLM's recovery rate problem-for-problem
    // while being ~7-8x faster per fit and, crucially, performing zero large per-fit
    // heap allocations — which restores multi-island scaling (4-thread cpu/wall
    // 1.4 -> 2.9 on the heap probe). EigenLM remains selectable for comparison.
    // See docs/25.
    OptimizerType optimizer_type = OptimizerType::SelfLM;
    OptimizerConfig optimizer_config{};
    std::uint64_t seed = 0;
    double target_loss = 1e-10;         // early stop once the best loss is below this
    double const_perturb_scale = 0.5;
    bool simplify_expressions = true;   // algebraically simplify fitted candidates
    // Fraction of evolution steps that perform subtree crossover instead of mutation.
    // 0.0 = mutation only (original behaviour); 0.5 = equal chance of each.
    double crossover_probability = 0.5;

    // Island-model parallelism. n_populations=1 reproduces the single-population
    // path exactly (island 0 uses seed directly; RNG sequence is identical).
    std::size_t n_populations      = 1;   // number of parallel islands
    std::size_t migration_interval = 10;  // evolve this many generations between migrations
    std::size_t migration_size     = 5;   // top-k individuals sent to the next island

    // Selection cost = loss / y_norm + parsimony * complexity, where
    // y_norm = sum((y - mean(y))^2). Borrowed from PySR's loss_to_cost: dividing
    // by y_norm makes the penalty scale-stable across problems of different y-variance
    // (equivalent to weighting by NMSE rather than raw SSE). 0 = off.
    // Default 1e-3 chosen from a sweep over {0, 1e-4, 5e-4, 1e-3, 5e-3} on
    // Nguyen N9/N10/N1/N5/N7 (pop=500, n_pops=4, gen=200, timeout=120s):
    // all cases recovered at every p; p=1e-3 gave ~10× speedup on N9 s5
    // (123s→12s) and 6× on N10 s3 (48s→8s) with no fast-set regression
    // (see docs/13, B2.8). B3 (adaptive parsimony) not needed.
    double parsimony = 1e-3;

    // Frequency-based adaptive parsimony (borrowed from PySR / SymbolicRegression.jl).
    // In tournament selection the base cost is multiplied by
    //   exp(adaptive_parsimony_scaling * normalized_frequency[complexity]),
    // where normalized_frequency is a per-island running histogram over complexity
    // (see RunningSearchStatistics in evolutionary_search.cpp). This penalises
    // *over-represented* complexities rather than *large* ones, so when the
    // population piles up at one size that size is pushed back — a self-balancing
    // diversity pressure that counters the premature-collapse trap the fixed-scalar
    // parsimony causes on low-variance targets (docs/23, docs/24). 0 = disabled,
    // which reproduces the pre-adaptive behaviour byte-for-byte. SR.jl ships 1040;
    // our normalization scale differs, so the default is chosen by measurement.
    double adaptive_parsimony_scaling = 0.0;

    // Sliding-window size for the frequency histogram: once the summed counts
    // exceed this, all bins are scaled down proportionally so the statistics track
    // recent search behaviour rather than the whole run. Matches SR.jl's default;
    // kept internal (not an R argument) — it only matters over very long runs.
    int parsimony_window = 100000;

    // Probability that a newly produced child has its constants LM-optimized.
    // 1.0 = optimize every child (pre-B1 behavior; expensive on bloated trees).
    // Lower values make constant optimization a probabilistic event (cf. PySR
    // weight_optimize), reducing the per-child LM cost that causes runtime blowup
    // on large trees. Default 0.1 chosen from a sweep over {1.0, 0.5, 0.2, 0.1}
    // on Nguyen N9/N10/N1/N5/N7: all cases recovered at every p; p=0.1 gave 3.5–4×
    // speedup on the slow cases with no fast-set regression (see docs/13, B1.8).
    // Children not LM-optimized receive a plain SSE evaluation with their inherited
    // constants; they are still valid candidates for selection.
    double optimize_probability = 0.1;

    // Wall-clock timeout. 0 = no limit (fully deterministic; default). Any value > 0
    // stops the search after approximately this many seconds. A run that times out is
    // NOT reproducible across machines — document this in user-facing roxygen.
    double timeout_seconds = 0.0;

    // Verbosity. 0 = silent. 1 = one diagnostic line per epoch on stderr:
    //   [epoch N  t=Xs] best=Y  size med/max=A/B  nconst med/max=C/D
    // Higher values reserved for future use (treated same as 1).
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
};

// Run the search to fit y from X by discovering an expression structure and optimizing
// its constants. Deterministic for a fixed seed.
SearchResult run_evolution(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y,
                           const SearchOptions& options);

}  // namespace rsymbolic
