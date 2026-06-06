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
    OptimizerType optimizer_type = OptimizerType::EigenLM;
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
