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
    double target_loss = 1e-10;       // early stop once the best loss is below this
    double const_perturb_scale = 0.5;
    bool simplify_expressions = true;  // algebraically simplify fitted candidates
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
