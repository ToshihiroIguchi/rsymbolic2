#include "rsymbolic/search/evolutionary_search.hpp"

#include <cmath>
#include <limits>
#include <random>

#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"

namespace rsymbolic {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Fit a candidate tree's constants to the data and return its loss (SSE). On failure
// the loss is +Inf so the candidate is never selected. The optimized constants are
// written back into the tree.
double fit(Tree& tree, const std::vector<std::vector<double>>& X,
           const std::vector<double>& y, const ConstantOptimizer& optimizer) {
    const std::vector<double> init = initial_constants(tree);
    OptimizationProblem problem = make_least_squares_problem(tree, X, y, init);
    const OptimizationResult result = optimizer.optimize(problem);
    if (!result.success || !std::isfinite(result.loss)) {
        return kInf;
    }
    if (!result.constants.empty()) {
        set_constants(tree, result.constants);
    }
    return result.loss;
}

std::size_t sample_index(std::size_t size, std::mt19937_64& rng) {
    return std::uniform_int_distribution<std::size_t>(0, size - 1)(rng);
}

// Tournament selection: index of the lowest-loss member among a random sample.
std::size_t tournament_best(const std::vector<PopMember>& pop, std::size_t k,
                            std::mt19937_64& rng) {
    std::size_t best = sample_index(pop.size(), rng);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        if (pop[cand].loss < pop[best].loss) best = cand;
    }
    return best;
}

// Reverse tournament: index of the highest-loss member among a random sample.
std::size_t tournament_worst(const std::vector<PopMember>& pop, std::size_t k,
                             std::mt19937_64& rng) {
    std::size_t worst = sample_index(pop.size(), rng);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        if (pop[cand].loss > pop[worst].loss) worst = cand;
    }
    return worst;
}

}  // namespace

SearchResult run_evolution(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y,
                           const SearchOptions& options) {
    std::mt19937_64 rng(options.seed);
    const std::unique_ptr<ConstantOptimizer> optimizer =
        OptimizerFactory::create(options.optimizer_type, options.optimizer_config);

    HallOfFame hall_of_fame;
    std::vector<PopMember> population;
    population.reserve(options.population_size);

    // Initialize: random trees, each fitted.
    for (std::size_t i = 0; i < options.population_size; ++i) {
        Tree tree = generate_random_tree(options.space, rng);
        const double loss = fit(tree, X, y, *optimizer);
        PopMember member{std::move(tree), loss, 0};
        member.complexity = static_cast<int>(member.tree.size());
        hall_of_fame.update(member);
        population.push_back(std::move(member));
    }

    const std::size_t tournament =
        options.tournament_size == 0 ? 1 : options.tournament_size;

    // Steady-state evolution.
    for (std::size_t gen = 0; gen < options.generations; ++gen) {
        if (!hall_of_fame.empty() && hall_of_fame.best().loss < options.target_loss) {
            break;
        }
        for (std::size_t step = 0; step < options.population_size; ++step) {
            const std::size_t parent = tournament_best(population, tournament, rng);
            Tree child_tree = population[parent].tree;
            mutate(child_tree, options.space, rng, options.const_perturb_scale);

            const double loss = fit(child_tree, X, y, *optimizer);
            PopMember child{std::move(child_tree), loss, 0};
            child.complexity = static_cast<int>(child.tree.size());
            hall_of_fame.update(child);

            const std::size_t worst = tournament_worst(population, tournament, rng);
            if (child.loss < population[worst].loss) {
                population[worst] = std::move(child);
            }
        }
    }

    const PopMember best = hall_of_fame.best();
    SearchResult result;
    result.tree = best.tree;
    result.loss = best.loss;
    result.complexity = best.complexity;
    result.expression = to_string(best.tree);
    result.pareto_front = hall_of_fame.pareto_front();
    return result;
}

}  // namespace rsymbolic
