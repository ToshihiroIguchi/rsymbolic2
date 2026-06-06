#include "rsymbolic/search/evolutionary_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/evolution/crossover.hpp"
#include "rsymbolic/evolution/hall_of_fame.hpp"
#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/simplification/simplify.hpp"

namespace rsymbolic {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Per-island state. Each island is fully self-contained: its own population,
// hall of fame, RNG, and optimizer instance. No state is shared across islands
// during parallel evolution, so no synchronisation is needed inside the loop.
struct Island {
    std::vector<PopMember>              population;
    HallOfFame                          hof;
    std::mt19937_64                     rng;
    std::unique_ptr<ConstantOptimizer>  optimizer;
};

double fit(Tree& tree, const std::vector<std::vector<double>>& X,
           const std::vector<double>& y, const ConstantOptimizer& optimizer) {
    const std::vector<double> init = initial_constants(tree);
    OptimizationProblem problem = make_least_squares_problem(tree, X, y, init);
    const OptimizationResult result = optimizer.optimize(problem);
    if (!result.success || !std::isfinite(result.loss)) return kInf;
    if (!result.constants.empty()) set_constants(tree, result.constants);
    return result.loss;
}

std::size_t sample_index(std::size_t size, std::mt19937_64& rng) {
    return std::uniform_int_distribution<std::size_t>(0, size - 1)(rng);
}

std::size_t tournament_best(const std::vector<PopMember>& pop, std::size_t k,
                            std::mt19937_64& rng) {
    std::size_t best = sample_index(pop.size(), rng);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        if (pop[cand].loss < pop[best].loss) best = cand;
    }
    return best;
}

std::size_t tournament_worst(const std::vector<PopMember>& pop, std::size_t k,
                             std::mt19937_64& rng) {
    std::size_t worst = sample_index(pop.size(), rng);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        if (pop[cand].loss > pop[worst].loss) worst = cand;
    }
    return worst;
}

void initialize_island(Island& isl,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y,
                       const SearchOptions& opts) {
    isl.population.clear();
    isl.population.reserve(opts.population_size);
    for (std::size_t i = 0; i < opts.population_size; ++i) {
        Tree tree = generate_random_tree(opts.space, isl.rng);
        const double loss = fit(tree, X, y, *isl.optimizer);
        if (opts.simplify_expressions) tree = simplify(tree);
        PopMember m{std::move(tree), loss, 0};
        m.complexity = static_cast<int>(m.tree.size());
        isl.hof.update(m);
        isl.population.push_back(std::move(m));
    }
}

// Evolve island for exactly n_gens generations (or until early stop).
// Returns true if the early-stop threshold was reached.
bool evolve_island(Island& isl,
                   const std::vector<std::vector<double>>& X,
                   const std::vector<double>& y,
                   const SearchOptions& opts,
                   std::size_t n_gens) {
    const std::size_t tournament =
        opts.tournament_size == 0 ? 1 : opts.tournament_size;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (std::size_t gen = 0; gen < n_gens; ++gen) {
        if (!isl.hof.empty() && isl.hof.best().loss < opts.target_loss)
            return true;

        for (std::size_t step = 0; step < opts.population_size; ++step) {
            const std::size_t parent = tournament_best(isl.population,
                                                       tournament, isl.rng);
            Tree child_tree;
            if (opts.crossover_probability > 0.0 &&
                unit(isl.rng) < opts.crossover_probability) {
                const std::size_t parent2 = tournament_best(isl.population,
                                                            tournament, isl.rng);
                child_tree = subtree_crossover(
                    isl.population[parent].tree,
                    isl.population[parent2].tree,
                    isl.rng, opts.space.max_nodes);
            } else {
                child_tree = isl.population[parent].tree;
                mutate(child_tree, opts.space, isl.rng, opts.const_perturb_scale);
            }

            const double loss = fit(child_tree, X, y, *isl.optimizer);
            if (opts.simplify_expressions) child_tree = simplify(child_tree);
            PopMember child{std::move(child_tree), loss, 0};
            child.complexity = static_cast<int>(child.tree.size());
            isl.hof.update(child);

            const std::size_t worst = tournament_worst(isl.population,
                                                       tournament, isl.rng);
            if (child.loss < isl.population[worst].loss)
                isl.population[worst] = std::move(child);
        }
    }
    return false;
}

// Ring migration: snapshot top-k from each island, then inject into the next.
// Called serially after each epoch barrier — no concurrent access.
void migrate(std::vector<Island>& islands, std::size_t k) {
    const std::size_t n = islands.size();
    if (n < 2 || k == 0) return;

    // Snapshot before any injection so emigrants are never stale.
    std::vector<std::vector<PopMember>> emigrants(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto& pop = islands[i].population;
        std::vector<PopMember*> ptrs(pop.size());
        for (std::size_t j = 0; j < pop.size(); ++j) ptrs[j] = &pop[j];
        const std::size_t take = std::min(k, pop.size());
        std::partial_sort(ptrs.begin(), ptrs.begin() + static_cast<std::ptrdiff_t>(take),
                          ptrs.end(),
                          [](const PopMember* a, const PopMember* b) {
                              return a->loss < b->loss;
                          });
        emigrants[i].reserve(take);
        for (std::size_t j = 0; j < take; ++j)
            emigrants[i].push_back(*ptrs[j]);
    }

    // Ring i -> (i+1)%n: replace the worst k slots in the destination.
    for (std::size_t i = 0; i < n; ++i) {
        Island& dst = islands[(i + 1) % n];
        auto& pop = dst.population;

        for (const auto& em : emigrants[i]) {
            // Replace the worst member if the emigrant is better.
            std::size_t worst_idx = 0;
            for (std::size_t j = 1; j < pop.size(); ++j)
                if (pop[j].loss > pop[worst_idx].loss) worst_idx = j;
            if (em.loss < pop[worst_idx].loss) {
                pop[worst_idx] = em;
                dst.hof.update(em);
            }
        }
    }
}

}  // namespace

SearchResult run_evolution(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y,
                           const SearchOptions& options) {
    const std::size_t n = std::max<std::size_t>(1, options.n_populations);

    // Build islands. Island 0 always uses options.seed so that n_populations=1
    // produces exactly the same RNG sequence as the pre-island implementation.
    std::vector<Island> islands(n);
    for (std::size_t i = 0; i < n; ++i) {
        islands[i].rng.seed(i == 0
            ? options.seed
            : options.seed + i * UINT64_C(0x9e3779b97f4a7c15));
        islands[i].optimizer =
            OptimizerFactory::create(options.optimizer_type,
                                     options.optimizer_config);
        initialize_island(islands[i], X, y, options);
    }

    const std::size_t interval = std::max<std::size_t>(1, options.migration_interval);

    for (std::size_t done = 0; done < options.generations; done += interval) {
        const std::size_t gens =
            std::min(interval, options.generations - done);

#ifdef _OPENMP
#       pragma omp parallel for schedule(dynamic) if(n > 1)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
            evolve_island(islands[static_cast<std::size_t>(i)], X, y, options, gens);

        if (n > 1) migrate(islands, options.migration_size);
    }

    // Merge per-island halls of fame into a single global result.
    HallOfFame global;
    for (auto& isl : islands) global.merge(isl.hof);

    const PopMember best = global.best();
    SearchResult result;
    result.tree       = best.tree;
    result.loss       = best.loss;
    result.complexity = best.complexity;
    result.expression = to_string(best.tree);
    result.pareto_front = global.pareto_front();
    return result;
}

}  // namespace rsymbolic
