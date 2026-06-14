#include "rsymbolic/search/evolutionary_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
// Use R's error channel when building inside the R package; fall back to
// stderr for the standalone cmake build (no R headers available there).
#ifdef __has_include
#  if __has_include(<R_ext/Print.h>)
#    include <R_ext/Print.h>
#    define rsym_eprintf(...) REprintf(__VA_ARGS__)
#  else
#    include <cstdio>
#    define rsym_eprintf(...) std::fprintf(stderr, __VA_ARGS__)
#  endif
#else
#  include <cstdio>
#  define rsym_eprintf(...) std::fprintf(stderr, __VA_ARGS__)
#endif
#include <memory>
#include <numeric>
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

// SSE with the tree's current constants — no LM, no Jacobian.
// Used for children that skip the optimizer this step (optimize_probability < 1).
// Strictly cheaper than fit(): one forward pass over the data vs. many LM iterations.
double sse_current(const Tree& tree, const std::vector<std::vector<double>>& X,
                   const std::vector<double>& y) {
    const std::vector<double> c = initial_constants(tree);
    double sse = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double pred = evaluate<double>(tree, X[i].data(), c.data());
        const double r = pred - y[i];
        if (!std::isfinite(r)) return kInf;
        sse += r * r;
    }
    return sse;
}

// y_norm = sum((y - mean(y))^2), used as the NMSE denominator.
// Returns 1.0 when the denominator would be zero (constant y or single point).
double compute_y_norm(const std::vector<double>& y) {
    if (y.size() <= 1) return 1.0;
    const double mean_y =
        std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
    double ss = 0.0;
    for (const double v : y) { const double d = v - mean_y; ss += d * d; }
    return ss > 0.0 ? ss : 1.0;
}

// Selection cost: loss/y_norm + parsimony*complexity.
// When parsimony=0 (default) this collapses to loss, preserving pre-B2 behaviour.
inline double selection_cost(const PopMember& m, double parsimony, double y_norm) {
    if (parsimony <= 0.0) return m.loss;
    return m.loss / y_norm + parsimony * static_cast<double>(m.complexity);
}

double fit(Tree& tree, const std::vector<std::vector<double>>& X,
           const std::vector<double>& y, const ConstantOptimizer& optimizer,
           const StopRequested& stop_requested) {
    const std::vector<double> init = initial_constants(tree);
    // Pass stop_requested through so the residual/Jacobian closures can poll it
    // mid-evaluation and signal an abort via problem.aborted (docs/22 Phase 1).
    OptimizationProblem problem = make_least_squares_problem(tree, X, y, init,
                                                             stop_requested);
    const OptimizationResult result = optimizer.optimize(problem, stop_requested);
    if (!result.success || !std::isfinite(result.loss)) return kInf;
    if (!result.constants.empty()) set_constants(tree, result.constants);
    return result.loss;
}

std::size_t sample_index(std::size_t size, std::mt19937_64& rng) {
    return std::uniform_int_distribution<std::size_t>(0, size - 1)(rng);
}

std::size_t tournament_best(const std::vector<PopMember>& pop, std::size_t k,
                            std::mt19937_64& rng, double parsimony, double y_norm) {
    std::size_t best = sample_index(pop.size(), rng);
    double best_cost = selection_cost(pop[best], parsimony, y_norm);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        const double c = selection_cost(pop[cand], parsimony, y_norm);
        if (c < best_cost) { best = cand; best_cost = c; }
    }
    return best;
}

std::size_t tournament_worst(const std::vector<PopMember>& pop, std::size_t k,
                             std::mt19937_64& rng, double parsimony, double y_norm) {
    std::size_t worst = sample_index(pop.size(), rng);
    double worst_cost = selection_cost(pop[worst], parsimony, y_norm);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        const double c = selection_cost(pop[cand], parsimony, y_norm);
        if (c > worst_cost) { worst = cand; worst_cost = c; }
    }
    return worst;
}

void initialize_island(Island& isl,
                       const std::vector<std::vector<double>>& X,
                       const std::vector<double>& y,
                       const SearchOptions& opts,
                       const StopRequested& stop_requested) {
    isl.population.clear();
    isl.population.reserve(opts.population_size);
    for (std::size_t i = 0; i < opts.population_size; ++i) {
        // Empty-population guard: always create at least one member. The downstream
        // tournament selection calls sample_index(pop.size(), ...) which computes
        // uniform_int_distribution(0, size-1) — undefined for size == 0. We only
        // check the predicate before the second member onward, so every island that
        // exists has at least one member and is safe for evolve_island.
        if (i > 0 && stop_requested()) break;
        Tree tree = generate_random_tree(opts.space, isl.rng);
        const double loss = fit(tree, X, y, *isl.optimizer, stop_requested);
        if (opts.simplify_expressions) tree = simplify(tree);
        PopMember m{std::move(tree), loss, 0};
        m.complexity = static_cast<int>(m.tree.size());
        isl.hof.update(m);
        isl.population.push_back(std::move(m));
    }
}

using Clock    = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Returns elapsed seconds since `start`.
inline double elapsed_sec(const TimePoint& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Evolve island for exactly n_gens generations (or until early stop / timeout).
// Returns true if the early-stop threshold was reached.
bool evolve_island(Island& isl,
                   const std::vector<std::vector<double>>& X,
                   const std::vector<double>& y,
                   const SearchOptions& opts,
                   std::size_t n_gens,
                   const TimePoint& t_start,
                   bool has_deadline,
                   double deadline_sec,
                   double y_norm) {
    const std::size_t tournament =
        opts.tournament_size == 0 ? 1 : opts.tournament_size;
    const double parsimony = opts.parsimony;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Build the stop closure once. It reads t_start (immutable after search start)
    // and the wall clock — safe to call from any thread under the OpenMP parallel-for.
    const StopRequested stop = [&] {
        return has_deadline && elapsed_sec(t_start) >= deadline_sec;
    };

    for (std::size_t gen = 0; gen < n_gens; ++gen) {
        if (!isl.hof.empty() && isl.hof.best().loss < opts.target_loss)
            return true;
        if (stop())
            return false;

        for (std::size_t step = 0; step < opts.population_size; ++step) {
            // Coarse step-boundary check: catches overshoot between fit() calls.
            // The finer check inside fit() -> optimizer catches overshoot within a
            // single bloated-tree optimization. Together they bound overshoot to one
            // LM step rather than one full minimize() call. See docs/20.
            if (stop())
                return false;

            const std::size_t parent = tournament_best(isl.population,
                                                       tournament, isl.rng,
                                                       parsimony, y_norm);
            Tree child_tree;
            if (opts.crossover_probability > 0.0 &&
                unit(isl.rng) < opts.crossover_probability) {
                const std::size_t parent2 = tournament_best(isl.population,
                                                            tournament, isl.rng,
                                                            parsimony, y_norm);
                child_tree = subtree_crossover(
                    isl.population[parent].tree,
                    isl.population[parent2].tree,
                    isl.rng, opts.space.max_nodes);
            } else {
                child_tree = isl.population[parent].tree;
                mutate(child_tree, opts.space, isl.rng, opts.const_perturb_scale);
            }

            // Optimize constants with probability optimize_probability (cf. PySR
            // weight_optimize). When skipped, evaluate SSE with inherited constants
            // — cheaper, does not write back to the tree.
            const double loss =
                (opts.optimize_probability >= 1.0 ||
                 unit(isl.rng) < opts.optimize_probability)
                    ? fit(child_tree, X, y, *isl.optimizer, stop)
                    : sse_current(child_tree, X, y);
            if (opts.simplify_expressions) child_tree = simplify(child_tree);
            PopMember child{std::move(child_tree), loss, 0};
            child.complexity = static_cast<int>(child.tree.size());
            isl.hof.update(child);

            const std::size_t worst = tournament_worst(isl.population,
                                                       tournament, isl.rng,
                                                       parsimony, y_norm);
            if (selection_cost(child, parsimony, y_norm) <
                    selection_cost(isl.population[worst], parsimony, y_norm))
                isl.population[worst] = std::move(child);
        }
    }
    return false;
}

// Ring migration: snapshot top-k from each island, then inject into the next.
// Called serially after each epoch barrier — no concurrent access.
void migrate(std::vector<Island>& islands, std::size_t k,
             double parsimony, double y_norm) {
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
                          [parsimony, y_norm](const PopMember* a, const PopMember* b) {
                              return selection_cost(*a, parsimony, y_norm) <
                                     selection_cost(*b, parsimony, y_norm);
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
            // Replace the worst member (by selection cost) if the emigrant is better.
            std::size_t worst_idx = 0;
            double worst_cost = selection_cost(pop[0], parsimony, y_norm);
            for (std::size_t j = 1; j < pop.size(); ++j) {
                const double c = selection_cost(pop[j], parsimony, y_norm);
                if (c > worst_cost) { worst_idx = j; worst_cost = c; }
            }
            if (selection_cost(em, parsimony, y_norm) < worst_cost) {
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

    // Start the clock before island initialisation so that init time counts against
    // the timeout budget. Previously t_start was set after initialize_island, meaning
    // a long init was not charged to the deadline. See docs/20.
    const TimePoint   t_start      = Clock::now();
    const bool        has_deadline = options.timeout_seconds > 0.0;
    const double      deadline_sec = options.timeout_seconds;

    const StopRequested init_stop = [&] {
        return has_deadline && elapsed_sec(t_start) >= deadline_sec;
    };

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
        initialize_island(islands[i], X, y, options, init_stop);
    }

    const std::size_t interval = std::max<std::size_t>(1, options.migration_interval);
    const double      y_norm    = compute_y_norm(y);  // denominator for selection cost

    bool reached_target = false;
    bool timed_out      = false;
    std::size_t epoch   = 0;

    for (std::size_t done = 0;
         done < options.generations && !reached_target && !timed_out;
         done += interval, ++epoch) {
        const std::size_t gens =
            std::min(interval, options.generations - done);

#ifdef _OPENMP
#       pragma omp parallel for schedule(dynamic) if(n > 1)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
            evolve_island(islands[static_cast<std::size_t>(i)], X, y, options,
                          gens, t_start, has_deadline, deadline_sec, y_norm);

        if (n > 1) migrate(islands, options.migration_size, options.parsimony, y_norm);

        // Global early stop: once any island reaches the target loss the answer is
        // found, so stop instead of grinding the remaining islands and epochs through
        // their full budget. Within-epoch early stop is handled inside evolve_island.
        for (const auto& isl : islands) {
            if (!isl.hof.empty() && isl.hof.best().loss < options.target_loss) {
                reached_target = true;
                break;
            }
        }

        if (has_deadline && elapsed_sec(t_start) >= deadline_sec)
            timed_out = true;

        // Per-epoch diagnostic log (verbosity >= 1).
        if (options.verbosity >= 1) {
            // Gather stats across all islands' populations.
            double best_loss = std::numeric_limits<double>::infinity();
            std::vector<int> sizes, nconsts;
            for (const auto& isl : islands) {
                if (!isl.hof.empty()) {
                    const double l = isl.hof.best().loss;
                    if (l < best_loss) best_loss = l;
                }
                for (const auto& m : isl.population) {
                    sizes.push_back(m.complexity);
                    nconsts.push_back(count_constants(m.tree));
                }
            }
            const std::size_t sz = sizes.size();
            int size_med = 0, size_max = 0, nc_med = 0, nc_max = 0;
            if (sz > 0) {
                std::nth_element(sizes.begin(),
                                 sizes.begin() + static_cast<std::ptrdiff_t>(sz / 2),
                                 sizes.end());
                size_med = sizes[sz / 2];
                size_max = *std::max_element(sizes.begin(), sizes.end());
                std::nth_element(nconsts.begin(),
                                 nconsts.begin() + static_cast<std::ptrdiff_t>(sz / 2),
                                 nconsts.end());
                nc_med = nconsts[sz / 2];
                nc_max = *std::max_element(nconsts.begin(), nconsts.end());
            }
            rsym_eprintf(
                "[epoch %zu  t=%.1fs] best=%.3e  size med/max=%d/%d"
                "  nconst med/max=%d/%d\n",
                epoch, elapsed_sec(t_start), best_loss,
                size_med, size_max, nc_med, nc_max);
        }
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
