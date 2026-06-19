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

// Per-island running histogram of complexity (node count), for PySR-style
// frequency-based adaptive parsimony. frequencies[c] counts how often a candidate
// of complexity c has been evaluated (every bin starts at 1.0 so none is ever
// zero); normalized[c] = frequencies[c] / sum. Tournament selection multiplies a
// member's base cost by exp(scaling * normalized[complexity]), penalising
// over-represented sizes. Kept per island (share-nothing) so n_populations=1 stays
// fully deterministic. See docs/24.
struct RunningSearchStatistics {
    std::vector<double> frequencies;
    std::vector<double> normalized;
    double              sum         = 0.0;
    int                 window_size = 100000;

    // Size the histogram so complexity in [1, max_nodes+1] indexes safely; bin 0 is
    // unused but kept for 1-based indexing. All bins start at 1.0 (SR.jl's "ones").
    void init(int max_nodes, int window) {
        const std::size_t n =
            static_cast<std::size_t>(std::max(1, max_nodes)) + 2;
        frequencies.assign(n, 1.0);
        normalized.assign(n, 1.0 / static_cast<double>(n));
        sum         = static_cast<double>(n);
        window_size = std::max(1, window);
    }

    int clamp_index(int complexity) const {
        if (complexity < 1) complexity = 1;
        const int last = static_cast<int>(frequencies.size()) - 1;
        if (complexity > last) complexity = last;
        return complexity;
    }

    void update(int complexity) {
        frequencies[static_cast<std::size_t>(clamp_index(complexity))] += 1.0;
        sum += 1.0;
        if (sum > static_cast<double>(window_size)) move_window();
    }

    // Sliding window: scale every bin down proportionally so the histogram tracks
    // recent search behaviour rather than the whole run. SR.jl instead subtracts
    // from the largest bins; the proportional form is simpler and keeps every bin
    // strictly positive (docs/24).
    void move_window() {
        const double factor = static_cast<double>(window_size) / sum;
        double s = 0.0;
        for (double& f : frequencies) { f *= factor; s += f; }
        sum = s;
    }

    void normalize() {
        if (sum <= 0.0) return;
        const double inv = 1.0 / sum;
        for (std::size_t i = 0; i < frequencies.size(); ++i)
            normalized[i] = frequencies[i] * inv;
    }

    double freq_at(int complexity) const {
        return normalized[static_cast<std::size_t>(clamp_index(complexity))];
    }
};

// Per-island state. Each island is fully self-contained: its own population,
// hall of fame, RNG, optimizer instance, and complexity statistics. No state is
// shared across islands during parallel evolution, so no synchronisation is
// needed inside the loop.
struct Island {
    std::vector<PopMember>              population;
    HallOfFame                          hof;
    std::mt19937_64                     rng;
    std::unique_ptr<ConstantOptimizer>  optimizer;
    RunningSearchStatistics             stats;
};

// SSE with the tree's current constants — no LM, no Jacobian.
// Used for children that skip the optimizer this step (optimize_probability < 1).
// Strictly cheaper than fit(): one forward pass over the data vs. many LM iterations.
double sse_current(const Tree& tree, const std::vector<std::vector<double>>& X,
                   const std::vector<double>& y) {
    const std::vector<double> c = initial_constants(tree);
    // Reuse one evaluation stack across all points (docs/23 §4): no per-point alloc.
    std::vector<double> stack;
    double sse = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double pred = evaluate<double>(tree, X[i].data(), c.data(), stack);
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

// Base selection cost: loss/y_norm + parsimony*complexity.
// When parsimony=0 this collapses to raw loss, preserving pre-B2 behaviour. Used
// by migration and as the base for the frequency-adjusted tournament cost.
inline double base_cost(const PopMember& m, double parsimony, double y_norm) {
    if (parsimony <= 0.0) return m.loss;
    return m.loss / y_norm + parsimony * static_cast<double>(m.complexity);
}

// Frequency-adjusted cost used in tournament selection. With scaling<=0 it is
// exactly base_cost (frequency penalty disabled — byte-identical to pre-adaptive
// behaviour). Otherwise the base is taken in normalized-loss units
// (loss/y_norm + parsimony*complexity) so the multiplicative factor is comparable
// across problems, and multiplied by exp(scaling * normalized_frequency[complexity]).
// The exponent is clamped to avoid overflow to +inf, which would tie every
// candidate. See docs/24.
inline double adjusted_cost(const PopMember& m, double parsimony, double y_norm,
                            const RunningSearchStatistics& stats, double scaling) {
    if (scaling <= 0.0) return base_cost(m, parsimony, y_norm);
    const double base =
        m.loss / y_norm + parsimony * static_cast<double>(m.complexity);
    double e = scaling * stats.freq_at(m.complexity);
    if (e > 50.0) e = 50.0;
    return base * std::exp(e);
}

double fit(Tree& tree, const std::shared_ptr<const Dataset>& data,
           const ConstantOptimizer& optimizer,
           const StopRequested& stop_requested) {
    const std::vector<double> init = initial_constants(tree);
    // Pass stop_requested through so the residual/Jacobian closures can poll it
    // mid-evaluation and signal an abort via problem.aborted (docs/22 Phase 1).
    // The dataset is shared (not copied) so each fit() avoids the per-call dataset
    // copy that was the dominant heap-contention source (docs/23 §4).
    OptimizationProblem problem = make_least_squares_problem(tree, data, init,
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
                            std::mt19937_64& rng, double parsimony, double y_norm,
                            const RunningSearchStatistics& stats, double scaling) {
    std::size_t best = sample_index(pop.size(), rng);
    double best_cost = adjusted_cost(pop[best], parsimony, y_norm, stats, scaling);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        const double c = adjusted_cost(pop[cand], parsimony, y_norm, stats, scaling);
        if (c < best_cost) { best = cand; best_cost = c; }
    }
    return best;
}

std::size_t tournament_worst(const std::vector<PopMember>& pop, std::size_t k,
                             std::mt19937_64& rng, double parsimony, double y_norm,
                             const RunningSearchStatistics& stats, double scaling) {
    std::size_t worst = sample_index(pop.size(), rng);
    double worst_cost = adjusted_cost(pop[worst], parsimony, y_norm, stats, scaling);
    for (std::size_t i = 1; i < k; ++i) {
        const std::size_t cand = sample_index(pop.size(), rng);
        const double c = adjusted_cost(pop[cand], parsimony, y_norm, stats, scaling);
        if (c > worst_cost) { worst = cand; worst_cost = c; }
    }
    return worst;
}

void initialize_island(Island& isl,
                       const std::shared_ptr<const Dataset>& data,
                       const SearchOptions& opts,
                       const StopRequested& stop_requested) {
    isl.population.clear();
    isl.population.reserve(opts.population_size);
    isl.stats.init(opts.space.max_nodes, opts.parsimony_window);
    for (std::size_t i = 0; i < opts.population_size; ++i) {
        // Empty-population guard: always create at least one member. The downstream
        // tournament selection calls sample_index(pop.size(), ...) which computes
        // uniform_int_distribution(0, size-1) — undefined for size == 0. We only
        // check the predicate before the second member onward, so every island that
        // exists has at least one member and is safe for evolve_island.
        if (i > 0 && stop_requested()) break;
        Tree tree = generate_random_tree(opts.space, isl.rng);
        const double loss = fit(tree, data, *isl.optimizer, stop_requested);
        if (opts.simplify_expressions) tree = simplify(tree);
        PopMember m{std::move(tree), loss, 0};
        m.complexity = static_cast<int>(m.tree.size());
        isl.hof.update(m);
        isl.stats.update(m.complexity);
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
                   const std::shared_ptr<const Dataset>& data,
                   const SearchOptions& opts,
                   std::size_t n_gens,
                   const TimePoint& t_start,
                   bool has_deadline,
                   double deadline_sec,
                   double y_norm) {
    const std::size_t tournament =
        opts.tournament_size == 0 ? 1 : opts.tournament_size;
    const double parsimony = opts.parsimony;
    const double scaling   = opts.adaptive_parsimony_scaling;
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

        // Refresh the normalized frequency snapshot once per generation (cheap,
        // O(max_nodes)). Tournament selection within this generation reads this
        // snapshot; the raw counts keep accumulating and are re-normalized next
        // generation (matches SR.jl's per-cycle cadence; deterministic).
        if (scaling > 0.0) isl.stats.normalize();

        for (std::size_t step = 0; step < opts.population_size; ++step) {
            // Coarse step-boundary check: catches overshoot between fit() calls.
            // The finer check inside fit() -> optimizer catches overshoot within a
            // single bloated-tree optimization. Together they bound overshoot to one
            // LM step rather than one full minimize() call. See docs/20.
            if (stop())
                return false;

            const std::size_t parent = tournament_best(isl.population,
                                                       tournament, isl.rng,
                                                       parsimony, y_norm,
                                                       isl.stats, scaling);
            Tree child_tree;
            if (opts.crossover_probability > 0.0 &&
                unit(isl.rng) < opts.crossover_probability) {
                const std::size_t parent2 = tournament_best(isl.population,
                                                            tournament, isl.rng,
                                                            parsimony, y_norm,
                                                            isl.stats, scaling);
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
                    ? fit(child_tree, data, *isl.optimizer, stop)
                    : sse_current(child_tree, data->X, data->y);
            if (opts.simplify_expressions) child_tree = simplify(child_tree);
            PopMember child{std::move(child_tree), loss, 0};
            child.complexity = static_cast<int>(child.tree.size());
            const int child_complexity = child.complexity;  // child may be moved below
            isl.hof.update(child);

            const std::size_t worst = tournament_worst(isl.population,
                                                       tournament, isl.rng,
                                                       parsimony, y_norm,
                                                       isl.stats, scaling);
            if (adjusted_cost(child, parsimony, y_norm, isl.stats, scaling) <
                    adjusted_cost(isl.population[worst], parsimony, y_norm,
                                  isl.stats, scaling))
                isl.population[worst] = std::move(child);

            // Record this child's complexity in the running histogram. Updating
            // after the selection above keeps every comparison in this generation
            // consistent with the start-of-generation normalized snapshot; the new
            // count only affects subsequent generations. No RNG is consumed, so
            // fixed-seed determinism is preserved.
            isl.stats.update(child_complexity);
        }
    }
    return false;
}

// Ring migration: snapshot top-k from each island, then inject into the next.
// Called serially after each epoch barrier — no concurrent access. Ranking uses
// base_cost (no frequency factor): migration transfers quality between islands and
// must not depend on any single island's complexity histogram.
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
                              return base_cost(*a, parsimony, y_norm) <
                                     base_cost(*b, parsimony, y_norm);
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
            double worst_cost = base_cost(pop[0], parsimony, y_norm);
            for (std::size_t j = 1; j < pop.size(); ++j) {
                const double c = base_cost(pop[j], parsimony, y_norm);
                if (c > worst_cost) { worst_idx = j; worst_cost = c; }
            }
            if (base_cost(em, parsimony, y_norm) < worst_cost) {
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

    // Copy the dataset once into shared, immutable storage. Every island's fit()
    // references this single Dataset instead of copying X/y per evaluation, which was
    // the dominant multi-island heap-contention source (docs/23 §4). Read-only and
    // shared by const pointer, so concurrent island workers never write to it.
    const auto data = std::make_shared<const Dataset>(Dataset{X, y});

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
        initialize_island(islands[i], data, options, init_stop);
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
        // One worker thread per island, capped at the hardware. The default
        // OpenMP team size is the core count; when that exceeds the number of
        // islands the surplus threads have no loop iteration and busy-wait at the
        // implicit barrier, starving the island workers. Under that starvation a
        // single node evaluation's wall time can balloon to tens of seconds, so
        // the deadline poll (which runs on the starved worker) cannot fire and the
        // run grossly overshoots its timeout (docs/22). Islands are the only unit
        // of parallelism here, so capping the team at n loses no useful concurrency.
        const int omp_threads = static_cast<int>(std::min<std::size_t>(
            n, static_cast<std::size_t>(std::max(1, omp_get_num_procs()))));
#       pragma omp parallel for schedule(dynamic) num_threads(omp_threads) if(n > 1)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i)
            evolve_island(islands[static_cast<std::size_t>(i)], data, options,
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
