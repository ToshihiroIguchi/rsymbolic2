// Non-fit hot-path contention probe (follow-up to bench_heap / docs/23 §4).
//
// bench_heap showed the *fit* workload now scales acceptably with the allocation-free
// SelfLMOptimizer (cpu/wall ~2.9 at 4 threads) and the pure residual/eval loop scales
// nearly ideally (~3.7). Yet the production island search still only reaches
// cpu/wall ~1.8 at n_populations=4 (benchmarks/diag_omp_check.R), and that probe runs
// optimize_probability=0.1 — i.e. ~90% of children never call fit(). So the remaining
// throughput loss must live in the *non-fit* per-child path of evolve_island():
//   parent copy / subtree_crossover  (a fresh Tree = std::vector<Node> per child)
//   sse_current                       (initial_constants + an eval stack per call)
//   simplify                          (returns a new Tree; internal subtree temporaries)
//   HallOfFame::update                (copies the tree into a std::map entry)
// Each is a small heap allocation, several per child, millions per search — exactly the
// process-wide-heap-lock contention pattern docs/23 traced for fit().
//
// This probe reproduces that per-child sequence in N INDEPENDENT std::threads (no
// OpenMP, no islands, no migration, no epoch barriers). Every thread has its own
// dataset, RNG, tree pool, eval stack and hall of fame — nothing is shared but the
// heap. So any cpu/wall < N is pure shared-allocator contention, with the OpenMP/
// migration machinery removed as a confound. If this reproduces the ~1.8 ceiling, the
// bottleneck is the per-child allocation path, not the parallel scaffolding.
//
// Variants attribute the contention to sub-steps by removing one at a time.
//
// Usage: bench_evolve [threads] [children_per_thread]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
static double process_cpu_seconds() {
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0.0;
    ULARGE_INTEGER ku, uu;
    ku.LowPart = k.dwLowDateTime; ku.HighPart = k.dwHighDateTime;
    uu.LowPart = u.dwLowDateTime; uu.HighPart = u.dwHighDateTime;
    return (static_cast<double>(ku.QuadPart) + static_cast<double>(uu.QuadPart)) * 1e-7;
}
#else
#  include <sys/resource.h>
static double process_cpu_seconds() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 +
           r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
}
#endif

#include "rsymbolic/evolution/crossover.hpp"
#include "rsymbolic/evolution/hall_of_fame.hpp"
#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/simplification/simplify.hpp"

using namespace rsymbolic;

namespace {

// Variant flags: which sub-steps of the non-fit child path to run.
struct Step { bool simplify; bool hof; const char* label; };

SearchSpace make_space(int features) {
    SearchSpace s;
    s.binary_ops   = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    s.unary_ops    = {UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp};
    s.num_features = features;
    s.max_depth    = 6;
    s.max_nodes    = 50;
    return s;
}

// SSE with the tree's current constants, reusing one eval stack — byte-for-byte the
// work sse_current() does in evolutionary_search.cpp (minus the early-exit on inf).
double sse_reused(const Tree& tree, const std::vector<std::vector<double>>& X,
                  const std::vector<double>& y, std::vector<double>& stack) {
    const std::vector<double> c = initial_constants(tree);
    double sse = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double pred = evaluate<double>(tree, X[i].data(), c.data(), stack);
        const double r = pred - y[i];
        sse += r * r;
    }
    return sse;
}

// One worker: its own dataset, pool, rng, stack and HoF. Runs `children`
// mutate-or-crossover child steps, mirroring evolve_island's non-fit branch
// (crossover_probability = 0.5, the Feynman gate value).
void worker(int children, std::uint64_t seed, const Step step) {
    const std::size_t m = 1000;
    const int features  = 4;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<std::vector<double>> X(m, std::vector<double>(features));
    std::vector<double> y(m);
    for (std::size_t i = 0; i < m; ++i) {
        for (int j = 0; j < features; ++j) X[i][j] = u(rng);
        y[i] = X[i][0] * X[i][1] + 0.5 * X[i][2];
    }

    const SearchSpace space = make_space(features);

    std::vector<Tree> pool;
    while (pool.size() < 64) {
        Tree t = generate_random_tree(space, rng);
        if (count_constants(t) > 0) pool.push_back(std::move(t));
    }

    HallOfFame hof;
    std::vector<double> stack;
    volatile double sink = 0.0;
    for (int it = 0; it < children; ++it) {
        const std::size_t a = static_cast<std::size_t>(it) % pool.size();
        Tree child;
        if (unit(rng) < 0.5) {
            const std::size_t b = static_cast<std::size_t>(it * 7 + 1) % pool.size();
            child = subtree_crossover(pool[a], pool[b], rng, space.max_nodes);
        } else {
            child = pool[a];                       // copy parent (one Tree alloc)
            mutate(child, space, rng, 0.5);
        }
        const double sse = sse_reused(child, X, y, stack);
        sink += sse;
        if (step.simplify) child = simplify(child);
        if (step.hof) {
            PopMember pm{std::move(child), sse, static_cast<int>(child.size())};
            // child was moved from; complexity above reads the moved-from size (0),
            // which is irrelevant to the allocation/timing being measured.
            hof.update(pm);
        }
    }
    (void)sink;
}

struct RunResult { double wall; double cpu; };

RunResult run(int threads, int children_per_thread, const Step step) {
    const double cpu0 = process_cpu_seconds();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t)
        pool.emplace_back(worker, children_per_thread,
                          static_cast<std::uint64_t>(100 + t), step);
    for (auto& th : pool) th.join();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {wall, process_cpu_seconds() - cpu0};
}

}  // namespace

int main(int argc, char* argv[]) {
    const int max_threads = argc > 1 ? std::atoi(argv[1]) : 4;
    const int children    = argc > 2 ? std::atoi(argv[2]) : 4000;

    const Step variants[] = {
        {true,  true,  "full non-fit child step (mutate/xover + sse + simplify + hof)"},
        {false, true,  "no simplify (mutate/xover + sse + hof)"},
        {true,  false, "no hof update (mutate/xover + sse + simplify)"},
        {false, false, "mutate/xover + sse only"},
    };

    for (const Step& v : variants) {
        std::printf("\n=== %s: %d children/thread, m=1000 ===\n", v.label, children);
        std::printf("%-10s %-12s %-12s %-12s %-12s\n",
                    "threads", "wall_s", "wall/wall1", "throughput", "cpu/wall");
        std::fflush(stdout);

        double wall1 = 0.0;
        for (int t : {1, 2, 4}) {
            if (t > max_threads) break;
            const RunResult r = run(t, children, v);
            if (t == 1) wall1 = r.wall;
            const double thru = (static_cast<double>(t) * children / r.wall) /
                                (static_cast<double>(children) / wall1);
            std::printf("%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n",
                        t, r.wall, r.wall / wall1, thru, r.cpu / r.wall);
            std::fflush(stdout);
        }
    }
    std::printf("\nideal: cpu/wall == #threads; cpu/wall << #threads => heap-lock contention\n");
    return 0;
}
