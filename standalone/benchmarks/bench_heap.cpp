// Heap-contention isolation probe (docs/23 §4).
//
// The island health probe shows cpu/wall ~1.5 at n_populations=4. Two hypotheses
// explain blocked threads: (1) heap-allocator lock contention from the ~6000
// mallocs every fit() does (see bench_alloc), or (2) island load imbalance /
// epoch-barrier idling. This probe isolates hypothesis (1): it runs N std::threads,
// each doing the SAME fixed amount of fit() work on its own copy of the dataset,
// with NO islands, NO migration, NO OpenMP barriers. Each thread is independent and
// perfectly balanced, so any super-linear wall growth from 1 to N threads is pure
// shared-resource (heap) contention.
//
//   ideal : wall(N) == wall(1)     (N threads do N x the work in the same time)
//   heap-bound : wall(N) >> wall(1), cpu/wall(N) < N (threads block on the heap lock)
//
// Usage: bench_heap [threads] [fits_per_thread]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

// Process CPU time (user+kernel, summed over all threads), to distinguish a
// frequency/power ceiling (threads busy but slow -> cpu/wall ~= #threads) from
// blocking on a shared resource (threads idle -> cpu/wall << #threads).
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

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

using namespace rsymbolic;

namespace {

SearchSpace make_space(int features, bool transcendental) {
    SearchSpace s;
    s.binary_ops   = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    // Toggle libm transcendentals to isolate whether they (not the heap/Eigen) are
    // the per-fit serialization point.
    s.unary_ops    = transcendental
        ? std::vector<UnaryOp>{UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp}
        : std::vector<UnaryOp>{UnaryOp::Neg, UnaryOp::Square};
    s.num_features = features;
    s.max_depth    = 6;
    s.max_nodes    = 50;
    return s;
}

// One worker: build its own dataset + tree set, then run `fits` fit()-equivalent
// evaluations. Self-contained — nothing shared between workers except the heap.
// mode: 0 = full EigenLM optimize; 1 = residual-only loop (no Eigen, our code only,
// reused buffer) to isolate whether the serialization is inside Eigen; 2 = full
// SelfLM optimize (allocation-free in-house LM) to confirm it scales like mode 1.
void worker(int fits, std::uint64_t seed, bool transcendental, int mode) {
    const std::size_t m = 1000;
    const int features  = 4;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<std::vector<double>> X(m, std::vector<double>(features));
    std::vector<double> y(m);
    for (std::size_t i = 0; i < m; ++i) {
        for (int j = 0; j < features; ++j) X[i][j] = u(rng);
        y[i] = X[i][0] * X[i][1] + 0.5 * X[i][2];
    }
    // Share the dataset once, exactly like run_evolution's hot path (docs/23 §4).
    const auto dataset =
        std::make_shared<const Dataset>(Dataset{std::move(X), std::move(y)});

    const SearchSpace space = make_space(features, transcendental);
    const OptimizerType opt_type =
        mode == 2 ? OptimizerType::SelfLM : OptimizerType::EigenLM;
    const auto optimizer = OptimizerFactory::create(opt_type, {});

    // Pre-generate a pool of trees with constants so the inner loop spends its time
    // in fit() (make_least_squares + optimize), exactly like the search hot path.
    std::vector<Tree> pool;
    while (pool.size() < 32) {
        Tree t = generate_random_tree(space, rng);
        if (count_constants(t) > 0) pool.push_back(std::move(t));
    }

    volatile double sink = 0.0;
    std::vector<double> rbuf(m, 0.0);  // reused residual buffer for mode 1
    for (int it = 0; it < fits; ++it) {
        const Tree& tree = pool[static_cast<std::size_t>(it) % pool.size()];
        const std::vector<double> init = initial_constants(tree);
        OptimizationProblem p = make_least_squares_problem(tree, dataset, init);
        if (mode == 0 || mode == 2) {
            const OptimizationResult r = optimizer->optimize(p);
            sink += r.loss;
        } else {
            // Residual-only: ~as many residual evaluations as a small LM run, all in
            // our own code (evaluate + reused stack), no Eigen at all.
            for (int e = 0; e < 50; ++e) {
                p.residuals(init, rbuf);
                sink += rbuf[0];
            }
        }
    }
    (void)sink;
}

// Pure-arithmetic worker: no heap allocation, no libm, no Eigen — just a tight FP
// loop. Isolates the hardware/scheduler from any library lock. If this scales
// (cpu/wall ~= #threads) but the fit workload does not, the block is in software
// (libm / Eigen / allocator), not the cores.
void busy_worker(long long iters) {
    volatile double sink = 0.0;
    double a = 1.0000001, x = 0.5;
    for (long long i = 0; i < iters; ++i) {
        x = x * a + 1e-9;
        x = x - 0.3 * x;            // pure mul/add/sub, no transcendentals
        if (x > 1e6) x = 0.5;
        sink += x;
    }
    (void)sink;
}

struct RunResult { double wall; double cpu; };

RunResult run_busy(int threads, long long iters) {
    const double cpu0 = process_cpu_seconds();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) pool.emplace_back(busy_worker, iters);
    for (auto& th : pool) th.join();
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return {wall, process_cpu_seconds() - cpu0};
}

RunResult run(int threads, int fits_per_thread, bool transcendental, int mode) {
    const double cpu0 = process_cpu_seconds();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t)
        pool.emplace_back(worker, fits_per_thread,
                          static_cast<std::uint64_t>(100 + t), transcendental, mode);
    for (auto& th : pool) th.join();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    return {wall, process_cpu_seconds() - cpu0};
}

}  // namespace

int main(int argc, char* argv[]) {
    const int max_threads = argc > 1 ? std::atoi(argv[1]) : 4;
    const int fits        = argc > 2 ? std::atoi(argv[2]) : 400;

    // Control: pure-arithmetic scaling on this hardware (no alloc/libm/Eigen).
    std::printf("=== control: pure FP arithmetic (no alloc/libm/Eigen) ===\n");
    std::printf("%-10s %-12s %-12s %-12s\n", "threads", "wall_s", "wall/wall1", "cpu/wall");
    std::fflush(stdout);
    double bwall1 = 0.0;
    for (int t : {1, 2, 4}) {
        if (t > max_threads) break;
        const RunResult r = run_busy(t, 2000000000LL);
        if (t == 1) bwall1 = r.wall;
        std::printf("%-10d %-12.2f %-12.2f %-12.2f\n",
                    t, r.wall, r.wall / bwall1, r.cpu / r.wall);
        std::fflush(stdout);
    }

    struct Variant { const char* label; bool transc; int mode; };
    const Variant variants[] = {
        {"EigenLM optimize, transcendental ops", true, 0},
        {"SelfLM optimize (allocation-free), transcendental ops", true, 2},
        {"residual-only (no Eigen), transcendental ops", true, 1},
    };
    for (const Variant& v : variants) {
        std::printf("\n=== fit workload (%s): %d fits on m=1000 dataset ===\n",
                    v.label, fits);
        std::printf("%-10s %-12s %-12s %-12s %-12s\n",
                    "threads", "wall_s", "wall/wall1", "throughput", "cpu/wall");
        std::fflush(stdout);

        double wall1 = 0.0;
        for (int t : {1, 2, 4}) {
            if (t > max_threads) break;
            const RunResult r = run(t, fits, v.transc, v.mode);
            if (t == 1) wall1 = r.wall;
            // throughput = total fits / wall, normalized so t=1 is 1.0.
            const double thru = (static_cast<double>(t) * fits / r.wall) /
                                (static_cast<double>(fits) / wall1);
            std::printf("%-10d %-12.2f %-12.2f %-12.2f %-12.2f\n",
                        t, r.wall, r.wall / wall1, thru, r.cpu / r.wall);
            std::fflush(stdout);
        }
    }
    std::printf("\nideal: wall/wall1 == 1.00, throughput == #threads, cpu/wall == #threads\n");
    std::printf("cpu/wall ~= #threads => busy but slow (frequency/power ceiling)\n");
    std::printf("cpu/wall << #threads => threads blocked (lock/contention)\n");
    return 0;
}
