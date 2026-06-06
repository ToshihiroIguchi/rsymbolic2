// Strong-scaling benchmark for the island model.
//
// Fixed work: n_populations=12 always. Only omp_num_threads varies: {1,2,3,4,6}.
// 12 divides evenly into each of these thread counts, so every thread gets
// exactly the same number of islands — no load-imbalance artifact.
//
// Speedup(k)    = T(1) / T(k)              (T = wall time, same total work)
// Efficiency(k) = Speedup(k) / k
// Gate          : Efficiency >= 0.7  (roadmap: speedup >= 0.7 × n_threads)
//
// Why strong scaling (fixed work, vary threads)?
//   Weak scaling (vary both islands and threads together) conflates parallelism
//   quality with "doing more work takes longer" — it cannot isolate thread
//   overhead. Speedup in the roadmap is the classical strong-scaling ratio
//   T(1)/T(k), so that is what we measure.
//
// Why n_populations=12?
//   12 = 1×12 = 2×6 = 3×4 = 4×3 = 6×2 — perfect divisibility into the tested
//   thread counts, eliminating unequal work distribution as a confound.
//   Physical cores on this machine: 10 (logical: 12), so testing ≤ 6 threads
//   stays within physical-core territory.
//
// Hardware note: Intel hybrid (P+E) cores may cause efficiency < 1.0 even with
//   perfect parallel code. Any shortfall must be attributed to hardware
//   heterogeneity before claiming an algorithmic issue.
//
// Usage (from build directory):
//   ./standalone/bench_parallel          # default: n_runs=3
//   ./standalone/bench_parallel 5        # n_runs=5
//
// First line printed for each (n_islands=12, threads=t) block is the result;
// stdout is flushed after every line so progress is visible without buffering.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

int main(int argc, char* argv[]) {
    const int n_runs = argc > 1 ? std::atoi(argv[1]) : 3;

    // Nguyen-1 (x^3+x^2+x): polynomial, no transcendentals → per-eval cost
    // moderate and stable. target_loss=-inf forces the full pop×gen budget on
    // every island regardless of fit quality, making work truly fixed.
    const auto prob = rsymbolic::problem_nguyen1();

    rsymbolic::SearchOptions opts;
    opts.space                = prob.space;
    opts.population_size      = 100;
    opts.generations          = 25;
    opts.tournament_size      = 4;
    opts.target_loss          = -std::numeric_limits<double>::infinity();
    opts.simplify_expressions = false;  // skip simplify: constant cost per eval
    opts.migration_interval   = 10;
    opts.migration_size       = 5;
    opts.n_populations        = 12;     // FIXED for strong scaling

    // Thread counts to test: all proper divisors of 12 that are ≤ physical cores.
    const std::vector<int> thread_counts = {1, 2, 3, 4, 6};

#ifdef _OPENMP
    std::printf("OpenMP : enabled\n");
#else
    std::printf("OpenMP : disabled (serial fallback — all rows will be identical)\n");
#endif
    std::printf("Problem: %s\n", prob.name.c_str());
    std::printf("Config : n_populations=12  pop=%d  gen=%d  "
                "migration_interval=%zu  migration_size=%zu\n",
                100, 25, opts.migration_interval, opts.migration_size);
    std::printf("Runs   : %d per thread count\n\n", n_runs);
    std::fflush(stdout);

    std::printf("%-10s %-12s %-12s %-12s %-12s  %s\n",
                "threads", "med_ms", "min_ms", "max_ms",
                "efficiency", "gate");
    std::printf("%-10s %-12s %-12s %-12s %-12s  %s\n",
                "-------", "------", "------", "------",
                "----------", "----");
    std::fflush(stdout);

    double t1_median = 0.0;

    for (const int nthreads : thread_counts) {
#ifdef _OPENMP
        omp_set_num_threads(nthreads);
#endif
        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(n_runs));

        for (int r = 0; r < n_runs; ++r) {
            opts.seed = static_cast<std::uint64_t>(42 + r);

            const auto t0 = std::chrono::steady_clock::now();
            run_evolution(prob.X, prob.y, opts);
            const auto t1 = std::chrono::steady_clock::now();

            times.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        std::sort(times.begin(), times.end());
        const double med = times[static_cast<std::size_t>(n_runs / 2)];
        const double mn  = times.front();
        const double mx  = times.back();

        if (nthreads == 1) t1_median = med;

        const double speedup = (t1_median > 0.0) ? t1_median / med : 1.0;
        const double eff     = speedup / static_cast<double>(nthreads);
        const bool   pass    = (nthreads == 1) || (eff >= 0.7);

        std::printf("%-10d %-12.0f %-12.0f %-12.0f %-12.3f  %s\n",
                    nthreads, med, mn, mx, eff,
                    pass ? "PASS" : "FAIL (< 0.70)");
        std::fflush(stdout);
    }

    std::printf("\nNote: efficiency < 1.0 on Intel hybrid (P+E) cores may reflect\n");
    std::printf("      core-speed heterogeneity, not parallelism overhead alone.\n");
    std::fflush(stdout);

    return 0;
}
