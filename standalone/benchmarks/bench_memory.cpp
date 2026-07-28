// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Peak-memory probe for a full search (docs/65).
//
// The search's working set is dominated by O(rows) terms whose multiplier is the number
// of ISLANDS, not the number of threads, so a probe run at n_populations=1 understates the
// real ceiling several-fold (docs/59 §3 records that exact mistake). This probe therefore
// takes the population count and the row count as explicit arguments and reports the
// process peak working set, so before/after numbers are comparable.
//
// Usage (from the build directory):
//   ./standalone/bench_memory                        # 100000 rows, 5 features, 31 pops
//   ./standalone/bench_memory 200000 10 31 8 28      # rows features pops threads gens
//
// The default operator set (+ - *) with maxsize 30 produces trees carrying roughly half a
// dozen constants within the first epoch, which is what sizes the LM Jacobian buffer — the
// term docs/59's fitted formula does not represent. Keep `generations` >= one migration
// interval (28) so at least one constant-optimisation pass runs; the buffers reach their
// high-water mark almost immediately and do not grow with a longer run.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <limits>
#include <string>
#include <vector>

#if defined(_WIN32)
#   include <windows.h>
#   include <psapi.h>
#elif defined(__linux__)
#   include <cstring>
#endif

#include "rsymbolic/expression/least_squares_problem.hpp"  // columns_from_rows
#include "rsymbolic/search/evolutionary_search.hpp"

namespace {

// Process peak working set / high-water RSS in bytes; 0 when unavailable.
std::size_t peak_rss_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return static_cast<std::size_t>(pmc.PeakWorkingSetSize);
    return 0;
#elif defined(__linux__)
    // VmHWM is the kernel's peak resident-set high-water mark for this process.
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            kb = static_cast<std::size_t>(std::strtoull(line + 6, nullptr, 10));
            break;
        }
    }
    std::fclose(f);
    return kb * 1024;
#else
    return 0;
#endif
}

double mib(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

// Deterministic synthetic data. A plain LCG rather than <random> so the dataset is
// identical on every platform and standard library — this probe is compared across
// builds and machines.
void make_data(std::size_t m, std::size_t p,
               std::vector<std::vector<double>>& X, std::vector<double>& y) {
    std::uint64_t s = 0x243f6a8885a308d3ULL;
    const auto next = [&s]() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        // Top 53 bits -> [0, 1).
        return static_cast<double>(s >> 11) / 9007199254740992.0;
    };
    X.assign(m, std::vector<double>(p));
    y.assign(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < p; ++j) X[i][j] = -2.0 + 4.0 * next();
        const double x0 = X[i][0];
        const double x1 = p > 1 ? X[i][1] : 1.0;
        y[i] = 1.7 * x0 * x1 - 0.4 * x0 + 2.3;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::size_t rows     = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100000;
    const std::size_t features = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 5;
    const std::size_t pops     = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 31;
    const int         threads  = argc > 4 ? std::atoi(argv[4]) : 0;
    const std::size_t gens     = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 28;
    // 0 = the row-major convenience entry point (what the tests and benchmark problems
    // use; the engine transposes, so a second full copy of X exists during the run).
    // 1 = the column-major entry point, which is what the R/Python/WASM bindings use:
    // the caller's columns are moved into the Dataset and no row-major copy is ever made.
    const int         layout   = argc > 6 ? std::atoi(argv[6]) : 0;

    std::printf("rows=%zu features=%zu n_populations=%zu n_threads=%d generations=%zu "
                "layout=%s\n",
                rows, features, pops, threads, gens,
                layout ? "column-major (binding path)" : "row-major (convenience path)");
    std::fflush(stdout);

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_data(rows, features, X, y);

    const std::size_t after_data = peak_rss_bytes();
    std::printf("peak after data build : %8.1f MiB\n", mib(after_data));
    std::fflush(stdout);

    rsymbolic::SearchOptions opts;
    opts.space.num_features = static_cast<int>(features);
    opts.n_populations      = pops;
    opts.n_threads          = threads;
    opts.generations        = gens;
    // Never stop early: the point is to run the full budget so every buffer reaches its
    // high-water mark. -inf also keeps the run insensitive to how well the search does.
    opts.target_loss        = -std::numeric_limits<double>::infinity();
    opts.seed               = 42;
    opts.verbosity          = 0;

    const auto t0 = std::chrono::steady_clock::now();
    rsymbolic::SearchResult res;
    if (layout) {
        // Mirror a binding: transpose once into columns, release the row-major source,
        // then hand the columns to the engine.
        rsymbolic::FeatureColumns cols{rsymbolic::columns_from_rows(X)};
        std::vector<std::vector<double>>().swap(X);
        res = rsymbolic::run_evolution(std::move(cols), y, opts);
    } else {
        res = rsymbolic::run_evolution(X, y, opts);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();

    const std::size_t peak = peak_rss_bytes();

    std::printf("peak after search     : %8.1f MiB\n", mib(peak));
    std::printf("attributable to search: %8.1f MiB\n", mib(peak - after_data));
    std::printf("bytes per row (total) : %8.1f\n",
                static_cast<double>(peak) / static_cast<double>(rows));
    std::printf("wall                  : %8.2f s\n", wall);
    std::printf("best loss             : %.10g   complexity=%d\n",
                res.loss, res.complexity);
    std::printf("expression            : %s\n", res.expression.c_str());
    std::fflush(stdout);
    return 0;
}
