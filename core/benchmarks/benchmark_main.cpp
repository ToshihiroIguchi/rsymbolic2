// Standalone benchmark runner. Build with CMake (RSYMBOLIC2_BUILD_BENCHMARKS=ON) and
// run manually; it is NOT wired to CTest to keep the default test suite fast.
//
// Usage (from build directory):
//   ./core/benchmark_main
//   ./core/benchmark_main 600 100  # pop_size generations
//
// Output: a recovery table comparable to those reported for PySR / Operon.

#include <cstdlib>
#include <cstdio>

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/benchmark/runner.hpp"

int main(int argc, char* argv[]) {
    rsymbolic::SearchOptions options;
    options.population_size =
        argc > 1 ? static_cast<std::size_t>(std::atoi(argv[1])) : 600;
    options.generations =
        argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 100;
    options.tournament_size = 5;
    options.seed = 42;
    options.target_loss = 1e-10;
    options.simplify_expressions = true;

    std::printf("rsymbolic2 benchmark  (pop=%zu  gen=%zu  seed=%llu)\n",
                options.population_size, options.generations,
                static_cast<unsigned long long>(options.seed));

    const auto problems = rsymbolic::standard_problems();
    const auto results = rsymbolic::run_suite(problems, options);
    rsymbolic::print_results(results);
    return 0;
}
