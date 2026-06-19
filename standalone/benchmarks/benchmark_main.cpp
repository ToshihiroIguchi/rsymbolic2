// Standalone benchmark runner. Build with CMake (RSYMBOLIC2_BUILD_BENCHMARKS=ON) and
// run manually; it is NOT wired to CTest to keep the default test suite fast.
//
// Usage (from build directory):
//   ./core/benchmark_main
//   ./core/benchmark_main 600 100             # pop_size generations (n_runs=5)
//   ./core/benchmark_main 600 100 5           # pop_size generations n_runs
//   ./core/benchmark_main 600 100 5 self_lm   # ... optimizer (eigen_lm | self_lm | ...)
//
// Output: a recovery table (k/n recovery rate + median/spread of test error) comparable
// to tables reported for PySR / Operon. Judgment is on held-out + extrapolation points,
// not training loss, so over-fitted approximations are classified as "failed".

#include <cstdlib>
#include <cstdio>
#include <string>

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/benchmark/runner.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

int main(int argc, char* argv[]) {
    rsymbolic::SearchOptions options;
    options.population_size =
        argc > 1 ? static_cast<std::size_t>(std::atoi(argv[1])) : 600;
    options.generations =
        argc > 2 ? static_cast<std::size_t>(std::atoi(argv[2])) : 100;
    const std::size_t n_runs =
        argc > 3 ? static_cast<std::size_t>(std::atoi(argv[3])) : 5;
    const std::string opt_name = argc > 4 ? argv[4] : "eigen_lm";
    options.optimizer_type = rsymbolic::OptimizerFactory::from_string(opt_name);
    const std::uint64_t base_seed = 42;

    options.tournament_size = 5;
    options.target_loss = 1e-10;
    options.simplify_expressions = true;

    std::printf("rsymbolic2 benchmark  (pop=%zu  gen=%zu  n_runs=%zu  opt=%s  base_seed=%llu)\n",
                options.population_size, options.generations, n_runs, opt_name.c_str(),
                static_cast<unsigned long long>(base_seed));
    std::printf("Recovery judged on held-out + extrapolation points (not training loss).\n");

    const auto problems = rsymbolic::standard_problems();
    const auto results = rsymbolic::run_suite(problems, options, n_runs, base_seed);
    rsymbolic::print_results(results);
    return 0;
}
