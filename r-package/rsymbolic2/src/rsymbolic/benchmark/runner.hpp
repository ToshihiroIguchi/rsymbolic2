#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

namespace rsymbolic {

// Outcome of one search run against a problem.
struct RunOutcome {
    double train_loss = 0.0;
    double test_error = 0.0;   // max |pred - y_test| over held-out/extrapolation set
    int complexity = 0;
    std::string expression;
    bool recovered = false;    // test_error < problem.recovery_threshold
    double time_ms = 0.0;
};

// Aggregate result over N repeated runs (varying seeds).
struct BenchmarkResult {
    std::string problem_name;
    std::size_t n_runs = 0;
    std::size_t n_recovered = 0;
    double recovery_rate = 0.0;       // n_recovered / n_runs
    double median_train_loss = 0.0;
    double median_test_error = 0.0;
    double min_test_error = 0.0;
    double max_test_error = 0.0;
    int best_complexity = 0;          // complexity of the run with lowest test error
    std::string best_expression;
    double median_time_ms = 0.0;
    double total_time_ms = 0.0;
};

// Run a single problem once and return the detailed outcome.
// options.space is overwritten with problem.space; seed is taken from options.seed.
RunOutcome run_single(const BenchmarkProblem& problem, SearchOptions options);

// Run the problem n_runs times using seeds base_seed, base_seed+1, …, and aggregate.
BenchmarkResult run_repeated(const BenchmarkProblem& problem,
                              SearchOptions options,
                              std::size_t n_runs,
                              std::uint64_t base_seed);

// Run each problem independently with run_repeated and collect results.
std::vector<BenchmarkResult> run_suite(
    const std::vector<BenchmarkProblem>& problems,
    const SearchOptions& options,
    std::size_t n_runs,
    std::uint64_t base_seed);

// Print a human-readable table to stdout.
void print_results(const std::vector<BenchmarkResult>& results);

}  // namespace rsymbolic
