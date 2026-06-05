#pragma once

#include <string>
#include <vector>

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

namespace rsymbolic {

// Result of a single benchmark run.
struct BenchmarkResult {
    std::string problem_name;
    bool recovered = false;       // loss < recovery_threshold
    double loss = 0.0;
    int complexity = 0;
    std::string expression;
    double time_ms = 0.0;
    std::size_t evaluations = 0;  // total optimizer evaluations (if available later)
};

// Run a single problem once and return the result. `options.space` is overwritten
// with `problem.space` so the problem controls the operator set; all other fields in
// `options` (population, generations, seed, …) come from the caller.
BenchmarkResult run_single(const BenchmarkProblem& problem, SearchOptions options);

// Run each problem independently and collect results.
std::vector<BenchmarkResult> run_suite(
    const std::vector<BenchmarkProblem>& problems, const SearchOptions& options);

// Print a human-readable table to stdout.
void print_results(const std::vector<BenchmarkResult>& results);

}  // namespace rsymbolic
