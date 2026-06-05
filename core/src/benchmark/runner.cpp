#include "rsymbolic/benchmark/runner.hpp"

#include <chrono>
#include <cstdio>

namespace rsymbolic {

BenchmarkResult run_single(const BenchmarkProblem& problem, SearchOptions options) {
    options.space = problem.space;  // problem owns the operator set

    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult sr = run_evolution(problem.X, problem.y, options);
    const auto t1 = std::chrono::steady_clock::now();

    BenchmarkResult r;
    r.problem_name = problem.name;
    r.recovered = sr.loss < problem.recovery_threshold;
    r.loss = sr.loss;
    r.complexity = sr.complexity;
    r.expression = sr.expression;
    r.time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r;
}

std::vector<BenchmarkResult> run_suite(
    const std::vector<BenchmarkProblem>& problems, const SearchOptions& options) {
    std::vector<BenchmarkResult> results;
    results.reserve(problems.size());
    for (const auto& p : problems) {
        results.push_back(run_single(p, options));
    }
    return results;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    // Header
    std::printf("\n%-35s %-10s %-12s %-5s %-10s  %s\n",
                "Problem", "Status", "Loss", "Cmpx", "Time(ms)", "Expression");
    std::printf("%-35s %-10s %-12s %-5s %-10s  %s\n",
                std::string(35, '-').c_str(), std::string(10, '-').c_str(),
                std::string(12, '-').c_str(), std::string(5, '-').c_str(),
                std::string(10, '-').c_str(), std::string(30, '-').c_str());

    int n_recovered = 0;
    for (const auto& r : results) {
        if (r.recovered) ++n_recovered;
        const char* status = r.recovered ? "RECOVERED" : "failed";
        // Truncate expression to 40 chars for display.
        std::string expr = r.expression;
        if (expr.size() > 40) expr = expr.substr(0, 37) + "...";
        std::printf("%-35s %-10s %-12.3e %-5d %-10.1f  %s\n",
                    r.problem_name.c_str(), status, r.loss, r.complexity,
                    r.time_ms, expr.c_str());
    }
    std::printf("\nRecovery: %d / %d\n\n", n_recovered,
                static_cast<int>(results.size()));
}

}  // namespace rsymbolic
