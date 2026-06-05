#include "rsymbolic/benchmark/runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

namespace {

// Evaluate max absolute error on a held-out set using the fitted tree.
// Returns +inf if any prediction is non-finite (domain error counts as failure).
double compute_test_error(const Tree& tree,
                          const std::vector<std::vector<double>>& X_test,
                          const std::vector<double>& y_test) {
    if (X_test.empty()) return 0.0;
    const std::vector<double> c = initial_constants(tree);
    double max_err = 0.0;
    for (std::size_t i = 0; i < X_test.size(); ++i) {
        const double pred = evaluate<double>(tree, X_test[i].data(), c.data());
        if (!std::isfinite(pred)) return std::numeric_limits<double>::infinity();
        const double err = std::abs(pred - y_test[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size() / 2;
    return (v.size() % 2 == 1) ? v[m] : (v[m - 1] + v[m]) * 0.5;
}

}  // namespace

RunOutcome run_single(const BenchmarkProblem& problem, SearchOptions options) {
    options.space = problem.space;

    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult sr = run_evolution(problem.X, problem.y, options);
    const auto t1 = std::chrono::steady_clock::now();

    RunOutcome out;
    out.train_loss = sr.loss;
    out.test_error = compute_test_error(sr.tree, problem.X_test, problem.y_test);
    out.complexity = sr.complexity;
    out.expression = sr.expression;
    out.recovered = out.test_error < problem.recovery_threshold;
    out.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return out;
}

BenchmarkResult run_repeated(const BenchmarkProblem& problem,
                              SearchOptions options,
                              std::size_t n_runs,
                              std::uint64_t base_seed) {
    std::vector<RunOutcome> outcomes;
    outcomes.reserve(n_runs);
    for (std::size_t i = 0; i < n_runs; ++i) {
        options.seed = base_seed + i;
        outcomes.push_back(run_single(problem, options));
    }

    BenchmarkResult r;
    r.problem_name = problem.name;
    r.n_runs = n_runs;

    std::vector<double> train_losses, test_errors, times;
    double best_test = std::numeric_limits<double>::infinity();
    double min_test = std::numeric_limits<double>::infinity();
    double max_test = 0.0;
    for (const auto& o : outcomes) {
        if (o.recovered) ++r.n_recovered;
        train_losses.push_back(o.train_loss);
        test_errors.push_back(o.test_error);
        times.push_back(o.time_ms);
        if (o.test_error < best_test) {
            best_test = o.test_error;
            r.best_complexity = o.complexity;
            r.best_expression = o.expression;
        }
        if (o.test_error < min_test) min_test = o.test_error;
        if (std::isfinite(o.test_error) && o.test_error > max_test) max_test = o.test_error;
    }
    r.recovery_rate = static_cast<double>(r.n_recovered) / static_cast<double>(n_runs);
    r.median_train_loss = median_of(train_losses);
    r.median_test_error = median_of(test_errors);
    r.min_test_error = min_test;
    r.max_test_error = max_test;
    r.median_time_ms = median_of(times);
    for (const auto& o : outcomes) r.total_time_ms += o.time_ms;
    return r;
}

std::vector<BenchmarkResult> run_suite(
    const std::vector<BenchmarkProblem>& problems,
    const SearchOptions& options,
    std::size_t n_runs,
    std::uint64_t base_seed) {
    std::vector<BenchmarkResult> results;
    results.reserve(problems.size());
    for (const auto& p : problems) {
        results.push_back(run_repeated(p, options, n_runs, base_seed));
    }
    return results;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    std::printf("\n%-35s  %5s  %12s  %12s  %12s  %5s  %8s\n",
                "Problem", "Recov", "MedTestErr", "MinTestErr", "MaxTestErr",
                "Cmpx", "MedTime");
    const std::string sep(35 + 2 + 5 + 2 + 12 + 2 + 12 + 2 + 12 + 2 + 5 + 2 + 8, '-');
    std::printf("%s\n", sep.c_str());

    std::size_t total_recovered = 0;
    std::size_t total_runs = 0;
    for (const auto& r : results) {
        total_recovered += r.n_recovered;
        total_runs += r.n_runs;

        char recov_buf[16];
        std::snprintf(recov_buf, sizeof(recov_buf), "%zu/%zu",
                      r.n_recovered, r.n_runs);

        const char* min_fmt = std::isfinite(r.min_test_error) ? "%-12.3e" : "%-12s";
        const char* max_fmt = std::isfinite(r.max_test_error) ? "%-12.3e" : "%-12s";

        std::printf("%-35s  %-5s  %-12.3e  ",
                    r.problem_name.c_str(), recov_buf, r.median_test_error);
        if (std::isfinite(r.min_test_error))
            std::printf("%-12.3e  ", r.min_test_error);
        else
            std::printf("%-12s  ", "inf");
        if (std::isfinite(r.max_test_error))
            std::printf("%-12.3e  ", r.max_test_error);
        else
            std::printf("%-12s  ", "inf");
        std::printf("%-5d  %-8.1f\n", r.best_complexity, r.median_time_ms);

        // Best expression on next line, indented.
        std::string expr = r.best_expression;
        if (expr.size() > 60) expr = expr.substr(0, 57) + "...";
        std::printf("  best: %s\n", expr.c_str());
        (void)min_fmt; (void)max_fmt;
    }
    const double overall_rate = total_runs > 0
        ? static_cast<double>(total_recovered) / static_cast<double>(total_runs) : 0.0;
    std::printf("\nRecovery: %zu / %zu runs  (%.0f%%)\n\n",
                total_recovered, total_runs, overall_rate * 100.0);
}

}  // namespace rsymbolic
