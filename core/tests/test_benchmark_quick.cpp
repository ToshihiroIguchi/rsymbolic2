// Fast sanity check for the benchmark runner (CTest, < 30 seconds).
//
// Gates: linear and exponential must recover in at least 1 of 3 runs when judged on
// held-out + extrapolation points. These are proven fast and structurally recoverable.
// Harder problems (Nguyen-1/5/7) belong in the standalone benchmark_main, not here.

#include <cstdio>

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/benchmark/runner.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s (%s:%d)\n", expr, file, line);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

}  // namespace

int main() {
    using namespace rsymbolic;

    // Linear: 3 runs, small params. All 3 should recover even on extrapolation points.
    {
        SearchOptions opts;
        opts.population_size = 200;
        opts.generations = 30;
        opts.tournament_size = 4;
        opts.target_loss = 1e-10;
        opts.simplify_expressions = true;

        const BenchmarkResult r = run_repeated(problem_linear(), opts, 3, 20240603ULL);
        std::printf("linear:      %zu/%zu recovered  medTestErr=%.3e  bestCmpx=%d\n",
                    r.n_recovered, r.n_runs, r.median_test_error, r.best_complexity);
        std::printf("  best: %s\n", r.best_expression.c_str());
        CHECK(r.n_recovered >= 1);
    }

    // Exponential: 3 runs, medium params. At least 1 should recover on extrapolation.
    {
        SearchOptions opts;
        opts.population_size = 400;
        opts.generations = 60;
        opts.tournament_size = 4;
        opts.target_loss = 1e-10;
        opts.simplify_expressions = true;

        const BenchmarkResult r = run_repeated(problem_exponential(), opts, 3, 7ULL);
        std::printf("exponential: %zu/%zu recovered  medTestErr=%.3e  bestCmpx=%d\n",
                    r.n_recovered, r.n_runs, r.median_test_error, r.best_complexity);
        std::printf("  best: %s\n", r.best_expression.c_str());
        CHECK(r.n_recovered >= 1);
    }

    // Runner smoke test: verify Nguyen-1 runs without crashing and the runner
    // interface is fully exercised. No assertion on recovery (too slow for CI).
    {
        SearchOptions opts;
        opts.population_size = 50;
        opts.generations = 5;
        opts.target_loss = 1e-10;
        opts.simplify_expressions = true;

        const RunOutcome o = run_single(problem_nguyen1(), opts);
        std::printf("nguyen1 smoke: trainLoss=%.3e  testErr=%.3e  cmpx=%d\n",
                    o.train_loss, o.test_error, o.complexity);
        CHECK(!o.expression.empty());
    }

    // Multi-variable smoke test: 2-feature problem must run without crashing.
    // No recovery assertion (too slow for CI; recovery is verified in benchmark_main).
    {
        SearchOptions opts;
        opts.population_size = 50;
        opts.generations = 5;
        opts.target_loss = 1e-10;
        opts.simplify_expressions = true;

        const RunOutcome o = run_single(problem_multivar(), opts);
        std::printf("multivar smoke: trainLoss=%.3e  testErr=%.3e  cmpx=%d\n",
                    o.train_loss, o.test_error, o.complexity);
        CHECK(!o.expression.empty());
    }

    if (g_failures == 0) {
        std::printf("All %d gate checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d gate checks FAILED\n", g_failures, g_checks);
    return 1;
}
