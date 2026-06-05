// Fast sanity check for the benchmark runner (CTest, < 5 seconds).
//
// Gates: linear and exponential must recover — these are proven fast.
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

    // Linear: tiny parameters, known to converge in < 0.2 s.
    {
        SearchOptions opts;
        opts.population_size = 200;
        opts.generations = 30;
        opts.tournament_size = 4;
        opts.seed = 20240603;
        opts.target_loss = 1e-10;
        opts.simplify_expressions = true;

        const BenchmarkResult r = run_single(problem_linear(), opts);
        std::printf("linear:      loss=%.3e  complexity=%d  %s\n",
                    r.loss, r.complexity, r.expression.c_str());
        CHECK(r.recovered);
    }

    // Exponential: medium parameters, known to converge in < 0.5 s.
    {
        SearchOptions opts;
        opts.population_size = 400;
        opts.generations = 60;
        opts.tournament_size = 4;
        opts.seed = 7;
        opts.target_loss = 1e-10;
        opts.simplify_expressions = true;

        const BenchmarkResult r = run_single(problem_exponential(), opts);
        std::printf("exponential: loss=%.3e  complexity=%d  %s\n",
                    r.loss, r.complexity, r.expression.c_str());
        CHECK(r.recovered);
    }

    // Runner smoke test: verify Nguyen-1 runs without crashing and the runner
    // interface is fully exercised. No assertion on recovery (too slow for CI).
    {
        SearchOptions opts;
        opts.population_size = 50;
        opts.generations = 5;
        opts.seed = 99;
        opts.simplify_expressions = true;

        const BenchmarkResult r = run_single(problem_nguyen1(), opts);
        std::printf("nguyen1 smoke: loss=%.3e  complexity=%d\n",
                    r.loss, r.complexity);
        CHECK(!r.expression.empty());  // runner produced an expression
    }

    if (g_failures == 0) {
        std::printf("All %d gate checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d gate checks FAILED\n", g_failures, g_checks);
    return 1;
}
