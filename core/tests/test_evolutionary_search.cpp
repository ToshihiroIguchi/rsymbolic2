// End-to-end test of the minimal evolutionary search: discover an expression that
// fits y = 2.5*x + 1.7 from clean data, optimizing both structure and constants.
//
// We verify recovery by fit quality (the search finds an expression equivalent to the
// target), not by requiring a specific AST: many structures represent a line exactly
// once their constants are optimized.

#include <cmath>
#include <cstdio>
#include <vector>

#include "rsymbolic/search/evolutionary_search.hpp"

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

using namespace rsymbolic;

void test_recovers_linear() {
    const double true_a = 2.5;
    const double true_b = 1.7;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 20; ++i) {
        const double x = static_cast<double>(i) - 10.0;  // -10 .. 9
        X.push_back({x});
        y.push_back(true_a * x + true_b);
    }

    SearchOptions options;
    options.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    options.space.unary_ops = {};
    options.space.num_features = 1;
    options.space.max_depth = 3;
    options.population_size = 200;
    options.generations = 20;
    options.tournament_size = 4;
    options.seed = 20240603;
    options.target_loss = 1e-10;

    const SearchResult result = run_evolution(X, y, options);

    std::printf("best expression: %s  (loss=%.3e, complexity=%d)\n",
                result.expression.c_str(), result.loss, result.complexity);
    std::printf("pareto front size: %zu\n", result.pareto_front.size());

    // Found an expression that fits the linear data essentially exactly.
    CHECK(result.loss < 1e-6);
    CHECK(result.complexity >= 1);
    CHECK(!result.expression.empty());
    CHECK(!result.pareto_front.empty());
}

}  // namespace

int main() {
    test_recovers_linear();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
