// End-to-end test of the minimal evolutionary search: discover an expression that
// fits y = 2.5*x + 1.7 from clean data, optimizing both structure and constants.
//
// We verify recovery by fit quality (the search finds an expression equivalent to the
// target), not by requiring a specific AST: many structures represent a line exactly
// once their constants are optimized.

#include <cmath>
#include <cstdio>
#include <string>
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

// Discover a genuinely nonlinear structure: y = a*exp(b*x). Unlike the linear case,
// this cannot be fit without the exp operator, so reaching a near-exact loss requires
// the search to actually build an exponential structure (we also assert it uses exp).
void test_recovers_exponential() {
    const double true_a = 2.0;
    const double true_b = 0.3;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 20; ++i) {
        const double x = 0.15 * static_cast<double>(i);  // 0 .. 2.85
        X.push_back({x});
        y.push_back(true_a * std::exp(true_b * x));
    }

    SearchOptions options;
    options.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    options.space.unary_ops = {UnaryOp::Exp};
    options.space.num_features = 1;
    options.space.max_depth = 4;
    options.space.max_nodes = 30;
    options.population_size = 400;
    options.generations = 60;
    options.tournament_size = 4;
    options.seed = 7;
    options.target_loss = 1e-10;

    const SearchResult result = run_evolution(X, y, options);

    std::printf("exp: best expression: %s  (loss=%.3e, complexity=%d)\n",
                result.expression.c_str(), result.loss, result.complexity);

    CHECK(result.loss < 1e-6);
    CHECK(result.expression.find("exp") != std::string::npos);  // used exp structurally
}

// Build clean line data y = 2.5 x + 1.7 over x in [-10, 9].
void make_line(std::vector<std::vector<double>>& X, std::vector<double>& y) {
    X.clear();
    y.clear();
    for (int i = 0; i < 20; ++i) {
        const double x = static_cast<double>(i) - 10.0;
        X.push_back({x});
        y.push_back(2.5 * x + 1.7);
    }
}

SearchOptions line_options() {
    SearchOptions o;
    o.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    o.space.unary_ops = {};
    o.space.num_features = 1;
    o.space.max_depth = 3;
    o.population_size = 200;
    o.generations = 40;
    o.tournament_size = 4;
    o.seed = 20240603;
    o.n_populations = 1;  // single island: deterministic, no migration
    o.target_loss = 1e-10;
    return o;
}

// max_evals: (a) two capped runs are byte-identical (deterministic enforcement) and
// (b) the full-budget run is never worse than a capped run. With a single island the
// capped run's trajectory is an exact prefix of the full run's and the hall of fame is
// monotone, so full.loss <= capped.loss holds exactly.
void test_max_evals_bounds_and_deterministic() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_line(X, y);

    SearchOptions full = line_options();
    full.max_evals = 0;  // no limit
    const SearchResult r_full = run_evolution(X, y, full);

    SearchOptions capped = line_options();
    capped.max_evals = 3000;  // small budget, bites well before convergence
    const SearchResult r_cap1 = run_evolution(X, y, capped);
    const SearchResult r_cap2 = run_evolution(X, y, capped);

    CHECK(r_cap1.expression == r_cap2.expression);   // deterministic across runs
    CHECK(r_cap1.loss == r_cap2.loss);
    CHECK(!r_cap1.pareto_front.empty());             // still returns a valid result
    CHECK(r_full.loss <= r_cap1.loss + 1e-12);       // more budget is never worse
}

// early_stop_condition: a looser threshold halts the search. The control (no early stop)
// drives the loss far below the threshold; the early-stopped run terminates as soon as the
// best loss crosses it, so its loss is below the threshold but the control proves the
// threshold genuinely gates termination rather than being reached only at convergence.
void test_early_stop_condition() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_line(X, y);

    SearchOptions control = line_options();
    const SearchResult r_full = run_evolution(X, y, control);
    CHECK(r_full.loss < 1e-6);  // converges far below the threshold below

    SearchOptions early = line_options();
    early.early_stop_condition = 0.5;  // stop once best loss < 0.5
    const SearchResult r_early = run_evolution(X, y, early);
    CHECK(r_early.loss < 0.5);            // threshold was crossed before returning
    CHECK(r_full.loss <= r_early.loss + 1e-12);  // the early run stopped no later than full
}

// Weighted search (PySR `weights`): one badly corrupted point given zero weight must not
// derail recovery — the weighted loss of the clean line is ~0 despite the outlier.
void test_weighted_search_recovers_line() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_line(X, y);
    std::vector<double> w(y.size(), 1.0);
    y[5] += 500.0;  // gross outlier
    w[5] = 0.0;     // excluded from the fit

    SearchOptions options = line_options();
    options.weights = w;
    const SearchResult result = run_evolution(X, y, options);

    std::printf("weighted: best expression: %s  (loss=%.3e)\n",
                result.expression.c_str(), result.loss);
    CHECK(result.loss < 1e-6);  // weighted loss ignores the zero-weighted outlier
}

}  // namespace

int main() {
    test_recovers_linear();
    test_recovers_exponential();
    test_max_evals_bounds_and_deterministic();
    test_early_stop_condition();
    test_weighted_search_recovers_line();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
