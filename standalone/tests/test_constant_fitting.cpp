// End-to-end walking-skeleton test: build a least-squares problem from an expression
// tree + dataset (residuals and dual-number Jacobian), then recover the constants with
// the Levenberg-Marquardt backend. Two cases:
//   1. y = a*x + b        (linear)
//   2. y = a*exp(b*x)     (nonlinear)

#include <cmath>
#include <cstdio>
#include <vector>

#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"
#include "rsymbolic/optimization/self_lm_optimizer.hpp"

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

bool close(double a, double b, double tol) { return std::fabs(a - b) < tol; }

using namespace rsymbolic;

Tree linear_tree() {
    return {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul),
            constant_node(1, 0.0), binary_node(BinaryOp::Add)};
}

Tree exp_tree() {
    return {constant_node(0, 1.0), constant_node(1, 0.1), variable_node(0),
            binary_node(BinaryOp::Mul), unary_node(UnaryOp::Exp),
            binary_node(BinaryOp::Mul)};
}

// Case 1: recover a = 2.5, b = 1.7 from y = a*x + b.
void test_fit_linear() {
    const double true_a = 2.5;
    const double true_b = 1.7;

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(true_a * x + true_b);
    }

    OptimizationProblem problem =
        make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0});

    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(res.constants.size() == 2);
    CHECK(close(res.constants[0], true_a, 1e-4));
    CHECK(close(res.constants[1], true_b, 1e-4));
    CHECK(res.loss < 1e-6);
    CHECK(res.evaluations > 0);
}

// Case 2: recover a = 2.0, b = 0.3 from y = a*exp(b*x).
void test_fit_exponential() {
    const double true_a = 2.0;
    const double true_b = 0.3;

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = 0.5 * static_cast<double>(i);  // 0.0 .. 4.5
        X.push_back({x});
        y.push_back(true_a * std::exp(true_b * x));
    }

    // Initial guess from the tree's stored values (a=1.0, b=0.1).
    OptimizationProblem problem = make_least_squares_problem(exp_tree(), X, y);

    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(res.constants.size() == 2);
    CHECK(close(res.constants[0], true_a, 1e-3));
    CHECK(close(res.constants[1], true_b, 1e-3));
    CHECK(res.loss < 1e-6);
}

// The analytic (dual) Jacobian and the numerical Jacobian should both converge to the
// same linear fit, confirming the analytic path is consistent.
void test_analytic_vs_numerical_jacobian() {
    const double true_a = 2.5;
    const double true_b = 1.7;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(true_a * x + true_b);
    }

    OptimizationProblem analytic =
        make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0});

    OptimizationProblem numeric = analytic;
    numeric.jacobian = nullptr;  // force NumericalDiff path

    SelfLMOptimizer opt;
    const OptimizationResult a = opt.optimize(analytic);
    const OptimizationResult n = opt.optimize(numeric);

    CHECK(a.success);
    CHECK(n.success);
    CHECK(close(a.constants[0], n.constants[0], 1e-4));
    CHECK(close(a.constants[1], n.constants[1], 1e-4));
}

// Weighted least squares (PySR `weights`): a corrupted point given zero weight must not
// affect the recovered line. This exercises the sqrt(w) scaling in BOTH the residual and
// the analytic Jacobian closures (the default path), since dropping the point from only
// one would leave a residual the optimiser still chases.
void test_weighted_fit_ignores_zero_weight_point() {
    const double true_a = 2.5;
    const double true_b = 1.7;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::vector<double> w;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(true_a * x + true_b);
        w.push_back(1.0);
    }
    // Corrupt the last point and zero its weight so it is excluded from the fit.
    y[9] += 1000.0;
    w[9] = 0.0;

    OptimizationProblem problem =
        make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0}, {}, w);
    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(close(res.constants[0], true_a, 1e-4));
    CHECK(close(res.constants[1], true_b, 1e-4));
    CHECK(res.loss < 1e-6);  // the zero-weighted corruption contributes nothing
}

// Halving every weight halves the weighted SSE but leaves the optimum unchanged: the
// recovered constants must be identical, confirming weights act as a pure per-point scale.
void test_weighted_fit_uniform_scale() {
    const double true_a = 2.5;
    const double true_b = 1.7;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(true_a * x + true_b + (i == 4 ? 0.3 : 0.0));  // mild noise so loss > 0
    }
    std::vector<double> w_half(10, 0.5), w_one(10, 1.0);

    SelfLMOptimizer opt;
    const OptimizationResult a =
        opt.optimize(make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0}, {}, w_one));
    const OptimizationResult b =
        opt.optimize(make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0}, {}, w_half));

    CHECK(a.success && b.success);
    CHECK(close(a.constants[0], b.constants[0], 1e-6));
    CHECK(close(a.constants[1], b.constants[1], 1e-6));
    CHECK(close(b.loss, 0.5 * a.loss, 1e-6));  // weighted SSE scales with the weights
}

// The backend must also be reachable through the factory.
void test_via_factory() {
    const double true_a = 2.5;
    const double true_b = 1.7;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(true_a * x + true_b);
    }
    OptimizationProblem problem =
        make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0});

    auto opt = OptimizerFactory::create(OptimizerType::SelfLM);
    CHECK(opt != nullptr);
    const OptimizationResult res = opt->optimize(problem);
    CHECK(res.success);
    CHECK(close(res.constants[0], true_a, 1e-4));
    CHECK(close(res.constants[1], true_b, 1e-4));
}

}  // namespace

int main() {
    test_fit_linear();
    test_fit_exponential();
    test_analytic_vs_numerical_jacobian();
    test_weighted_fit_ignores_zero_weight_point();
    test_weighted_fit_uniform_scale();
    test_via_factory();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
