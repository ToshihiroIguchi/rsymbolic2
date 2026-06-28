// Convergence and contract tests for the allocation-free SelfLMOptimizer — the
// default and only least-squares backend. Covers raw residual problems and
// test_constant_fitting.cpp's tree + dual-number Jacobian problems against the
// in-house Levenberg-Marquardt backend. Self-contained (no test framework);
// returns 0 on success, 1 on failure.

#include <cmath>
#include <cstdio>
#include <exception>
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

constexpr double kTrueA = 2.5;
constexpr double kTrueB = 1.7;

// --- Raw residual problem (no Jacobian => forward-difference fallback) --------

struct LinearData {
    std::vector<double> x;
    std::vector<double> y;
};

LinearData make_data() {
    LinearData d;
    for (int i = 0; i < 10; ++i) {
        const double xi = static_cast<double>(i);
        d.x.push_back(xi);
        d.y.push_back(kTrueA * xi + kTrueB);
    }
    return d;
}

OptimizationProblem make_raw_problem(const LinearData& data) {
    OptimizationProblem problem;
    problem.num_residuals = data.x.size();
    problem.initial_constants = {0.0, 0.0};  // a, b
    problem.residuals = [&data](const std::vector<double>& c,
                                std::vector<double>& r) {
        const double a = c[0];
        const double b = c[1];
        for (std::size_t i = 0; i < data.x.size(); ++i)
            r[i] = (a * data.x[i] + b) - data.y[i];
    };
    return problem;
}

// Recovery of (2.5, 1.7) from noise-free data using the finite-difference fallback.
void test_recovers_linear_numeric_jacobian() {
    const LinearData data = make_data();
    const OptimizationProblem problem = make_raw_problem(data);

    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(res.constants.size() == 2);
    CHECK(close(res.constants[0], kTrueA, 1e-4));
    CHECK(close(res.constants[1], kTrueB, 1e-4));
    CHECK(res.loss < 1e-6);
    CHECK(res.evaluations > 0);
}

// Reachable through the factory under the "self_lm" name.
void test_via_factory() {
    const LinearData data = make_data();
    const OptimizationProblem problem = make_raw_problem(data);

    auto opt = OptimizerFactory::create(OptimizerType::SelfLM);
    CHECK(opt != nullptr);
    CHECK(opt->name() == "SelfLMOptimizer");
    CHECK(OptimizerFactory::from_string("self_lm") == OptimizerType::SelfLM);

    const OptimizationResult res = opt->optimize(problem);
    CHECK(res.success);
    CHECK(close(res.constants[0], kTrueA, 1e-4));
    CHECK(close(res.constants[1], kTrueB, 1e-4));
}

// Deterministic: two FRESH optimizers (same default seed) on the same problem give
// identical results. Two fresh instances rather than two calls on one instance, because
// the restart-perturbation RNG advances across optimize() calls — so determinism is
// "same seed => same result", reproduced per island in the search by per-island seeding.
void test_reproducibility() {
    const LinearData data = make_data();

    SelfLMOptimizer opt_a;
    SelfLMOptimizer opt_b;
    const OptimizationResult a = opt_a.optimize(make_raw_problem(data));
    const OptimizationResult b = opt_b.optimize(make_raw_problem(data));

    CHECK(a.constants.size() == b.constants.size());
    CHECK(a.constants[0] == b.constants[0]);
    CHECK(a.constants[1] == b.constants[1]);
    CHECK(a.loss == b.loss);
}

// --- Multi-start (SR.jl _optimize_constants parity) ---------------------------

// A 1-parameter double-well objective: SSE(c) = (c-2)^2(c-3)^2 + 0.09(c-2)^2, i.e.
// residuals r0=(c-2)(c-3), r1=0.3(c-2). Global min at c=2 (SSE=0); a local min near
// c≈2.88 (SSE≈0.08) separated by a hump near c≈2.62. Starting at c0=3.0, a single LM
// settles in the local basin; reaching the global min requires a restart that perturbs
// below the hump.
OptimizationProblem make_double_well(double c0) {
    OptimizationProblem p;
    p.num_residuals = 2;
    p.initial_constants = {c0};
    p.residuals = [](const std::vector<double>& c, std::vector<double>& r) {
        const double x = c[0];
        r[0] = (x - 2.0) * (x - 3.0);
        r[1] = 0.3 * (x - 2.0);
    };
    return p;
}

// n_restarts=0 (single start from x0=3.0) stays trapped in the local basin; multi-start
// escapes to the global minimum, and is never worse than the single start.
void test_multistart_escapes_local_min() {
    const OptimizerConfig single_cfg{0, 0, 100, 0.5};
    SelfLMOptimizer single(single_cfg);
    const OptimizationResult s = single.optimize(make_double_well(3.0));
    CHECK(s.success);
    CHECK(s.loss > 1e-3);  // trapped in the local minimum

    const OptimizerConfig multi_cfg{0, 20, 100, 0.5};
    SelfLMOptimizer multi(multi_cfg);
    const OptimizationResult m = multi.optimize(make_double_well(3.0));
    CHECK(m.success);
    CHECK(m.loss < 1e-6);                       // escaped to the global minimum
    CHECK(close(m.constants[0], 2.0, 1e-3));
    CHECK(m.loss <= s.loss);                    // multi-start is never worse
}

// n_restarts=0 reduces to the previous single-LM behaviour: it still recovers the linear
// fit through the start-0 path with no perturbation draws.
void test_single_start_backward_compat() {
    const LinearData data = make_data();
    const OptimizerConfig cfg{0, 0, 100, 0.5};
    SelfLMOptimizer opt(cfg);
    const OptimizationResult res = opt.optimize(make_raw_problem(data));
    CHECK(res.success);
    CHECK(close(res.constants[0], kTrueA, 1e-4));
    CHECK(close(res.constants[1], kTrueB, 1e-4));
}

// Multi-start is deterministic across fresh instances with the same seed, even on the
// restart-driven double-well problem (identical perturbation stream => identical result).
void test_multistart_deterministic() {
    const OptimizerConfig cfg{0, 20, 100, 0.5};
    SelfLMOptimizer a(cfg);
    SelfLMOptimizer b(cfg);
    const OptimizationResult ra = a.optimize(make_double_well(3.0));
    const OptimizationResult rb = b.optimize(make_double_well(3.0));
    CHECK(ra.constants.size() == rb.constants.size());
    CHECK(ra.constants[0] == rb.constants[0]);
    CHECK(ra.loss == rb.loss);
}

// A null residual function is misuse and must throw.
void test_null_residuals_throws() {
    SelfLMOptimizer opt;
    OptimizationProblem problem;
    problem.num_residuals = 1;
    problem.initial_constants = {0.0};
    // problem.residuals left null

    bool threw = false;
    try {
        opt.optimize(problem);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

// k == 0 (no tunable constants): a single SSE evaluation, no solve.
void test_no_constants_reports_loss() {
    SelfLMOptimizer opt;
    OptimizationProblem problem;
    problem.num_residuals = 3;
    problem.initial_constants = {};  // k == 0
    problem.residuals = [](const std::vector<double>&, std::vector<double>& r) {
        r[0] = 1.0;
        r[1] = 2.0;
        r[2] = 2.0;  // SSE = 1 + 4 + 4 = 9
    };
    const OptimizationResult res = opt.optimize(problem);
    CHECK(res.constants.empty());
    CHECK(close(res.loss, 9.0, 1e-12));
    CHECK(res.success);
    CHECK(res.evaluations == 1);
}

// With an immediately-true stop predicate, optimize() returns a valid-shaped result
// (constants.size() == k, never NaN loss) without crashing.
void test_immediate_stop_returns_valid() {
    const LinearData data = make_data();
    const OptimizationProblem problem = make_raw_problem(data);

    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem, [] { return true; });

    CHECK(res.constants.size() == 2);
    CHECK(res.loss == res.loss);  // not NaN
}

// --- Tree + dual-number Jacobian problems (analytic Jacobian path) ------------

Tree linear_tree() {
    return {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul),
            constant_node(1, 0.0), binary_node(BinaryOp::Add)};
}

Tree exp_tree() {
    return {constant_node(0, 1.0), constant_node(1, 0.1), variable_node(0),
            binary_node(BinaryOp::Mul), unary_node(UnaryOp::Exp),
            binary_node(BinaryOp::Mul)};
}

// Recover a = 2.5, b = 1.7 from y = a*x + b using the analytic dual Jacobian.
void test_fit_linear_analytic() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(kTrueA * x + kTrueB);
    }

    OptimizationProblem problem =
        make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0});

    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(close(res.constants[0], kTrueA, 1e-4));
    CHECK(close(res.constants[1], kTrueB, 1e-4));
    CHECK(res.loss < 1e-6);
}

// Recover a = 2.0, b = 0.3 from y = a*exp(b*x) (nonlinear; tests conditioning).
void test_fit_exponential() {
    const double true_a = 2.0;
    const double true_b = 0.3;
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = 0.5 * static_cast<double>(i);
        X.push_back({x});
        y.push_back(true_a * std::exp(true_b * x));
    }

    OptimizationProblem problem = make_least_squares_problem(exp_tree(), X, y);

    SelfLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(close(res.constants[0], true_a, 1e-3));
    CHECK(close(res.constants[1], true_b, 1e-3));
    CHECK(res.loss < 1e-6);
}

// The analytic (dual) Jacobian and the finite-difference fallback must converge to the
// same fit, confirming the two Jacobian paths are consistent.
void test_analytic_vs_numerical_jacobian() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 10; ++i) {
        const double x = static_cast<double>(i);
        X.push_back({x});
        y.push_back(kTrueA * x + kTrueB);
    }

    OptimizationProblem analytic =
        make_least_squares_problem(linear_tree(), X, y, {0.0, 0.0});
    OptimizationProblem numeric = analytic;
    numeric.jacobian = nullptr;  // force the finite-difference path

    SelfLMOptimizer opt;
    const OptimizationResult a = opt.optimize(analytic);
    const OptimizationResult n = opt.optimize(numeric);

    CHECK(a.success);
    CHECK(n.success);
    CHECK(close(a.constants[0], n.constants[0], 1e-4));
    CHECK(close(a.constants[1], n.constants[1], 1e-4));
}

}  // namespace

int main() {
    test_recovers_linear_numeric_jacobian();
    test_via_factory();
    test_reproducibility();
    test_null_residuals_throws();
    test_no_constants_reports_loss();
    test_immediate_stop_returns_valid();
    test_fit_linear_analytic();
    test_fit_exponential();
    test_analytic_vs_numerical_jacobian();
    test_multistart_escapes_local_min();
    test_single_start_backward_compat();
    test_multistart_deterministic();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
