// Convergence test for the Eigen Levenberg-Marquardt backend on the canonical
// two-constant linear model y = a*x + b, with ground truth a = 2.5, b = 1.7.
// Self-contained (no test framework); returns 0 on success, 1 on failure.

#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

#include "rsymbolic/optimization/constant_optimizer.hpp"
#include "rsymbolic/optimization/eigen_lm_optimizer.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

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

constexpr double kTrueA = 2.5;
constexpr double kTrueB = 1.7;

// Fixed dataset x = 0, 1, ..., 9 and y = 2.5*x + 1.7 (no noise).
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

rsymbolic::OptimizationProblem make_problem(const LinearData& data) {
    using namespace rsymbolic;
    OptimizationProblem problem;
    problem.num_residuals = data.x.size();
    problem.initial_constants = {0.0, 0.0};  // a, b
    problem.residuals = [&data](const std::vector<double>& c,
                                std::vector<double>& r) {
        const double a = c[0];
        const double b = c[1];
        for (std::size_t i = 0; i < data.x.size(); ++i) {
            r[i] = (a * data.x[i] + b) - data.y[i];
        }
    };
    return problem;
}

// Exact recovery of (2.5, 1.7) from noise-free data.
void test_recovers_linear_constants() {
    using namespace rsymbolic;
    const LinearData data = make_data();
    const OptimizationProblem problem = make_problem(data);

    EigenLMOptimizer opt;
    const OptimizationResult res = opt.optimize(problem);

    CHECK(res.success);
    CHECK(res.constants.size() == 2);
    CHECK(std::fabs(res.constants[0] - kTrueA) < 1e-4);
    CHECK(std::fabs(res.constants[1] - kTrueB) < 1e-4);
    CHECK(res.loss < 1e-6);
    CHECK(res.evaluations > 0);
}

// The backend must be reachable through the factory and behave identically.
void test_via_factory() {
    using namespace rsymbolic;
    const LinearData data = make_data();
    const OptimizationProblem problem = make_problem(data);

    auto opt = OptimizerFactory::create(OptimizerType::EigenLM);
    CHECK(opt != nullptr);
    CHECK(opt->name() == "EigenLMOptimizer");

    const OptimizationResult res = opt->optimize(problem);
    CHECK(res.success);
    CHECK(std::fabs(res.constants[0] - kTrueA) < 1e-4);
    CHECK(std::fabs(res.constants[1] - kTrueB) < 1e-4);
}

// Deterministic: optimizing the same problem twice gives identical results.
void test_reproducibility() {
    using namespace rsymbolic;
    const LinearData data = make_data();

    EigenLMOptimizer opt;
    const OptimizationResult a = opt.optimize(make_problem(data));
    const OptimizationResult b = opt.optimize(make_problem(data));

    CHECK(a.constants.size() == b.constants.size());
    CHECK(a.constants[0] == b.constants[0]);
    CHECK(a.constants[1] == b.constants[1]);
    CHECK(a.loss == b.loss);
}

// A null residual function is misuse and must throw.
void test_null_residuals_throws() {
    using namespace rsymbolic;
    EigenLMOptimizer opt;
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

}  // namespace

int main() {
    test_recovers_linear_constants();
    test_via_factory();
    test_reproducibility();
    test_null_residuals_throws();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
