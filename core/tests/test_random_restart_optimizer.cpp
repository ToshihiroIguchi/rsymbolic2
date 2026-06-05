// Minimal self-contained test for the walking skeleton. No external test framework is
// used yet: the executable returns 0 if all checks pass and 1 otherwise, and is wired
// to CTest via add_test. (Catch2 can be adopted later without changing the structure.)

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "rsymbolic/optimization/constant_optimizer.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"
#include "rsymbolic/optimization/random_restart_optimizer.hpp"

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

rsymbolic::OptimizerConfig test_config() {
    rsymbolic::OptimizerConfig cfg;
    cfg.seed = 42;
    cfg.n_restarts = 10;
    cfg.max_iterations = 500;
    cfg.perturbation_scale = 0.5;
    return cfg;
}

// 1D convex quadratic with optimum at x = 3.
void test_quadratic_1d() {
    using namespace rsymbolic;
    RandomRestartOptimizer opt(test_config());

    OptimizationProblem problem;
    problem.objective = [](const std::vector<double>& c) {
        const double d = c[0] - 3.0;
        return d * d;
    };
    problem.initial_constants = {0.0};

    const OptimizationResult res = opt.optimize(problem);
    CHECK(res.success);
    CHECK(res.constants.size() == 1);
    CHECK(std::fabs(res.constants[0] - 3.0) < 0.25);
    CHECK(res.loss < 0.1);
    CHECK(res.evaluations > 0);
}

// 2D convex quadratic with optimum at (2, -1).
void test_quadratic_2d() {
    using namespace rsymbolic;
    RandomRestartOptimizer opt(test_config());

    OptimizationProblem problem;
    problem.objective = [](const std::vector<double>& c) {
        const double a = c[0] - 2.0;
        const double b = c[1] + 1.0;
        return a * a + b * b;
    };
    problem.initial_constants = {0.0, 0.0};

    const OptimizationResult res = opt.optimize(problem);
    CHECK(res.success);
    CHECK(res.constants.size() == 2);
    CHECK(std::fabs(res.constants[0] - 2.0) < 0.3);
    CHECK(std::fabs(res.constants[1] + 1.0) < 0.3);
    CHECK(res.loss < 0.15);
}

// Same seed must yield identical results (reproducibility is a project requirement).
void test_reproducibility() {
    using namespace rsymbolic;
    auto make_problem = []() {
        OptimizationProblem p;
        p.objective = [](const std::vector<double>& c) {
            const double d = c[0] - 1.5;
            return d * d;
        };
        p.initial_constants = {0.0};
        return p;
    };

    RandomRestartOptimizer opt(test_config());
    const OptimizationResult a = opt.optimize(make_problem());
    const OptimizationResult b = opt.optimize(make_problem());

    CHECK(a.constants.size() == b.constants.size());
    CHECK(a.constants[0] == b.constants[0]);  // exact: deterministic given the seed
    CHECK(a.loss == b.loss);
    CHECK(a.evaluations == b.evaluations);
}

// Non-finite objective regions must be rejected (treated as +Inf).
void test_non_finite_rejected() {
    using namespace rsymbolic;
    RandomRestartOptimizer opt(test_config());

    OptimizationProblem problem;
    problem.objective = [](const std::vector<double>& c) {
        if (c[0] < 0.0) return std::numeric_limits<double>::infinity();
        const double d = c[0] - 2.0;
        return d * d;
    };
    problem.initial_constants = {5.0};  // start inside the finite region

    const OptimizationResult res = opt.optimize(problem);
    CHECK(res.success);
    CHECK(res.constants[0] >= 0.0);
    CHECK(std::fabs(res.constants[0] - 2.0) < 0.25);
    CHECK(res.loss < 0.1);
}

// k == 0: an expression with no tunable constants.
void test_zero_dimension() {
    using namespace rsymbolic;
    RandomRestartOptimizer opt(test_config());

    OptimizationProblem problem;
    problem.objective = [](const std::vector<double>& c) {
        (void)c;
        return 5.0;
    };
    problem.initial_constants = {};

    const OptimizationResult res = opt.optimize(problem);
    CHECK(res.success);
    CHECK(res.constants.empty());
    CHECK(res.loss == 5.0);
    CHECK(res.evaluations > 0);
}

void test_name() {
    rsymbolic::RandomRestartOptimizer opt;
    CHECK(opt.name() == "RandomRestartOptimizer");
}

// The factory builds the implemented backend, throws for unimplemented ones, and
// parses names. This is the swappability seam.
void test_factory() {
    using namespace rsymbolic;

    auto opt = OptimizerFactory::create(OptimizerType::RandomRestart, test_config());
    CHECK(opt != nullptr);
    CHECK(opt->name() == "RandomRestartOptimizer");

    OptimizationProblem problem;
    problem.objective = [](const std::vector<double>& c) {
        const double d = c[0] - 3.0;
        return d * d;
    };
    problem.initial_constants = {0.0};
    const OptimizationResult res = opt->optimize(problem);
    CHECK(res.success);
    CHECK(std::fabs(res.constants[0] - 3.0) < 0.25);

    bool threw = false;
    try {
        OptimizerFactory::create(OptimizerType::EigenLM, test_config());
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    CHECK(OptimizerFactory::from_string("random_restart") ==
          OptimizerType::RandomRestart);
    CHECK(OptimizerFactory::from_string("eigen_lm") == OptimizerType::EigenLM);
    CHECK(OptimizerFactory::from_string("ceres_tiny") ==
          OptimizerType::CeresTinySolver);
    CHECK(OptimizerFactory::from_string("ceres") == OptimizerType::Ceres);

    bool name_threw = false;
    try {
        OptimizerFactory::from_string("does_not_exist");
    } catch (const std::invalid_argument&) {
        name_threw = true;
    }
    CHECK(name_threw);
}

}  // namespace

int main() {
    test_quadratic_1d();
    test_quadratic_2d();
    test_reproducibility();
    test_non_finite_rejected();
    test_zero_dimension();
    test_name();
    test_factory();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
