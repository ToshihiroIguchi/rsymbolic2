// Phase 1 interruptible-evaluation tests (docs/22 §5.4).
//
// Test (a) honours_stop: pass a stop predicate that trips on the first invocation;
//   assert optimize() returns with nfev far below maxfev, success==true, finite loss.
// Test (b) no_stop_unchanged: with empty stop, assert the result (constants/loss/nfev)
//   equals a reference run built without the new parameter — pins the no-behavioural-
//   change invariant (docs/22 §5.3).

#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"
#include "rsymbolic/optimization/self_lm_optimizer.hpp"

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s (%s:%d)\n", expr, file, line);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

using namespace rsymbolic;

// Build a linear tree representing c0 * x + c1.
Tree linear_tree() {
    return {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul),
            constant_node(1, 0.0), binary_node(BinaryOp::Add)};
}

// Generate a small noise-free dataset y = 2.5*x + 1.7 with m=50 points so the
// residual loop definitely reaches the kStride=256 boundary on a sufficiently
// large dataset. We use m=300 (> kStride) so the stride check fires in one pass.
constexpr int    kNPoints = 300;
constexpr double kTrueA   = 2.5;
constexpr double kTrueB   = 1.7;

struct Dataset {
    std::vector<std::vector<double>> X;
    std::vector<double>              y;
};

Dataset make_dataset() {
    Dataset d;
    d.X.reserve(kNPoints);
    d.y.reserve(kNPoints);
    for (int i = 0; i < kNPoints; ++i) {
        const double x = static_cast<double>(i) / 10.0;
        d.X.push_back({x});
        d.y.push_back(kTrueA * x + kTrueB);
    }
    return d;
}

// (a) The stop predicate fires after the first poll (at point 256 of 300 in the
// residual loop). We verify:
//   - optimize() returns without hanging
//   - nfev is at most a small number (far below the default maxfev)
//   - success == true (UserAsked is not ImproperInputParameters)
//   - loss is finite (not NaN)
void test_honours_stop() {
    const Dataset d = make_dataset();

    // Count how many times the predicate is invoked.
    std::atomic<int> poll_count{0};
    StopRequested stop = [&poll_count]() -> bool {
        // Fire on the first poll so the first residual evaluation is aborted
        // at point kStride=256.
        return poll_count.fetch_add(1, std::memory_order_relaxed) >= 0;
    };

    OptimizationProblem problem =
        make_least_squares_problem(linear_tree(), d.X, d.y, {0.0, 0.0}, stop);

    SelfLMOptimizer opt;
    // Use a large maxfev so that if the stop does not fire the test would take
    // many evaluations (and we could detect the failure by nfev being large).
    OptimizerConfig cfg;
    cfg.max_iterations = 1000;
    SelfLMOptimizer opt2(cfg);
    const OptimizationResult res = opt2.optimize(problem, stop);

    // The abort must have fired (poll_count > 0).
    CHECK(poll_count.load() > 0);
    // nfev must be tiny — the abort fires inside the first (or second) evaluation.
    CHECK(res.evaluations < 20);
    // success must be true (UserAsked is not ImproperInputParameters).
    CHECK(res.success);
    // loss must be finite (not NaN).
    CHECK(std::isfinite(res.loss));
    // constants vector must have the right size.
    CHECK(res.constants.size() == 2);
}

// (b) With an empty stop predicate, optimize() must produce the same result as
// a reference built without the stop parameter. This pins the invariant that the
// new parameter adds no behavioural change when unused.
void test_no_stop_unchanged() {
    const Dataset d = make_dataset();

    // Reference: old call site shape (no stop predicate).
    OptimizationProblem ref_problem =
        make_least_squares_problem(linear_tree(), d.X, d.y, {0.0, 0.0});

    // New: empty StopRequested{} — must be identical to the reference.
    OptimizationProblem new_problem =
        make_least_squares_problem(linear_tree(), d.X, d.y, {0.0, 0.0},
                                   StopRequested{});

    SelfLMOptimizer opt;
    const OptimizationResult ref = opt.optimize(ref_problem);
    const OptimizationResult res = opt.optimize(new_problem);

    CHECK(ref.success == res.success);
    CHECK(ref.evaluations == res.evaluations);
    CHECK(ref.loss == res.loss);
    CHECK(ref.constants.size() == res.constants.size());
    if (ref.constants.size() == res.constants.size()) {
        for (std::size_t i = 0; i < ref.constants.size(); ++i) {
            CHECK(ref.constants[i] == res.constants[i]);
        }
    }
}

}  // namespace

int main() {
    test_honours_stop();
    test_no_stop_unchanged();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
