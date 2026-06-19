#pragma once

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// Number of data points to process between stop-predicate polls in the residual
// and Jacobian closures. A compile-time constant (not a public knob). At 256
// points per check, the overhead is one std::function call per ~256 evaluations
// in the worst case; the coarse granularity prevents excessive polling overhead.
// See docs/22 Phase 1.
static constexpr std::size_t kStride = 256;

// Sentinel residual value used when a stop is triggered mid-evaluation. Matches
// the non-finite clamping value already used in fill_residuals (eigen_lm_optimizer.cpp).
static constexpr double kLargeResidualSentinel = 1.0e10;

// An immutable dataset (input rows + targets) shared across every per-candidate
// problem in a search. Sharing it by shared_ptr means building a problem costs one
// tree copy, not a full dataset copy — the dataset copy was ~m heap allocations per
// fit() and the dominant source of multi-island heap-lock contention (docs/23 §4).
struct Dataset {
    std::vector<std::vector<double>> X;  // X[i] is the feature vector for point i
    std::vector<double> y;               // y[i] corresponds to X[i]
};

// Builds a least-squares OptimizationProblem from an expression tree and a dataset.
//
//   tree     - the model, in postfix form, whose Constant nodes are the tunable params
//   dataset  - shared, immutable inputs/targets (referenced, not copied)
//   initial  - initial constant values (size k); if empty, taken from the tree
//   stop     - optional stop predicate (docs/22 Phase 1). When non-empty, the
//              residual/Jacobian closures poll it every kStride points and signal an
//              abort via problem.aborted when it returns true. With an empty or
//              never-true stop, behaviour is identical to the no-stop case.
//
// The returned problem provides:
//   residuals: r_i(c) = model(X[i]; c) - y[i]              (plain double evaluation)
//   jacobian : d r_i / d c_j via forward-mode dual numbers (k passes over the data)
//
// The tree is copied into shared storage and the dataset is shared by pointer, so the
// returned closures remain valid independently of the caller's tree object.
inline OptimizationProblem make_least_squares_problem(
    Tree tree,
    std::shared_ptr<const Dataset> dataset,
    std::vector<double> initial = {},
    StopRequested stop = {}) {
    struct Model {
        Tree tree;
        std::shared_ptr<const Dataset> data;
        int k;
    };

    const int k = count_constants(tree);
    if (initial.empty()) {
        initial = initial_constants(tree);
    }

    auto model = std::make_shared<Model>(
        Model{std::move(tree), std::move(dataset), k});

    // Abort flag: created only when a stop predicate is provided so that the
    // no-stop path has no overhead beyond an extra nullptr check. Both closures
    // capture the same shared_ptr; the optimizer resets it before each optimize().
    auto aborted = stop
        ? std::make_shared<std::atomic<bool>>(false)
        : std::shared_ptr<std::atomic<bool>>{};

    OptimizationProblem problem;
    problem.num_residuals = model->data->y.size();
    problem.initial_constants = std::move(initial);
    problem.aborted = aborted;

    problem.residuals = [model, stop, aborted](const std::vector<double>& params,
                                              std::vector<double>& residuals) {
        const Dataset& d = *model->data;
        const std::size_t m = d.y.size();
        // One evaluation stack reused across all m points (docs/23 §4): the per-point
        // evaluate() call no longer allocates. Local to this closure invocation, so
        // it is never shared across threads.
        std::vector<double> stack;
        for (std::size_t i = 0; i < m; ++i) {
            // Poll stop predicate every kStride points (docs/22 §5.1 step 3).
            if (aborted && i % kStride == 0 && stop && stop()) {
                aborted->store(true, std::memory_order_relaxed);
                // Fill remaining residuals with the sentinel so Eigen sees a
                // well-defined (large) residual rather than an uninitialised value.
                for (std::size_t j = i; j < m; ++j)
                    residuals[j] = kLargeResidualSentinel;
                return;
            }
            const double prediction =
                evaluate<double>(model->tree, d.X[i].data(), params.data(), stack);
            residuals[i] = prediction - d.y[i];
        }
    };

    problem.jacobian = [model, stop, aborted](const std::vector<double>& params,
                                             std::vector<double>& jacobian) {
        const int k = model->k;
        const Dataset& d = *model->data;
        const std::size_t m = d.y.size();
        std::vector<Dual> constants(static_cast<std::size_t>(k));
        // Evaluation stack reused across all k*m dual-number evaluations (docs/23 §4).
        // Local to this closure invocation — never shared across threads.
        std::vector<Dual> stack;
        for (int kk = 0; kk < k; ++kk) {
            // Seed constant kk with derivative 1, all others with 0.
            for (int j = 0; j < k; ++j) {
                constants[static_cast<std::size_t>(j)] =
                    Dual(params[static_cast<std::size_t>(j)], j == kk ? 1.0 : 0.0);
            }
            for (std::size_t i = 0; i < m; ++i) {
                // Poll stop predicate every kStride points (docs/22 §5.1 step 3).
                if (aborted && i % kStride == 0 && stop && stop()) {
                    aborted->store(true, std::memory_order_relaxed);
                    // The caller (AnalyticFunctor::df) initialises the jac vector
                    // to 0.0, so unfilled entries are already defined. Just return;
                    // the abort flag causes df() to return -1 and Eigen discards
                    // the (partial, untrustworthy) matrix.
                    return;
                }
                const Dual prediction =
                    evaluate<Dual>(model->tree, d.X[i].data(), constants.data(),
                                   stack);
                // d r_i / d c_kk == d prediction_i / d c_kk (y_i is constant).
                jacobian[i * static_cast<std::size_t>(k) +
                         static_cast<std::size_t>(kk)] = prediction.deriv;
            }
        }
    };

    return problem;
}

// Value-copy overload (tests and other non-search callers). Copies X and y once into
// a shared Dataset and delegates to the shared-dataset overload above, preserving the
// original contract that the returned closures own their data independently of the
// caller's X/y.
inline OptimizationProblem make_least_squares_problem(
    Tree tree,
    std::vector<std::vector<double>> X,
    std::vector<double> y,
    std::vector<double> initial = {},
    StopRequested stop = {}) {
    auto data = std::make_shared<const Dataset>(Dataset{std::move(X), std::move(y)});
    return make_least_squares_problem(std::move(tree), std::move(data),
                                      std::move(initial), std::move(stop));
}

}  // namespace rsymbolic
