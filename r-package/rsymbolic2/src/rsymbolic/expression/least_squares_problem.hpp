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

// Builds a least-squares OptimizationProblem from an expression tree and a dataset.
//
//   tree     - the model, in postfix form, whose Constant nodes are the tunable params
//   X        - rows of input features (X[i] is the feature vector for data point i)
//   y        - target values (y[i] corresponds to X[i])
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
// The tree and dataset are copied into shared storage so the returned closures remain
// valid independently of the caller's objects.
inline OptimizationProblem make_least_squares_problem(
    Tree tree,
    std::vector<std::vector<double>> X,
    std::vector<double> y,
    std::vector<double> initial = {},
    StopRequested stop = {}) {
    struct Data {
        Tree tree;
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        int k;
    };

    const int k = count_constants(tree);
    if (initial.empty()) {
        initial = initial_constants(tree);
    }

    auto data = std::make_shared<Data>(
        Data{std::move(tree), std::move(X), std::move(y), k});

    // Abort flag: created only when a stop predicate is provided so that the
    // no-stop path has no overhead beyond an extra nullptr check. Both closures
    // capture the same shared_ptr; the optimizer resets it before each optimize().
    auto aborted = stop
        ? std::make_shared<std::atomic<bool>>(false)
        : std::shared_ptr<std::atomic<bool>>{};

    OptimizationProblem problem;
    problem.num_residuals = data->y.size();
    problem.initial_constants = std::move(initial);
    problem.aborted = aborted;

    problem.residuals = [data, stop, aborted](const std::vector<double>& params,
                                              std::vector<double>& residuals) {
        const std::size_t m = data->y.size();
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
                evaluate<double>(data->tree, data->X[i].data(), params.data());
            residuals[i] = prediction - data->y[i];
        }
    };

    problem.jacobian = [data, stop, aborted](const std::vector<double>& params,
                                             std::vector<double>& jacobian) {
        const int k = data->k;
        const std::size_t m = data->y.size();
        std::vector<Dual> constants(static_cast<std::size_t>(k));
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
                    evaluate<Dual>(data->tree, data->X[i].data(), constants.data());
                // d r_i / d c_kk == d prediction_i / d c_kk (y_i is constant).
                jacobian[i * static_cast<std::size_t>(k) +
                         static_cast<std::size_t>(kk)] = prediction.deriv;
            }
        }
    };

    return problem;
}

}  // namespace rsymbolic
