#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// Builds a least-squares OptimizationProblem from an expression tree and a dataset.
//
//   tree     - the model, in postfix form, whose Constant nodes are the tunable params
//   X        - rows of input features (X[i] is the feature vector for data point i)
//   y        - target values (y[i] corresponds to X[i])
//   initial  - initial constant values (size k); if empty, taken from the tree
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
    std::vector<double> initial = {}) {
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

    OptimizationProblem problem;
    problem.num_residuals = data->y.size();
    problem.initial_constants = std::move(initial);

    problem.residuals = [data](const std::vector<double>& params,
                               std::vector<double>& residuals) {
        for (std::size_t i = 0; i < data->y.size(); ++i) {
            const double prediction =
                evaluate<double>(data->tree, data->X[i].data(), params.data());
            residuals[i] = prediction - data->y[i];
        }
    };

    problem.jacobian = [data](const std::vector<double>& params,
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
