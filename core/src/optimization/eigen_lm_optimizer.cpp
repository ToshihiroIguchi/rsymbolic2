#include "rsymbolic/optimization/eigen_lm_optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

namespace rsymbolic {

namespace {

// A non-finite residual would poison the Levenberg-Marquardt step. We clamp such
// values to a large finite magnitude so the solver simply treats that point as a very
// poor fit and steps away from it, rather than producing NaNs in the normal equations.
constexpr double kLargeResidual = 1.0e10;

// Eigen functor adapting rsymbolic's ResidualFunction to the interface expected by
// Eigen::NumericalDiff / Eigen::LevenbergMarquardt. Eigen is confined to this file.
struct ResidualFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    const ResidualFunction& residuals;
    int n_inputs;   // k (number of constants)
    int n_values;   // m (number of residuals)

    int inputs() const { return n_inputs; }
    int values() const { return n_values; }

    int operator()(const InputType& x, ValueType& fvec) const {
        std::vector<double> params(x.data(), x.data() + x.size());
        std::vector<double> r(static_cast<std::size_t>(n_values), 0.0);
        residuals(params, r);
        for (int i = 0; i < n_values; ++i) {
            const double value = r[static_cast<std::size_t>(i)];
            fvec[i] = std::isfinite(value) ? value : kLargeResidual;
        }
        return 0;
    }
};

}  // namespace

EigenLMOptimizer::EigenLMOptimizer(OptimizerConfig config) : config_(config) {}

std::string EigenLMOptimizer::name() const { return "EigenLMOptimizer"; }

OptimizationResult EigenLMOptimizer::optimize(
    const OptimizationProblem& problem) const {
    if (!problem.residuals) {
        throw std::invalid_argument(
            "EigenLMOptimizer: problem.residuals must not be null");
    }

    const int k = static_cast<int>(problem.initial_constants.size());
    const int m = static_cast<int>(problem.num_residuals);

    OptimizationResult result;
    result.constants = problem.initial_constants;

    // Degenerate case: no tunable constants. There is nothing to solve; just evaluate
    // the residuals once and report the loss.
    if (k == 0) {
        result.loss = sum_of_squared_residuals(problem, problem.initial_constants);
        result.success = std::isfinite(result.loss);
        result.evaluations = 1;
        return result;
    }

    Eigen::VectorXd x(k);
    for (int i = 0; i < k; ++i) {
        x[i] = problem.initial_constants[static_cast<std::size_t>(i)];
    }

    ResidualFunctor functor{problem.residuals, k, m};
    Eigen::NumericalDiff<ResidualFunctor> num_diff(functor);
    Eigen::LevenbergMarquardt<Eigen::NumericalDiff<ResidualFunctor>> lm(num_diff);

    if (config_.max_iterations > 0) {
        lm.parameters.maxfev = static_cast<int>(config_.max_iterations) * (k + 1);
    }

    const Eigen::LevenbergMarquardtSpace::Status status = lm.minimize(x);

    result.constants.assign(x.data(), x.data() + x.size());
    const double loss = lm.fvec.squaredNorm();
    result.loss = loss;
    result.evaluations = static_cast<std::size_t>(lm.nfev);
    result.success =
        std::isfinite(loss) &&
        status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters;
    return result;
}

}  // namespace rsymbolic
