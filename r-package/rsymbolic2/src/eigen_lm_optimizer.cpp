#include "rsymbolic/optimization/eigen_lm_optimizer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

namespace rsymbolic {

namespace {

// A non-finite residual would poison the Levenberg-Marquardt step. We clamp such
// values to a large finite magnitude so the solver treats that point as a very poor
// fit and steps away, rather than producing NaNs in the normal equations.
constexpr double kLargeResidual = 1.0e10;

void fill_residuals(const ResidualFunction& residuals, const Eigen::VectorXd& x,
                    int m, Eigen::VectorXd& fvec) {
    std::vector<double> params(x.data(), x.data() + x.size());
    std::vector<double> r(static_cast<std::size_t>(m), 0.0);
    residuals(params, r);
    for (int i = 0; i < m; ++i) {
        const double value = r[static_cast<std::size_t>(i)];
        fvec[i] = std::isfinite(value) ? value : kLargeResidual;
    }
}

// Functor used when no analytic Jacobian is provided: residuals only, with the
// Jacobian supplied by Eigen::NumericalDiff.
struct NumericFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

    const ResidualFunction& residuals;
    int n_inputs;
    int n_values;

    int inputs() const { return n_inputs; }
    int values() const { return n_values; }

    int operator()(const InputType& x, ValueType& fvec) const {
        fill_residuals(residuals, x, n_values, fvec);
        return 0;
    }
};

// Functor used when an analytic Jacobian (e.g. from dual numbers) is available.
struct AnalyticFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

    const ResidualFunction& residuals;
    const JacobianFunction& jacobian;
    int n_inputs;
    int n_values;

    int inputs() const { return n_inputs; }
    int values() const { return n_values; }

    int operator()(const InputType& x, ValueType& fvec) const {
        fill_residuals(residuals, x, n_values, fvec);
        return 0;
    }

    int df(const InputType& x, JacobianType& fjac) const {
        std::vector<double> params(x.data(), x.data() + x.size());
        std::vector<double> jac(
            static_cast<std::size_t>(n_values) * static_cast<std::size_t>(n_inputs),
            0.0);
        jacobian(params, jac);
        fjac.resize(n_values, n_inputs);
        for (int i = 0; i < n_values; ++i) {
            for (int j = 0; j < n_inputs; ++j) {
                double value =
                    jac[static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(n_inputs) +
                        static_cast<std::size_t>(j)];
                fjac(i, j) = std::isfinite(value) ? value : kLargeResidual;
            }
        }
        return 0;
    }
};

template <typename Lm>
void finalize(Lm& lm, Eigen::LevenbergMarquardtSpace::Status status,
              const Eigen::VectorXd& x, OptimizationResult& result) {
    result.constants.assign(x.data(), x.data() + x.size());
    const double loss = lm.fvec.squaredNorm();
    result.loss = loss;
    result.evaluations = static_cast<std::size_t>(lm.nfev);
    result.success =
        std::isfinite(loss) &&
        status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters;
}

// Run Levenberg-Marquardt with a stop poll between steps. This is the explicit
// decomposition of LevenbergMarquardt::minimize(): in the vendored Eigen, minimize()
// is exactly `minimizeInit(x); do { minimizeOneStep(x); } while (Running);`
// (NonLinearOptimization/LevenbergMarquardt.h:157-166), so with an always-false
// predicate this is the same code path, bit-identical by construction. The poll lets
// a long fit on a bloated tree honour the search-loop deadline instead of running the
// full maxfev budget uninterruptibly. See docs/20.
template <typename Lm>
Eigen::LevenbergMarquardtSpace::Status run_lm(Lm& lm, Eigen::VectorXd& x,
                                              const StopRequested& stop_requested) {
    using Status = Eigen::LevenbergMarquardtSpace::Status;
    Status status = lm.minimizeInit(x);
    if (status == Status::ImproperInputParameters) return status;
    do {
        status = lm.minimizeOneStep(x);
    } while (status == Status::Running && !stop_requested());
    return status;
}

}  // namespace

EigenLMOptimizer::EigenLMOptimizer(OptimizerConfig config) : config_(config) {}

std::string EigenLMOptimizer::name() const { return "EigenLMOptimizer"; }

OptimizationResult EigenLMOptimizer::optimize(
    const OptimizationProblem& problem, const StopRequested& stop_requested) const {
    if (!problem.residuals) {
        throw std::invalid_argument(
            "EigenLMOptimizer: problem.residuals must not be null");
    }

    const int k = static_cast<int>(problem.initial_constants.size());
    const int m = static_cast<int>(problem.num_residuals);

    OptimizationResult result;
    result.constants = problem.initial_constants;

    // Degenerate case: no tunable constants. Nothing to solve; just report the loss.
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

    const int maxfev =
        config_.max_iterations > 0
            ? static_cast<int>(config_.max_iterations) * (k + 1)
            : 0;

    if (problem.jacobian) {
        AnalyticFunctor functor{problem.residuals, problem.jacobian, k, m};
        Eigen::LevenbergMarquardt<AnalyticFunctor> lm(functor);
        if (maxfev > 0) lm.parameters.maxfev = maxfev;
        const Eigen::LevenbergMarquardtSpace::Status status =
            run_lm(lm, x, stop_requested);
        finalize(lm, status, x, result);
    } else {
        NumericFunctor functor{problem.residuals, k, m};
        Eigen::NumericalDiff<NumericFunctor> num_diff(functor);
        Eigen::LevenbergMarquardt<Eigen::NumericalDiff<NumericFunctor>> lm(num_diff);
        if (maxfev > 0) lm.parameters.maxfev = maxfev;
        const Eigen::LevenbergMarquardtSpace::Status status =
            run_lm(lm, x, stop_requested);
        finalize(lm, status, x, result);
    }

    return result;
}

}  // namespace rsymbolic
