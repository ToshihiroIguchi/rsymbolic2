#include "rsymbolic/optimization/eigen_lm_optimizer.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
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

// Copy Eigen's parameter vector into the reusable `params` buffer (no allocation:
// `params` is pre-sized to k), evaluate residuals into the reusable `r` buffer, then
// clamp non-finite values into Eigen's fvec.
void fill_residuals(const ResidualFunction& residuals, const Eigen::VectorXd& x,
                    int m, Eigen::VectorXd& fvec,
                    std::vector<double>& params, std::vector<double>& r) {
    for (int i = 0; i < static_cast<int>(x.size()); ++i)
        params[static_cast<std::size_t>(i)] = x[i];
    residuals(params, r);
    for (int i = 0; i < m; ++i) {
        const double value = r[static_cast<std::size_t>(i)];
        fvec[i] = std::isfinite(value) ? value : kLargeResidual;
    }
}

// Functor used when no analytic Jacobian is provided: residuals only, with the
// Jacobian supplied by Eigen::NumericalDiff. The params/residual scratch buffers are
// owned by the optimizer and reused across evaluations and fits (docs/23 §4).
struct NumericFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

    const ResidualFunction& residuals;
    // Abort flag from OptimizationProblem (docs/22 Phase 1). Null when no stop
    // predicate was supplied; when set by the closure, operator() returns -1 so
    // Eigen's minimizeOneStep returns UserAsked and run_lm exits.
    std::shared_ptr<std::atomic<bool>> aborted;
    int n_inputs;
    int n_values;
    std::vector<double>& params;  // scratch (size n_inputs), owned by optimizer
    std::vector<double>& rbuf;    // scratch (size n_values), owned by optimizer

    int inputs() const { return n_inputs; }
    int values() const { return n_values; }

    int operator()(const InputType& x, ValueType& fvec) const {
        fill_residuals(residuals, x, n_values, fvec, params, rbuf);
        if (aborted && aborted->load(std::memory_order_relaxed)) return -1;
        return 0;
    }
};

// Functor used when an analytic Jacobian (e.g. from dual numbers) is available. The
// params/residual/Jacobian scratch buffers are owned by the optimizer and reused
// across evaluations and fits (docs/23 §4).
struct AnalyticFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };

    const ResidualFunction& residuals;
    const JacobianFunction& jacobian;
    // Abort flag from OptimizationProblem (docs/22 Phase 1). Null when no stop
    // predicate was supplied; when set by a closure, operator()/df() returns -1
    // so Eigen's minimizeOneStep returns UserAsked and run_lm exits.
    std::shared_ptr<std::atomic<bool>> aborted;
    int n_inputs;
    int n_values;
    std::vector<double>& params;  // scratch (size n_inputs), owned by optimizer
    std::vector<double>& rbuf;    // scratch (size n_values), owned by optimizer
    std::vector<double>& jbuf;    // scratch (size n_values*n_inputs), owned by optimizer

    int inputs() const { return n_inputs; }
    int values() const { return n_values; }

    int operator()(const InputType& x, ValueType& fvec) const {
        fill_residuals(residuals, x, n_values, fvec, params, rbuf);
        if (aborted && aborted->load(std::memory_order_relaxed)) return -1;
        return 0;
    }

    int df(const InputType& x, JacobianType& fjac) const {
        for (int i = 0; i < static_cast<int>(x.size()); ++i)
            params[static_cast<std::size_t>(i)] = x[i];
        // Re-zero the reused buffer: the jacobian closure may return early (abort)
        // leaving entries unfilled, which must read as a defined 0.0.
        std::fill(jbuf.begin(), jbuf.end(), 0.0);
        jacobian(params, jbuf);
        if (aborted && aborted->load(std::memory_order_relaxed)) return -1;
        fjac.resize(n_values, n_inputs);
        for (int i = 0; i < n_values; ++i) {
            for (int j = 0; j < n_inputs; ++j) {
                double value =
                    jbuf[static_cast<std::size_t>(i) *
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

    // Reset the abort flag before each optimize() so a problem that gets
    // re-optimized after an aborted run starts fresh (docs/22 §5.1 step 2).
    if (problem.aborted) {
        problem.aborted->store(false, std::memory_order_relaxed);
    }

    // Size the reusable scratch buffers. resize() only reallocates when the capacity
    // is insufficient, so after the first few fits these never reallocate (m is fixed
    // for a search; k grows to the largest tree seen, then stays) — eliminating the
    // per-fit page faults that serialized island workers (docs/23 §4).
    params_.resize(static_cast<std::size_t>(k));
    rbuf_.resize(static_cast<std::size_t>(m));

    if (problem.jacobian) {
        jbuf_.resize(static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
        AnalyticFunctor functor{problem.residuals, problem.jacobian,
                                problem.aborted, k, m, params_, rbuf_, jbuf_};
        Eigen::LevenbergMarquardt<AnalyticFunctor> lm(functor);
        if (maxfev > 0) lm.parameters.maxfev = maxfev;
        const Eigen::LevenbergMarquardtSpace::Status status =
            run_lm(lm, x, stop_requested);
        finalize(lm, status, x, result);
    } else {
        NumericFunctor functor{problem.residuals, problem.aborted, k, m,
                               params_, rbuf_};
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
