#include "rsymbolic/optimization/self_lm_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace rsymbolic {

namespace {

// A non-finite residual or Jacobian entry would poison the normal equations. We clamp
// such values to a large finite magnitude so the solver treats that point as a very poor
// fit and steps away, rather than producing NaNs in A = JᵀJ. Matches the clamping value
// used by the Eigen backend's fill_residuals (eigen_lm_optimizer.cpp).
constexpr double kLargeResidual = 1.0e10;

double clamp_finite(double v) { return std::isfinite(v) ? v : kLargeResidual; }

// Levenberg-Marquardt control constants. The damping factor lambda is increased by
// kLambdaUp on a rejected step and decreased by kLambdaDown on an accepted one, the
// classic Marquardt schedule. kMaxInner bounds the per-iteration damping search so a
// stuck problem terminates instead of looping.
constexpr double kLambdaInit = 1.0e-3;  // lambda0 = kLambdaInit * max diag(A)
constexpr double kLambdaUp = 10.0;
constexpr double kLambdaDown = 0.1;
constexpr int kMaxInner = 30;

// Convergence tolerances (relative). The loop stops when the gradient is tiny, the step
// is tiny relative to the parameters, or the relative SSE reduction is tiny.
constexpr double kGTol = 1.0e-12;  // ‖g‖∞
constexpr double kXTol = 1.0e-12;  // ‖δ‖ / ‖p‖
constexpr double kFTol = 1.0e-12;  // relative SSE reduction

// Evaluate residuals at `p` into `out` (clamped to finite in place) and return the sum
// of squared (clamped) residuals. `out` must be pre-sized to m.
double eval_sse(const OptimizationProblem& problem, const std::vector<double>& p,
                std::vector<double>& out, std::size_t m) {
    problem.residuals(p, out);
    double sse = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        const double v = clamp_finite(out[i]);
        out[i] = v;
        sse += v * v;
    }
    return sse;
}

// Solve the SPD system M x = b for the k×k matrix M (row-major, supplied in `aug`) via
// an in-place Cholesky factorization. `aug` is overwritten with the factor; `rhs` holds
// b on entry and is overwritten with the solution x. Returns false if M is not
// positive-definite (a non-positive pivot), signalling the caller to increase damping.
bool solve_spd(std::vector<double>& aug, std::vector<double>& rhs, int k) {
    // Cholesky: aug = L Lᵀ, L stored in the lower triangle of `aug`.
    for (int j = 0; j < k; ++j) {
        double diag = aug[static_cast<std::size_t>(j) * k + j];
        for (int p = 0; p < j; ++p) {
            const double ljp = aug[static_cast<std::size_t>(j) * k + p];
            diag -= ljp * ljp;
        }
        if (!(diag > 0.0)) return false;
        const double ljj = std::sqrt(diag);
        aug[static_cast<std::size_t>(j) * k + j] = ljj;
        for (int i = j + 1; i < k; ++i) {
            double s = aug[static_cast<std::size_t>(i) * k + j];
            for (int p = 0; p < j; ++p) {
                s -= aug[static_cast<std::size_t>(i) * k + p] *
                     aug[static_cast<std::size_t>(j) * k + p];
            }
            aug[static_cast<std::size_t>(i) * k + j] = s / ljj;
        }
    }
    // Forward solve L y = b (b = rhs), overwriting rhs with y.
    for (int i = 0; i < k; ++i) {
        double s = rhs[static_cast<std::size_t>(i)];
        for (int p = 0; p < i; ++p) {
            s -= aug[static_cast<std::size_t>(i) * k + p] *
                 rhs[static_cast<std::size_t>(p)];
        }
        rhs[static_cast<std::size_t>(i)] =
            s / aug[static_cast<std::size_t>(i) * k + i];
    }
    // Back solve Lᵀ x = y, overwriting rhs with x.
    for (int i = k - 1; i >= 0; --i) {
        double s = rhs[static_cast<std::size_t>(i)];
        for (int p = i + 1; p < k; ++p) {
            s -= aug[static_cast<std::size_t>(p) * k + i] *
                 rhs[static_cast<std::size_t>(p)];
        }
        rhs[static_cast<std::size_t>(i)] =
            s / aug[static_cast<std::size_t>(i) * k + i];
    }
    return true;
}

}  // namespace

SelfLMOptimizer::SelfLMOptimizer(OptimizerConfig config) : config_(config) {}

std::string SelfLMOptimizer::name() const { return "SelfLMOptimizer"; }

OptimizationResult SelfLMOptimizer::optimize(
    const OptimizationProblem& problem, const StopRequested& stop_requested) const {
    if (!problem.residuals) {
        throw std::invalid_argument(
            "SelfLMOptimizer: problem.residuals must not be null");
    }

    const int k = static_cast<int>(problem.initial_constants.size());
    const std::size_t m = problem.num_residuals;

    OptimizationResult result;
    result.constants = problem.initial_constants;

    // Degenerate case: no tunable constants. Nothing to solve; just report the loss.
    if (k == 0) {
        result.loss = sum_of_squared_residuals(problem, problem.initial_constants);
        result.success = std::isfinite(result.loss);
        result.evaluations = 1;
        return result;
    }

    // Reset the abort flag before each optimize() so a problem re-optimized after an
    // aborted run starts fresh (docs/22 §5.1 step 2).
    if (problem.aborted) {
        problem.aborted->store(false, std::memory_order_relaxed);
    }

    // Size the reusable scratch. resize() only reallocates when capacity is
    // insufficient, so after the first few fits these never reallocate — no per-fit
    // page faults to serialize island workers (docs/23 §4).
    const std::size_t ku = static_cast<std::size_t>(k);
    params_.resize(ku);
    trial_.resize(ku);
    rbuf_.resize(m);
    trial_rbuf_.resize(m);
    jbuf_.resize(m * ku);
    ata_.resize(ku * ku);
    aug_.resize(ku * ku);
    g_.resize(ku);
    delta_.resize(ku);

    params_ = problem.initial_constants;

    const bool aborted_set = static_cast<bool>(problem.aborted);
    auto aborted = [&]() {
        return aborted_set &&
               problem.aborted->load(std::memory_order_relaxed);
    };

    std::size_t nfev = 0;
    double sse = eval_sse(problem, params_, rbuf_, m);
    ++nfev;

    const std::size_t max_iter =
        config_.max_iterations > 0 ? config_.max_iterations : 200;
    double lambda = -1.0;  // set from max diag(A) on the first iteration

    for (std::size_t iter = 0; iter < max_iter && !aborted(); ++iter) {
        if (stop_requested()) break;

        // --- Jacobian J (m×k, row-major), clamped to finite ---------------------
        if (problem.jacobian) {
            // Re-zero: the closure may early-return on abort, leaving entries unset,
            // which must read as a defined 0.0 (matches AnalyticFunctor::df).
            std::fill(jbuf_.begin(), jbuf_.end(), 0.0);
            problem.jacobian(params_, jbuf_);
            if (aborted()) break;
            for (std::size_t e = 0; e < jbuf_.size(); ++e)
                jbuf_[e] = clamp_finite(jbuf_[e]);
        } else {
            // Forward-difference fallback when no analytic Jacobian is supplied
            // (mirrors the Eigen backend's NumericalDiff path). Reuses trial_ /
            // trial_rbuf_ as perturbation scratch; both are rebuilt for the step below.
            constexpr double kSqrtEps = 1.4901161193847656e-08;  // sqrt(DBL_EPSILON)
            for (int a = 0; a < k; ++a) {
                const double p0 = params_[static_cast<std::size_t>(a)];
                const double h = kSqrtEps * std::max(std::fabs(p0), 1.0);
                trial_ = params_;
                trial_[static_cast<std::size_t>(a)] = p0 + h;
                problem.residuals(trial_, trial_rbuf_);
                ++nfev;
                for (std::size_t i = 0; i < m; ++i) {
                    const double rp = clamp_finite(trial_rbuf_[i]);
                    jbuf_[i * ku + static_cast<std::size_t>(a)] =
                        (rp - rbuf_[i]) / h;
                }
            }
            if (aborted()) break;
        }

        // --- Normal equations: A = JᵀJ (k×k), g = Jᵀr (k) -----------------------
        for (int a = 0; a < k; ++a) {
            double ga = 0.0;
            for (std::size_t i = 0; i < m; ++i)
                ga += jbuf_[i * ku + static_cast<std::size_t>(a)] * rbuf_[i];
            g_[static_cast<std::size_t>(a)] = ga;
            for (int b = a; b < k; ++b) {
                double s = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    s += jbuf_[i * ku + static_cast<std::size_t>(a)] *
                         jbuf_[i * ku + static_cast<std::size_t>(b)];
                }
                ata_[static_cast<std::size_t>(a) * ku + static_cast<std::size_t>(b)] = s;
                ata_[static_cast<std::size_t>(b) * ku + static_cast<std::size_t>(a)] = s;
            }
        }

        // Gradient convergence (g = Jᵀr is the gradient of ½·SSE).
        double gmax = 0.0;
        for (int a = 0; a < k; ++a)
            gmax = std::max(gmax, std::fabs(g_[static_cast<std::size_t>(a)]));
        if (gmax <= kGTol) break;

        if (lambda < 0.0) {
            double maxdiag = 0.0;
            for (int a = 0; a < k; ++a)
                maxdiag = std::max(
                    maxdiag, ata_[static_cast<std::size_t>(a) * ku +
                                  static_cast<std::size_t>(a)]);
            lambda = kLambdaInit * (maxdiag > 0.0 ? maxdiag : 1.0);
        }

        // --- Damping search: find an accepted step, adjusting lambda ------------
        const double sse_before = sse;
        bool accepted = false;
        for (int inner = 0; inner < kMaxInner; ++inner) {
            if (stop_requested()) break;

            // aug = A + lambda·diag(A). Floor the per-column scale with 1.0 so a null
            // column (A[a][a] == 0) still yields a positive-definite diagonal.
            for (int a = 0; a < k; ++a) {
                for (int b = 0; b < k; ++b) {
                    aug_[static_cast<std::size_t>(a) * ku +
                         static_cast<std::size_t>(b)] =
                        ata_[static_cast<std::size_t>(a) * ku +
                             static_cast<std::size_t>(b)];
                }
                const double d = ata_[static_cast<std::size_t>(a) * ku +
                                      static_cast<std::size_t>(a)];
                aug_[static_cast<std::size_t>(a) * ku + static_cast<std::size_t>(a)] +=
                    lambda * (d > 0.0 ? d : 1.0);
            }

            // delta solves (A + λ·diag(A)) δ = -g (rhs = -g, solved in place).
            for (int a = 0; a < k; ++a)
                delta_[static_cast<std::size_t>(a)] =
                    -g_[static_cast<std::size_t>(a)];
            if (!solve_spd(aug_, delta_, k)) {
                lambda *= kLambdaUp;
                continue;
            }

            for (int a = 0; a < k; ++a)
                trial_[static_cast<std::size_t>(a)] =
                    params_[static_cast<std::size_t>(a)] +
                    delta_[static_cast<std::size_t>(a)];

            const double sse_trial = eval_sse(problem, trial_, trial_rbuf_, m);
            ++nfev;
            if (aborted()) break;

            if (sse_trial < sse) {
                std::swap(params_, trial_);
                std::swap(rbuf_, trial_rbuf_);
                sse = sse_trial;
                lambda *= kLambdaDown;
                accepted = true;
                break;
            }
            lambda *= kLambdaUp;
        }

        if (!accepted || aborted()) break;

        // Relative SSE reduction convergence.
        if (sse_before - sse <= kFTol * sse_before) break;

        // Step-size convergence: ‖δ‖ small relative to ‖p‖.
        double dnorm = 0.0;
        double pnorm = 0.0;
        for (int a = 0; a < k; ++a) {
            dnorm += delta_[static_cast<std::size_t>(a)] *
                     delta_[static_cast<std::size_t>(a)];
            pnorm += params_[static_cast<std::size_t>(a)] *
                     params_[static_cast<std::size_t>(a)];
        }
        if (std::sqrt(dnorm) <= kXTol * (std::sqrt(pnorm) + kXTol)) break;
    }

    result.constants = params_;
    result.loss = sse;
    result.success = std::isfinite(sse);
    result.evaluations = nfev;
    return result;
}

}  // namespace rsymbolic
