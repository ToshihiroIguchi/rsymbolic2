// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/optimization/self_lm_optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace rsymbolic {

namespace {

// A non-finite residual or Jacobian entry would poison the normal equations. We clamp
// such values to a large finite magnitude so the solver treats that point as a very poor
// fit and steps away, rather than producing NaNs in A = JᵀJ. Matches the clamping value
// least_squares_problem.hpp's kLargeResidualSentinel so the value and stop paths agree.
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

SelfLMOptimizer::SelfLMOptimizer(OptimizerConfig config)
    : config_(config), rng_(config.seed) {}

std::string SelfLMOptimizer::name() const { return "SelfLMOptimizer"; }

OptimizationResult SelfLMOptimizer::optimize(
    const OptimizationProblem& problem, const StopRequested& stop_requested,
    OptimizerScratch& s) const {
    if (!problem.residuals) {
        throw std::invalid_argument(
            "SelfLMOptimizer: problem.residuals must not be null");
    }

    const int k = static_cast<int>(problem.initial_constants.size());
    const std::size_t m = problem.num_residuals;

    OptimizationResult result;
    result.constants = problem.initial_constants;

    // Degenerate case: no tunable constants. Nothing to solve; just report the loss.
    // The stop can fire part-way through even this single evaluation, in which case the
    // residual closure fills the remaining points with the large FINITE sentinel
    // (least_squares_problem.hpp) and the sum is a fabricated number that isfinite()
    // would happily accept. Report it as unsuccessful instead (docs/73).
    if (k == 0) {
        if (problem.aborted) {
            problem.aborted->store(false, std::memory_order_relaxed);
        }
        result.loss = sum_of_squared_residuals(problem, problem.initial_constants);
        const bool eval_aborted =
            problem.aborted && problem.aborted->load(std::memory_order_relaxed);
        if (eval_aborted) result.loss = std::numeric_limits<double>::infinity();
        result.success = std::isfinite(result.loss);
        result.evaluations = 1;
        return result;
    }

    // Reset the abort flag before each optimize() so a problem re-optimized after an
    // aborted run starts fresh (docs/22 §5.1 step 2). Reset once for the whole
    // multi-start sequence below, not per restart.
    if (problem.aborted) {
        problem.aborted->store(false, std::memory_order_relaxed);
    }

    // Size the caller-owned scratch. resize() only reallocates when capacity is
    // insufficient, so after the first few fits these never reallocate — no per-fit
    // page faults to serialize island workers (docs/23 §4). Sized once here; every
    // start (start 0 + restarts) reuses these buffers.
    const std::size_t ku = static_cast<std::size_t>(k);
    s.params.resize(ku);
    s.trial.resize(ku);
    s.rbuf.resize(m);
    s.trial_rbuf.resize(m);
    // With an analytic Jacobian only one row block is materialised at a time and the
    // normal equations are accumulated across blocks, so this is rows_per_block × k
    // rather than m × k (docs/65). The finite-difference fallback builds the Jacobian
    // column by column over all rows and still needs the full matrix.
    s.jac.resize(problem.jacobian ? jacobian_block_rows(problem, m) * ku : m * ku);
    s.ata.resize(ku * ku);
    s.aug.resize(ku * ku);
    s.g.resize(ku);
    s.delta.resize(ku);
    s.xt.resize(ku);

    // Multi-start (SR.jl _optimize_constants parity): start 0 runs LM from the tree's
    // current constants; each further start perturbs x0 multiplicatively by a Gaussian
    // and re-runs LM, keeping the best. Because start 0 is monotone from x0, `best.loss`
    // is always <= the SSE at x0, so SR.jl's "restore x0 unless improved" baseline guard
    // holds automatically — no member is ever made worse. n_restarts=0 reduces to a
    // single LM from x0 (the previous behaviour). The restart perturbation is the only
    // RNG use; rng_ is seeded deterministically (per island in the search).
    const std::vector<double>& x0 = problem.initial_constants;
    std::size_t nfev = 0;
    std::size_t njev = 0;
    OptimizationResult best = run_lm_from(problem, x0, stop_requested, s, nfev, njev);

    std::normal_distribution<double> gaussian(0.0, 1.0);
    const auto aborted = [&]() {
        return problem.aborted &&
               problem.aborted->load(std::memory_order_relaxed);
    };
    for (std::size_t r = 0; r < config_.n_restarts; ++r) {
        if (stop_requested() || aborted()) break;
        // xt[i] = x0[i] * (1 + perturbation_scale * N(0,1)), matching SR.jl's
        // xt = @. x0 * (1 + (1//2) * randn). A zero constant stays zero, as in SR.jl.
        for (std::size_t i = 0; i < ku; ++i)
            s.xt[i] = x0[i] * (1.0 + config_.perturbation_scale * gaussian(rng_));
        OptimizationResult cand =
            run_lm_from(problem, s.xt, stop_requested, s, nfev, njev);
        if (cand.success && cand.loss < best.loss) best = std::move(cand);
    }

    best.evaluations = nfev;
    best.jacobian_evaluations = njev;
    return best;
}

// One LM run from start point `x0`. Reuses the caller's scratch `s` (already sized by
// optimize()). Accumulates residual-function evaluations into `nfev` and Jacobian builds
// into `njev`; the returned result carries the optimized constants and SSE but not the
// (caller-accumulated) evaluation counts.
OptimizationResult SelfLMOptimizer::run_lm_from(
    const OptimizationProblem& problem, const std::vector<double>& x0,
    const StopRequested& stop_requested, OptimizerScratch& s, std::size_t& nfev,
    std::size_t& njev) const {
    const int k = static_cast<int>(x0.size());
    const std::size_t m = problem.num_residuals;
    const std::size_t ku = static_cast<std::size_t>(k);

    s.params = x0;

    const bool aborted_set = static_cast<bool>(problem.aborted);
    auto aborted = [&]() {
        return aborted_set &&
               problem.aborted->load(std::memory_order_relaxed);
    };

    double sse = eval_sse(problem, s.params, s.rbuf, m);
    ++nfev;
    // If the stop fired DURING this first evaluation, `sse` mixes real residuals with the
    // large FINITE sentinel the residual closure writes on abort (least_squares_problem.hpp)
    // — a fabricated value, but one isfinite() accepts, so without this flag the result
    // below would be reported as a successful fit. Every LATER evaluation is already
    // discarded by an aborted() check before it can be accepted as the new best, which is
    // why this first one is the only unguarded case (docs/73).
    const bool first_eval_aborted = aborted();

    const std::size_t max_iter =
        config_.max_iterations > 0 ? config_.max_iterations : 200;
    double lambda = -1.0;  // set from max diag(A) on the first iteration

    // Rows per Jacobian block. The analytic path materialises one block at a time and
    // accumulates the normal equations across blocks; the finite-difference path builds
    // the whole matrix column by column and therefore uses a single all-rows block.
    const std::size_t block_rows =
        problem.jacobian ? jacobian_block_rows(problem, m) : m;

    for (std::size_t iter = 0; iter < max_iter && !aborted(); ++iter) {
        if (stop_requested()) break;

        // --- Normal equations: A = JᵀJ (k×k), g = Jᵀr (k) -----------------------
        //
        // The Jacobian is never materialised as an m×k matrix. It is produced one row
        // block at a time and reduced immediately, because A and g are the only things
        // the step needs. This is what keeps the LM working set O(block_rows × k)
        // instead of O(m × k) per worker (docs/65).
        //
        // BIT-EXACTNESS (load-bearing). Every accumulator below — g[a] and the upper
        // triangle of ata[a][b] — is a single scalar that receives its products in
        // strictly increasing row order, exactly as the previous whole-matrix reduction
        // did. Blocking interleaves *different* accumulators but never reassociates any
        // one of them, so each entry's floating-point summation sequence is unchanged and
        // the result is bit-identical for ANY block size. test_self_lm_optimizer asserts
        // this directly by sweeping the block size.
        for (std::size_t a = 0; a < ku; ++a) {
            s.g[a] = 0.0;
            for (std::size_t b = a; b < ku; ++b) s.ata[a * ku + b] = 0.0;
        }

        // One Jacobian build per LM iteration (either branch, however many blocks it
        // takes); counted for evaluation accounting only — never charged to the
        // max_evals budget.
        ++njev;
        bool abort_now = false;

        for (std::size_t lo = 0; lo < m; lo += block_rows) {
            const std::size_t rows = std::min(block_rows, m - lo);

            if (problem.jacobian) {
                // Re-zero: the closure may early-return on abort, leaving entries unset,
                // which must read as a defined 0.0 (matches AnalyticFunctor::df).
                std::fill(s.jac.begin(), s.jac.begin() + static_cast<std::ptrdiff_t>(rows * ku),
                          0.0);
                problem.jacobian(s.params, lo, rows, s.jac);
                if (aborted()) { abort_now = true; break; }
                for (std::size_t e = 0; e < rows * ku; ++e)
                    s.jac[e] = clamp_finite(s.jac[e]);
            } else {
                // Forward-difference fallback when no analytic Jacobian is supplied
                // (standard sqrt(eps)-scaled finite differences). Reuses trial /
                // trial_rbuf as perturbation scratch; both are rebuilt for the step
                // below. block_rows == m here, so this runs once and fills the whole
                // matrix, exactly as before.
                constexpr double kSqrtEps = 1.4901161193847656e-08;  // sqrt(DBL_EPSILON)
                for (std::size_t a = 0; a < ku; ++a) {
                    const double p0 = s.params[a];
                    const double h = kSqrtEps * std::max(std::fabs(p0), 1.0);
                    s.trial = s.params;
                    s.trial[a] = p0 + h;
                    problem.residuals(s.trial, s.trial_rbuf);
                    ++nfev;
                    for (std::size_t i = 0; i < m; ++i) {
                        const double rp = clamp_finite(s.trial_rbuf[i]);
                        s.jac[i * ku + a] = (rp - s.rbuf[i]) / h;
                    }
                }
                if (aborted()) { abort_now = true; break; }
            }

            for (std::size_t a = 0; a < ku; ++a) {
                double ga = s.g[a];
                for (std::size_t i = 0; i < rows; ++i)
                    ga += s.jac[i * ku + a] * s.rbuf[lo + i];
                s.g[a] = ga;
                for (std::size_t b = a; b < ku; ++b) {
                    double acc = s.ata[a * ku + b];
                    for (std::size_t i = 0; i < rows; ++i)
                        acc += s.jac[i * ku + a] * s.jac[i * ku + b];
                    s.ata[a * ku + b] = acc;
                }
            }
        }
        if (abort_now) break;

        // Mirror the accumulated upper triangle into the lower one.
        for (std::size_t a = 0; a < ku; ++a)
            for (std::size_t b = a + 1; b < ku; ++b)
                s.ata[b * ku + a] = s.ata[a * ku + b];

        // Gradient convergence (g = Jᵀr is the gradient of ½·SSE).
        double gmax = 0.0;
        for (std::size_t a = 0; a < ku; ++a)
            gmax = std::max(gmax, std::fabs(s.g[a]));
        if (gmax <= kGTol) break;

        if (lambda < 0.0) {
            double maxdiag = 0.0;
            for (std::size_t a = 0; a < ku; ++a)
                maxdiag = std::max(maxdiag, s.ata[a * ku + a]);
            lambda = kLambdaInit * (maxdiag > 0.0 ? maxdiag : 1.0);
        }

        // --- Damping search: find an accepted step, adjusting lambda ------------
        const double sse_before = sse;
        bool accepted = false;
        for (int inner = 0; inner < kMaxInner; ++inner) {
            if (stop_requested()) break;

            // aug = A + lambda·diag(A). Floor the per-column scale with 1.0 so a null
            // column (A[a][a] == 0) still yields a positive-definite diagonal.
            for (std::size_t a = 0; a < ku; ++a) {
                for (std::size_t b = 0; b < ku; ++b)
                    s.aug[a * ku + b] = s.ata[a * ku + b];
                const double d = s.ata[a * ku + a];
                s.aug[a * ku + a] += lambda * (d > 0.0 ? d : 1.0);
            }

            // delta solves (A + λ·diag(A)) δ = -g (rhs = -g, solved in place).
            for (std::size_t a = 0; a < ku; ++a) s.delta[a] = -s.g[a];
            if (!solve_spd(s.aug, s.delta, k)) {
                lambda *= kLambdaUp;
                continue;
            }

            for (std::size_t a = 0; a < ku; ++a)
                s.trial[a] = s.params[a] + s.delta[a];

            const double sse_trial = eval_sse(problem, s.trial, s.trial_rbuf, m);
            ++nfev;
            if (aborted()) break;

            if (sse_trial < sse) {
                std::swap(s.params, s.trial);
                std::swap(s.rbuf, s.trial_rbuf);
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
        for (std::size_t a = 0; a < ku; ++a) {
            dnorm += s.delta[a] * s.delta[a];
            pnorm += s.params[a] * s.params[a];
        }
        if (std::sqrt(dnorm) <= kXTol * (std::sqrt(pnorm) + kXTol)) break;
    }

    OptimizationResult res;
    res.constants = s.params;
    // An aborted first evaluation never produced a loss for x0, so there is no
    // "best so far" to report: +Inf / success=false, which fit() maps to kInf and the
    // caller keeps the pre-optimisation member. A stop that fires on any LATER iteration
    // still returns the genuinely-improved sse reached before it (docs/73).
    res.loss = first_eval_aborted ? std::numeric_limits<double>::infinity() : sse;
    res.success = std::isfinite(res.loss);
    // res.evaluations / res.jacobian_evaluations left 0: the caller (optimize)
    // accumulates nfev/njev across all starts.
    return res;
}

}  // namespace rsymbolic
