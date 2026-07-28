// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace rsymbolic {

// Residual function: given a vector of constant values, fill `residuals` with the
// per-data-point residuals r_i (size must equal OptimizationProblem::num_residuals).
//
// A residual representation (not a scalar loss) is required because the least-squares
// backend (the in-house self-LM; Ceres later) operates on the residual vector and its
// Jacobian. The scalar loss used by derivative-free backends is derived as the sum of
// squared residuals (see sum_of_squared_residuals).
//
// This abstraction also keeps the optimizer decoupled from the expression tree: later,
// the tree + dataset + evaluator will produce this closure (constants -> residuals).
using ResidualFunction =
    std::function<void(const std::vector<double>& params,
                       std::vector<double>& residuals)>;

// Optional analytic Jacobian, supplied ONE ROW BLOCK AT A TIME: given the constants and
// a row range [row_lo, row_lo + rows), fill `jac_block` in row-major order so that
// d r_(row_lo + i) / d c_j lands at jac_block[i*k + j], for i in [0, rows). When supplied
// (e.g. from forward-mode dual numbers), a least-squares backend can use it instead of
// numerical differentiation. When left null, backends fall back to numerical
// differentiation.
//
// Blocked rather than whole-matrix because the normal-equations backends only ever reduce
// the Jacobian into JᵀJ (k×k) and Jᵀr (k); materialising the full m×k matrix made the
// working set O(rows × constants) per island, which dominated the process (docs/65). The
// caller sizes `jac_block` to at least rows*k; entries beyond `rows*k` must not be touched.
using JacobianFunction =
    std::function<void(const std::vector<double>& params,
                       std::size_t row_lo, std::size_t rows,
                       std::vector<double>& jac_block)>;

// A constant-optimization problem. `num_residuals` (m) is the number of residuals the
// function fills; `initial_constants` size (k) defines the dimensionality. k == 0 is
// valid (an expression with no tunable constants).
//
// `aborted` is an optional abort flag set by the residual/Jacobian closures when a
// stop predicate fires mid-evaluation (docs/22 Phase 1). When set, the optimizer
// stops at its next coarse boundary and returns the best result so far. The optimizer
// resets it to false at the start of each optimize() call.
struct OptimizationProblem {
    ResidualFunction residuals;             // required; must not be null
    JacobianFunction jacobian;              // optional; null => numerical Jacobian
    std::size_t num_residuals = 0;          // m
    std::vector<double> initial_constants;  // x0; size defines k
    // Abort flag shared between closures and the optimizer functor. May be null
    // (meaning the closures never abort early). Created by make_least_squares_problem
    // when a stop predicate is supplied; left null otherwise.
    std::shared_ptr<std::atomic<bool>> aborted;
    // How many rows the `jacobian` closure is asked for per call. Purely an
    // implementation knob: the normal equations accumulate across blocks with per-entry
    // running accumulators, so every block size produces bit-identical results (asserted
    // by test_self_lm_optimizer). make_least_squares_problem sets it to the evaluator's
    // own tile width so the block boundary and the stop-poll boundary coincide. 0 means
    // "one block covering every row", i.e. the pre-blocking behaviour.
    std::size_t jacobian_rows_per_block = 0;
};

// Rows per Jacobian block for a problem with `m` residuals: the problem's request,
// clamped to [1, m]; 0 (the default) means a single block covering all rows.
inline std::size_t jacobian_block_rows(const OptimizationProblem& problem,
                                       std::size_t m) {
    const std::size_t requested =
        problem.jacobian_rows_per_block == 0 ? m : problem.jacobian_rows_per_block;
    if (requested > m) return m;
    return requested < 1 ? 1 : requested;
}

// Outcome of an optimization run.
struct OptimizationResult {
    std::vector<double> constants;  // best constants found
    double loss = 0.0;              // sum of squared residuals at `constants`
    bool success = false;           // true iff a finite-loss solution was found
    std::size_t evaluations = 0;    // number of residual-function evaluations
    // Number of Jacobian builds (analytic closure calls, or full finite-difference
    // sweeps when no analytic Jacobian is supplied). Reported for evaluation
    // accounting only; never charged to the max_evals budget, which counts residual
    // evaluations (a finite-difference sweep's k residual calls are already in
    // `evaluations`). Accumulated across multi-start restarts like `evaluations`.
    std::size_t jacobian_evaluations = 0;
};

// Common configuration shared by backends. Each backend reads the fields it needs and
// ignores the rest. A single struct avoids a config-class explosion in the skeleton.
struct OptimizerConfig {
    std::uint64_t seed = 0;             // RNG seed (reproducibility is required)
    std::size_t n_restarts = 4;         // number of restarts (including the initial)
    std::size_t max_iterations = 100;   // local-search iters / solver evaluation cap
    double perturbation_scale = 0.5;    // relative scale of random perturbations
};

// Stop predicate polled by optimizers at their own coarse boundaries. When it returns
// true the optimizer stops early and returns the best result found so far. A callback
// (rather than a deadline value) keeps timing semantics — wall-clock, evaluation
// budget, or a test stub — entirely out of the optimizer interface; the search loop,
// which owns the deadline, captures it in the closure. See docs/20.
using StopRequested = std::function<bool()>;

// Scalar loss convention used across all backends: SSE = sum_i r_i^2. A non-finite
// residual maps the whole loss to +Inf so it is never accepted.
inline double sum_of_squared_residuals(const OptimizationProblem& problem,
                                       const std::vector<double>& params) {
    std::vector<double> residuals(problem.num_residuals, 0.0);
    problem.residuals(params, residuals);
    double sse = 0.0;
    for (const double r : residuals) {
        if (!std::isfinite(r)) {
            return std::numeric_limits<double>::infinity();
        }
        sse += r * r;
    }
    return sse;
}

// Reusable working storage for a constant-optimization backend, owned by the CALLER.
//
// Why the caller owns it (docs/65): the search creates one optimizer per island, but only
// as many islands run at once as there are OpenMP workers. Buffers that live on the
// optimizer are therefore allocated `n_populations` times while at most `n_threads` of
// them are ever in use — and the big ones are O(rows), so at the default 31 populations
// they dominate the whole process working set. Hoisting them here lets the search keep one
// scratch per WORKER and hand the right one to each fit.
//
// This carries no information between calls: every buffer is written before it is read
// within a single optimize(). That is what makes sharing one scratch across islands sound
// — which worker picks up which island is not observable in any result, so determinism
// under a dynamically scheduled island loop is unaffected. Backend RNG state is
// deliberately NOT here; it stays on the per-island optimizer object.
//
// Buffers are sized on first use and never shrink, so a fit performs no heap allocation
// once a worker has seen its first problem of a given size.
struct OptimizerScratch {
    std::vector<double> params;      // k: current parameters
    std::vector<double> trial;       // k: trial parameters p + delta
    std::vector<double> xt;          // k: perturbed restart start point
    std::vector<double> rbuf;        // m: residuals at params
    std::vector<double> trial_rbuf;  // m: residuals at trial
    // Jacobian working storage. With an analytic Jacobian this holds ONE row block
    // (rows_per_block × k, see OptimizationProblem::jacobian_rows_per_block), not the
    // whole m×k matrix — the normal equations are accumulated block by block (docs/65).
    // The finite-difference fallback still needs the full m×k and sizes it accordingly.
    std::vector<double> jac;
    std::vector<double> ata;         // k*k: A = JᵀJ
    std::vector<double> aug;         // k*k: A + λ·diag(A) (Cholesky in place)
    std::vector<double> g;           // k: g = Jᵀr
    std::vector<double> delta;       // k: LM step
};

// Strategy interface. All constant-optimization backends implement this. The search
// loop depends only on this interface, never on a concrete optimizer, so backends
// (RandomRestart, SelfLM now; Ceres TinySolver, full Ceres later) are interchangeable.
class ConstantOptimizer {
public:
    virtual ~ConstantOptimizer() = default;

    // Optimize the constants of `problem`. Implementations must not throw on a merely
    // poor result; they return success == false instead. They may throw only on
    // misuse (e.g. a null residual function).
    //
    // Implementations poll `stop_requested()` at their own coarse boundaries; when it
    // returns true they stop early and return the best result found so far (`success`
    // reflects whether that result is finite). The overshoot past the first true poll
    // is bounded by one unit of inner work (one residual evaluation / one LM step).
    //
    // `scratch` is caller-owned working storage (see OptimizerScratch). It must not be
    // shared between concurrent optimize() calls; the caller keeps one per worker
    // thread. Backends that need no scratch ignore it.
    virtual OptimizationResult optimize(const OptimizationProblem& problem,
                                        const StopRequested& stop_requested,
                                        OptimizerScratch& scratch) const = 0;

    // Convenience overloads for callers that do not manage scratch or a deadline (tests,
    // one-off fits). Each forwards to the full form with a locally owned scratch, so the
    // only cost of not managing one is the allocation this call makes and frees.
    OptimizationResult optimize(const OptimizationProblem& problem,
                                const StopRequested& stop_requested) const {
        OptimizerScratch scratch;
        return optimize(problem, stop_requested, scratch);
    }
    OptimizationResult optimize(const OptimizationProblem& problem) const {
        return optimize(problem, [] { return false; });
    }

    // Human-readable backend name, for logging and benchmark reporting.
    virtual std::string name() const = 0;
};

}  // namespace rsymbolic
