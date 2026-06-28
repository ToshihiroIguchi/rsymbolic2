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

// Optional analytic Jacobian: given the constants, fill `jacobian` (size m*k) in
// row-major order with d r_i / d c_j at jacobian[i*k + j]. When supplied (e.g. from
// forward-mode dual numbers), a least-squares backend can use it instead of numerical
// differentiation. When left null, backends fall back to numerical differentiation.
using JacobianFunction =
    std::function<void(const std::vector<double>& params,
                       std::vector<double>& jacobian)>;

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
};

// Outcome of an optimization run.
struct OptimizationResult {
    std::vector<double> constants;  // best constants found
    double loss = 0.0;              // sum of squared residuals at `constants`
    bool success = false;           // true iff a finite-loss solution was found
    std::size_t evaluations = 0;    // number of residual-function evaluations
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
    virtual OptimizationResult optimize(const OptimizationProblem& problem,
                                        const StopRequested& stop_requested) const = 0;

    // Convenience overload: no deadline. Used by tests and any caller that does not
    // impose a time limit. Non-virtual; forwards with an always-false predicate, so
    // every existing call site that passes no predicate is unchanged.
    OptimizationResult optimize(const OptimizationProblem& problem) const {
        return optimize(problem, [] { return false; });
    }

    // Human-readable backend name, for logging and benchmark reporting.
    virtual std::string name() const = 0;
};

}  // namespace rsymbolic
