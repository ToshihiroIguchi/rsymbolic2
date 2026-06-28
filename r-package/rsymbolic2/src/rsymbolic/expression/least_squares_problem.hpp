// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>
#ifdef RSYMBOLIC2_PROFILE
#include <chrono>
#include <cstdint>
#endif

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/multi_dual.hpp"
#include "rsymbolic/expression/soa_eval.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// ---------------------------------------------------------------------------
// Optional within-fit profiler — DIAGNOSTIC ONLY (compiled out unless
// RSYMBOLIC2_PROFILE is defined; see evolutionary_search.cpp). Splits fit() time
// into its residual vs. Jacobian closures, which the per-phase fit timers cannot
// see. The Jacobian runs k forward-mode Dual passes over all m points (below), so
// this number decides whether to attack the Jacobian's k-passes (batched dual) or
// the per-eval cost (Float32). The closures run concurrently on island threads, so
// the accumulators are atomic; the contention is negligible because each call spans
// m points. Work-nanoseconds summed across threads.
#ifdef RSYMBOLIC2_PROFILE
inline std::atomic<std::uint64_t> g_resid_ns{0};
inline std::atomic<std::uint64_t> g_resid_calls{0};
inline std::atomic<std::uint64_t> g_jac_ns{0};
inline std::atomic<std::uint64_t> g_jac_calls{0};
struct ClosureTimer {
    std::atomic<std::uint64_t>& ns;
    std::atomic<std::uint64_t>& calls;
    std::chrono::steady_clock::time_point t0;
    ClosureTimer(std::atomic<std::uint64_t>& n, std::atomic<std::uint64_t>& c)
        : ns(n), calls(c), t0(std::chrono::steady_clock::now()) {}
    ~ClosureTimer() {
        ns.fetch_add(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count()),
            std::memory_order_relaxed);
        calls.fetch_add(1, std::memory_order_relaxed);
    }
};
#  define RSYM_CLOSURE_TIMER(ns, calls) ClosureTimer rsym_ct_(ns, calls)
#else
#  define RSYM_CLOSURE_TIMER(ns, calls)
#endif

// Number of data points to process between stop-predicate polls in the residual
// and Jacobian closures. A compile-time constant (not a public knob). At 256
// points per check, the overhead is one std::function call per ~256 evaluations
// in the worst case; the coarse granularity prevents excessive polling overhead.
// See docs/22 Phase 1.
static constexpr std::size_t kStride = 256;

// Sentinel residual value used when a stop is triggered mid-evaluation. Matches
// the non-finite clamping value used by the self-LM optimizer (self_lm_optimizer.cpp).
static constexpr double kLargeResidualSentinel = 1.0e10;

// An immutable dataset (input rows + targets) shared across every per-candidate
// problem in a search. Sharing it by shared_ptr means building a problem costs one
// tree copy, not a full dataset copy — the dataset copy was ~m heap allocations per
// fit() and the dominant source of multi-island heap-lock contention (docs/23 §4).
struct Dataset {
    std::vector<std::vector<double>> X;     // X[i] is the feature vector for point i (row-major)
    std::vector<double> y;                  // y[i] corresponds to X[i]
    std::vector<std::vector<double>> Xcol;  // column-major view: Xcol[j][i] = X[i][j]
    // Per-point sqrt(weight). Empty => unweighted (every point counts equally); the hot
    // paths skip the multiply entirely in that case. When non-empty (size m), the
    // residual r_i and its Jacobian row are scaled by sqrt_weights[i], so the optimiser's
    // SSE = sum_i sqrt(w_i)^2 * (pred_i - y_i)^2 = sum_i w_i * (pred_i - y_i)^2 — the
    // weighted least-squares reduction (PySR `weights`). sqrt(w) is precomputed once here
    // so the per-point evaluators multiply rather than call sqrt on the hot path.
    std::vector<double> sqrt_weights;

    Dataset() = default;
    // Build the column-major view once at construction; the Dataset is shared immutably
    // across islands afterwards, so the transpose is computed exactly once per search.
    // Xcol is what the SoA point-batched residual/Jacobian evaluators consume
    // (soa_eval.hpp); row-major X is kept for the scalar callers (final prediction etc.).
    // `w_` are per-point weights (empty => unweighted); sqrt(w_i) is precomputed into
    // sqrt_weights once, here, for the same once-per-search reason as the transpose.
    Dataset(std::vector<std::vector<double>> X_, std::vector<double> y_,
            std::vector<double> w_ = {})
        : X(std::move(X_)), y(std::move(y_)) {
        const std::size_t m = X.size();
        const std::size_t nf = m ? X[0].size() : 0;
        Xcol.assign(nf, std::vector<double>(m));
        for (std::size_t i = 0; i < m; ++i)
            for (std::size_t j = 0; j < nf; ++j) Xcol[j][i] = X[i][j];
        if (!w_.empty()) {
            sqrt_weights.resize(w_.size());
            for (std::size_t i = 0; i < w_.size(); ++i)
                sqrt_weights[i] = std::sqrt(w_[i]);
        }
    }
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
        // Per-problem scratch reused across all residual/Jacobian evaluations of
        // this problem (docs/23 §4, docs/25). The problem (and its Model) is created
        // locally per fit() call, never shared across threads, and the residual and
        // Jacobian closures are called sequentially within one optimize() — so
        // reusing these buffers across calls is safe. The residual and Jacobian
        // pools/stacks never alias (different closures, separate buffers). Sized on
        // first use; no allocation on the hot path thereafter.
        std::vector<double>  res_pool;    // SoA residual segment pool (tree.size()*P)
        std::vector<double*> res_stk;     // SoA residual evaluation stack
        std::vector<double>  jac_pool;    // SoA Jacobian pool (tree.size()*(1+N)*P)
        std::vector<double*> jac_stk;     // SoA Jacobian evaluation stack
        std::vector<double>  jac_coeff;   // SoA Jacobian per-op coefficient scratch (3*P)
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
        RSYM_CLOSURE_TIMER(g_resid_ns, g_resid_calls);
        const Dataset& d = *model->data;
        const std::size_t m = d.y.size();
        // SoA point-batched residual (soa_eval.hpp): process a tile of up to kStride
        // points per node instead of one point per evaluate() call. Bit-identical to the
        // old per-point evaluate<double> path (points are independent), so the search is
        // unchanged. Pool/stack held in the Model and reused across optimize() iterations
        // (docs/23 §4, docs/25). Tiling at kStride also preserves the stop-poll cadence.
        std::vector<double>&  pool = model->res_pool;
        std::vector<double*>& stk  = model->res_stk;
        // Per-point weighting (PySR `weights`): scale each residual by sqrt(w_i) so the
        // optimiser minimises the weighted SSE. nullptr => unweighted (skip the multiply).
        const double* sw = d.sqrt_weights.empty() ? nullptr : d.sqrt_weights.data();
        for (std::size_t lo = 0; lo < m; lo += kStride) {
            // Poll stop predicate at each tile boundary (docs/22 §5.1 step 3).
            if (aborted && stop && stop()) {
                aborted->store(true, std::memory_order_relaxed);
                // Fill remaining residuals with the sentinel so Eigen sees a
                // well-defined (large) residual rather than an uninitialised value.
                for (std::size_t j = lo; j < m; ++j)
                    residuals[j] = kLargeResidualSentinel;
                return;
            }
            const std::size_t P = std::min(kStride, m - lo);
            const double* pred = evaluate_soa_residual(
                model->tree, d.Xcol, params.data(), lo, P, pool, stk);
            for (std::size_t p = 0; p < P; ++p) {
                double r = pred[p] - d.y[lo + p];
                if (sw) r *= sw[lo + p];
                residuals[lo + p] = r;
            }
        }
    };

    problem.jacobian = [model, stop, aborted](const std::vector<double>& params,
                                             std::vector<double>& jacobian) {
        RSYM_CLOSURE_TIMER(g_jac_ns, g_jac_calls);
        constexpr int N = kJacobianBlockWidth;
        const int k = model->k;
        const Dataset& d = *model->data;
        const std::size_t m = d.y.size();
        // SoA vector-mode AD (soa_eval.hpp): differentiate N constant directions per pass
        // AND batch a tile of points per node. ceil(k/N) passes (the redundant-recompute
        // saving of docs/30 Optimization 1), now with the point loop inside each node so
        // the operator kernels can vectorise (Optimization 3). Each entry is
        // bit-identical to the scalar k-pass / per-point MultiDual path. Pool/stack/coeff
        // held in the Model and reused across optimize() iterations (docs/23 §4, docs/25).
        std::vector<double>&  pool  = model->jac_pool;
        std::vector<double*>& stk   = model->jac_stk;
        std::vector<double>&  coeff = model->jac_coeff;
        // Same per-point weighting as the residual closure: d(sqrt(w_i)*r_i)/dc = sqrt(w_i)
        // * dr_i/dc, so each Jacobian row is scaled by sqrt(w_i). nullptr => unweighted.
        const double* sw = d.sqrt_weights.empty() ? nullptr : d.sqrt_weights.data();
        for (int block = 0; block < k; block += N) {
            const int w = std::min(N, k - block);  // active columns in this block
            for (std::size_t lo = 0; lo < m; lo += kStride) {
                // Poll stop predicate at each tile boundary (docs/22 §5.1 step 3).
                if (aborted && stop && stop()) {
                    aborted->store(true, std::memory_order_relaxed);
                    // The caller (AnalyticFunctor::df) initialises the jac vector
                    // to 0.0, so unfilled entries are already defined. Just return;
                    // the abort flag causes df() to return -1 and Eigen discards
                    // the (partial, untrustworthy) matrix.
                    return;
                }
                const std::size_t P = std::min(kStride, m - lo);
                const double* seg = evaluate_soa_jacobian<N>(
                    model->tree, d.Xcol, params.data(), k, block, lo, P, pool, stk, coeff);
                // d r_i / d c_(block+c) == d prediction_i / d c_(block+c) (y_i constant).
                for (int c = 0; c < w; ++c) {
                    const double* gc = seg + static_cast<std::size_t>(1 + c) * P;
                    for (std::size_t p = 0; p < P; ++p) {
                        double v = gc[p];
                        if (sw) v *= sw[lo + p];
                        jacobian[(lo + p) * static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(block + c)] = v;
                    }
                }
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
    StopRequested stop = {},
    std::vector<double> weights = {}) {
    auto data = std::make_shared<const Dataset>(
        Dataset{std::move(X), std::move(y), std::move(weights)});
    return make_least_squares_problem(std::move(tree), std::move(data),
                                      std::move(initial), std::move(stop));
}

}  // namespace rsymbolic
