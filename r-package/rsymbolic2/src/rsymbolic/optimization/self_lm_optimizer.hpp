// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <random>
#include <vector>

#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// Allocation-free Levenberg-Marquardt backend — the default constant optimizer and the
// only least-squares backend the library ships. It implements the LM loop itself on top
// of the residual/Jacobian closures, performing zero per-fit heap allocation once its
// scratch buffers are sized, and pulls in no external linear-algebra library.
//
// Motivation (docs/23 §4, docs/25): an earlier Eigen-based backend (since removed)
// delegated the iteration to Eigen's unsupported NonLinearOptimization module, which
// made ~33-41 large dynamic heap allocations per fit() (QR / lmpar / norm temporaries).
// On Windows/MinGW those serialize on the process-wide allocator lock, collapsing
// multi-island scaling (4-thread cpu/wall ~1.5 instead of ~3.7). This backend removes
// them by keeping all working storage persistent across fits, and removing Eigen also
// drops the library's last third-party C++ dependency.
//
// Algorithm: normal-equations LM with Marquardt diagonal scaling. Each iteration forms
// A = JᵀJ (k×k) and g = Jᵀr (k) by point loops, then solves (A + λ·diag(A)) δ = -g with
// a hand-written Cholesky factorization (k is small, typically <= 5; <= ~10 is safe). The
// normal equations square the condition number, which is acceptable for the small k seen
// in symbolic regression; if a problem proves ill-conditioned the solve can be swapped
// for an allocation-free QR without changing this interface.
//
// Multi-start (SR.jl parity, docs ConstantOptimization.jl): the core LM loop runs from the
// tree's current constants (start 0), then `config.n_restarts` further starts from a
// multiplicative Gaussian perturbation of x0 (xt[i] = x0[i] * (1 + perturbation_scale *
// N(0,1))), keeping the best. perturbation_scale = 0.5 matches SR.jl's hardcoded T(1//2)
// restart scale (this is NOT PySR's perturbation_factor=0.129, which is the mutate_constant
// kernel scale — a different mechanism). n_restarts=0 reduces to a single LM from x0
// (the previous behaviour). Because start 0 is monotone from x0, the returned best is always
// <= the SSE at x0, so SR.jl's "restore x0 unless improved" baseline guard holds automatically.
//
// Determinism: the LM loop itself uses only floating-point arithmetic; the only RNG is the
// restart perturbation, drawn from `rng_` which is seeded deterministically (per island in the
// search, via OptimizerConfig::seed). A fit is therefore reproducible and independent of
// thread count.
//
// This header includes no external math library: the implementation is confined to the
// .cpp (pure C++/STL), keeping the optimizer abstraction free of math dependencies.
class SelfLMOptimizer final : public ConstantOptimizer {
public:
    explicit SelfLMOptimizer(OptimizerConfig config = {});

    OptimizationResult optimize(const OptimizationProblem& problem,
                                const StopRequested& stop_requested) const override;
    using ConstantOptimizer::optimize;  // keep the no-deadline convenience overload
    std::string name() const override;

private:
    OptimizerConfig config_;

    // RNG for the restart perturbations only (the LM loop is RNG-free). Seeded from
    // config_.seed in the constructor; the search seeds each island's optimizer
    // deterministically so results are reproducible and thread-count independent.
    mutable std::mt19937_64 rng_;

    // Run the LM loop from a given start point `x0`, returning the optimized constants and
    // SSE and accumulating residual-function evaluations into `nfev` and Jacobian builds
    // into `njev`. Used once per start by optimize() (start 0 + n_restarts perturbed
    // starts). All scratch below is reused.
    OptimizationResult run_lm_from(const OptimizationProblem& problem,
                                   const std::vector<double>& x0,
                                   const StopRequested& stop_requested,
                                   std::size_t& nfev,
                                   std::size_t& njev) const;

    // Scratch buffers reused across optimize() calls so a fit performs no heap
    // allocation once these are sized (m is fixed for a search; k grows to the largest
    // tree seen, then stays). `mutable` because optimize() is const; thread-safe because
    // each island owns its own optimizer instance and never calls optimize()
    // concurrently on it. See docs/23 §4.
    mutable std::vector<double> params_;      // size k: current parameters
    mutable std::vector<double> trial_;       // size k: trial parameters p + delta
    mutable std::vector<double> rbuf_;        // size m: residuals at params_
    mutable std::vector<double> trial_rbuf_;  // size m: residuals at trial_
    mutable std::vector<double> jbuf_;        // size m*k: row-major Jacobian
    mutable std::vector<double> ata_;         // size k*k: A = JᵀJ
    mutable std::vector<double> aug_;         // size k*k: A + λ·diag(A) (Cholesky in place)
    mutable std::vector<double> g_;           // size k: g = Jᵀr
    mutable std::vector<double> delta_;       // size k: LM step
    mutable std::vector<double> xt_;          // size k: perturbed restart start point
};

}  // namespace rsymbolic
