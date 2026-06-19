#pragma once

#include <vector>

#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// Allocation-free Levenberg-Marquardt backend. Unlike EigenLMOptimizer (which delegates
// the linear algebra and iteration to Eigen's unsupported NonLinearOptimization module),
// this backend implements the LM loop itself on top of the same residual/Jacobian
// closures, performing zero per-fit heap allocation once its scratch buffers are sized.
//
// Motivation (docs/23 §4, docs/25): Eigen's LevenbergMarquardt makes ~33-41 large
// dynamic heap allocations per fit() (QR / lmpar / norm temporaries). On Windows/MinGW
// those serialize on the process-wide allocator lock, collapsing multi-island scaling
// (4-thread cpu/wall ~1.5 instead of ~3.7). Reusing the Eigen solver object does not
// remove them — they are intrinsic to its algorithm. This backend removes them by
// keeping all working storage persistent across fits.
//
// Algorithm: normal-equations LM with Marquardt diagonal scaling. Each iteration forms
// A = JᵀJ (k×k) and g = Jᵀr (k) by point loops, then solves (A + λ·diag(A)) δ = -g with
// a hand-written Cholesky factorization (k is small, typically <= 5; <= ~10 is safe). The
// normal equations square the condition number, which is acceptable for the small k seen
// in symbolic regression; if a problem proves ill-conditioned the solve can be swapped
// for an allocation-free QR without changing this interface.
//
// Determinism is structural: the loop uses only floating-point arithmetic (no RNG), so a
// fit is reproducible and independent of thread count.
//
// Like the Eigen header, this one includes no external math library: the implementation
// is confined to the .cpp, keeping the optimizer abstraction free of math dependencies.
class SelfLMOptimizer final : public ConstantOptimizer {
public:
    explicit SelfLMOptimizer(OptimizerConfig config = {});

    OptimizationResult optimize(const OptimizationProblem& problem,
                                const StopRequested& stop_requested) const override;
    using ConstantOptimizer::optimize;  // keep the no-deadline convenience overload
    std::string name() const override;

private:
    OptimizerConfig config_;

    // Scratch buffers reused across optimize() calls so a fit performs no heap
    // allocation once these are sized (m is fixed for a search; k grows to the largest
    // tree seen, then stays). `mutable` because optimize() is const; thread-safe because
    // each island owns its own optimizer instance and never calls optimize()
    // concurrently on it (same contract as EigenLMOptimizer). See docs/23 §4.
    mutable std::vector<double> params_;      // size k: current parameters
    mutable std::vector<double> trial_;       // size k: trial parameters p + delta
    mutable std::vector<double> rbuf_;        // size m: residuals at params_
    mutable std::vector<double> trial_rbuf_;  // size m: residuals at trial_
    mutable std::vector<double> jbuf_;        // size m*k: row-major Jacobian
    mutable std::vector<double> ata_;         // size k*k: A = JᵀJ
    mutable std::vector<double> aug_;         // size k*k: A + λ·diag(A) (Cholesky in place)
    mutable std::vector<double> g_;           // size k: g = Jᵀr
    mutable std::vector<double> delta_;       // size k: LM step
};

}  // namespace rsymbolic
