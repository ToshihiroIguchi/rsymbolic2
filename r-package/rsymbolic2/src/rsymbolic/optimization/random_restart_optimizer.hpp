#pragma once

#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// Derivative-free baseline optimizer.
//
// From several starting points (the initial guess, plus perturbed variants), it runs
// a simple stochastic hill-climb -- a (1+1)-style random search with an annealing step
// size -- and keeps the best result across all restarts. Non-finite objective values
// (NaN / +/-Inf) are treated as +Inf and therefore never accepted.
//
// Purpose in the roadmap: this is the lower-bound reference backend. It requires no
// gradients and no external libraries, so it lets the whole pipeline (objective ->
// optimize -> result) run and be tested before a gradient-based backend (Eigen MINPACK
// LM) is connected. It is also a candidate in the Phase 2 comparison as the
// "random-restart-only" baseline.
class RandomRestartOptimizer final : public ConstantOptimizer {
public:
    explicit RandomRestartOptimizer(OptimizerConfig config = {});

    OptimizationResult optimize(const OptimizationProblem& problem,
                                const StopRequested& stop_requested) const override;
    using ConstantOptimizer::optimize;  // keep the no-deadline convenience overload
    std::string name() const override;

private:
    OptimizerConfig config_;
};

}  // namespace rsymbolic
