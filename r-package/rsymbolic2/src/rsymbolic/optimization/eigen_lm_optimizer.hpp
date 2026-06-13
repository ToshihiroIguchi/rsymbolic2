#pragma once

#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// Gradient-based backend using the Levenberg-Marquardt implementation from Eigen's
// unsupported NonLinearOptimization module (a port of the long-established MINPACK
// routines). The Jacobian is obtained by numerical differentiation of the residual
// function (Eigen::NumericalDiff); a forward-mode dual-number Jacobian can replace it
// later behind this same backend without changing the interface.
//
// This is a leading candidate for k <= 5 constant optimization (see docs/07). It is
// intentionally hidden behind the ConstantOptimizer abstraction: Eigen is confined to
// the .cpp, so swapping in Ceres TinySolver or full Ceres later requires only adding a
// new derived class, with no change to callers.
//
// Note: this header does NOT include any Eigen header, keeping Eigen out of the public
// interface and off every consumer of the optimizer abstraction.
class EigenLMOptimizer final : public ConstantOptimizer {
public:
    explicit EigenLMOptimizer(OptimizerConfig config = {});

    OptimizationResult optimize(const OptimizationProblem& problem,
                                const StopRequested& stop_requested) const override;
    using ConstantOptimizer::optimize;  // keep the no-deadline convenience overload
    std::string name() const override;

private:
    OptimizerConfig config_;
};

}  // namespace rsymbolic
