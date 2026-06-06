#pragma once

#include <memory>
#include <string>

#include "rsymbolic/optimization/constant_optimizer.hpp"

namespace rsymbolic {

// The set of constant-optimization backends. Only RandomRestart is implemented in the
// walking skeleton; the others are declared now to fix the extension points and to
// make the "swappable backend" seam explicit. Requesting an unimplemented backend
// throws (see OptimizerFactory::create).
enum class OptimizerType {
    RandomRestart,    // implemented: derivative-free baseline / lower bound
    EigenLM,          // planned (Phase 2): Eigen MINPACK Levenberg-Marquardt
    CeresTinySolver,  // planned (Phase 2): vendored Ceres TinySolver (header-only)
    Ceres,            // planned (Phase 2): full Ceres Solver
};

// Creates ConstantOptimizer instances by type. Callers depend on this factory and the
// ConstantOptimizer interface, not on concrete classes, so a backend can be swapped by
// changing a single OptimizerType value.
class OptimizerFactory {
public:
    // Throws std::runtime_error if `type` names a backend that is not yet implemented.
    static std::unique_ptr<ConstantOptimizer> create(
        OptimizerType type, const OptimizerConfig& config = {});

    // Parses a backend name (e.g. "random_restart", "eigen_lm", "ceres_tiny",
    // "ceres"). Throws std::invalid_argument on an unknown name.
    static OptimizerType from_string(const std::string& name);
};

}  // namespace rsymbolic
