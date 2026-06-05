#include "rsymbolic/optimization/optimizer_factory.hpp"

#include <stdexcept>

#include "rsymbolic/optimization/random_restart_optimizer.hpp"

namespace rsymbolic {

std::unique_ptr<ConstantOptimizer> OptimizerFactory::create(
    OptimizerType type, const OptimizerConfig& config) {
    switch (type) {
        case OptimizerType::RandomRestart:
            return std::make_unique<RandomRestartOptimizer>(config);
        case OptimizerType::EigenLM:
        case OptimizerType::CeresTinySolver:
        case OptimizerType::Ceres:
            throw std::runtime_error(
                "OptimizerFactory: requested backend is not yet implemented "
                "(planned for Phase 2)");
    }
    throw std::runtime_error("OptimizerFactory: unknown optimizer type");
}

OptimizerType OptimizerFactory::from_string(const std::string& name) {
    if (name == "random_restart") return OptimizerType::RandomRestart;
    if (name == "eigen_lm") return OptimizerType::EigenLM;
    if (name == "ceres_tiny") return OptimizerType::CeresTinySolver;
    if (name == "ceres") return OptimizerType::Ceres;
    throw std::invalid_argument(
        "OptimizerFactory: unknown optimizer name '" + name + "'");
}

}  // namespace rsymbolic
