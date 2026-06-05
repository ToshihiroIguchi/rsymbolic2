#include "rsymbolic/optimization/random_restart_optimizer.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace rsymbolic {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Evaluate the objective, mapping non-finite results to +Inf so they are never
// accepted as an improvement. Increments the evaluation counter.
double evaluate(const ObjectiveFunction& objective,
                const std::vector<double>& x,
                std::size_t& eval_count) {
    ++eval_count;
    const double value = objective(x);
    return std::isfinite(value) ? value : kInf;
}

}  // namespace

RandomRestartOptimizer::RandomRestartOptimizer(OptimizerConfig config)
    : config_(config) {}

std::string RandomRestartOptimizer::name() const {
    return "RandomRestartOptimizer";
}

OptimizationResult RandomRestartOptimizer::optimize(
    const OptimizationProblem& problem) const {
    if (!problem.objective) {
        throw std::invalid_argument(
            "RandomRestartOptimizer: problem.objective must not be null");
    }

    const std::vector<double>& x0 = problem.initial_constants;
    const std::size_t k = x0.size();

    std::mt19937_64 rng(config_.seed);
    std::normal_distribution<double> gaussian(0.0, 1.0);

    OptimizationResult result;
    result.constants = x0;
    result.loss = kInf;
    result.evaluations = 0;

    // Relative perturbation: a component near zero still moves, because the step is
    // scaled by (|value| + 1).
    const auto perturb = [&](const std::vector<double>& base, double step_scale) {
        std::vector<double> out = base;
        for (std::size_t i = 0; i < k; ++i) {
            const double magnitude = std::fabs(base[i]) + 1.0;
            out[i] += gaussian(rng) * step_scale * magnitude;
        }
        return out;
    };

    const std::size_t n_restarts = config_.n_restarts == 0 ? 1 : config_.n_restarts;

    for (std::size_t restart = 0; restart < n_restarts; ++restart) {
        // Restart 0 starts exactly at x0; later restarts start from a perturbation of
        // x0 so the search explores different basins.
        std::vector<double> current =
            (restart == 0) ? x0 : perturb(x0, config_.perturbation_scale);
        double current_loss = evaluate(problem.objective, current, result.evaluations);

        if (current_loss < result.loss) {
            result.loss = current_loss;
            result.constants = current;
        }

        for (std::size_t it = 0; it < config_.max_iterations; ++it) {
            // Anneal the step from perturbation_scale down to ~0.1 of it, so early
            // iterations explore and later ones refine.
            const double frac =
                (config_.max_iterations > 1)
                    ? static_cast<double>(it) /
                          static_cast<double>(config_.max_iterations - 1)
                    : 0.0;
            const double step_scale = config_.perturbation_scale * (1.0 - 0.9 * frac);

            std::vector<double> candidate = perturb(current, step_scale);
            const double candidate_loss =
                evaluate(problem.objective, candidate, result.evaluations);

            if (candidate_loss < current_loss) {
                current = std::move(candidate);
                current_loss = candidate_loss;
                if (current_loss < result.loss) {
                    result.loss = current_loss;
                    result.constants = current;
                }
            }
        }
    }

    result.success = std::isfinite(result.loss);
    return result;
}

}  // namespace rsymbolic
