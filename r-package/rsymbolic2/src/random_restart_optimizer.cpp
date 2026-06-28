// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/optimization/random_restart_optimizer.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace rsymbolic {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Evaluate the sum of squared residuals at `x`, mapping any non-finite residual to
// +Inf so it is never accepted as an improvement. Reuses `scratch` to avoid
// reallocating the residual buffer on every call. Increments the evaluation counter.
double evaluate(const OptimizationProblem& problem,
                const std::vector<double>& x,
                std::vector<double>& scratch,
                std::size_t& eval_count) {
    ++eval_count;
    problem.residuals(x, scratch);
    double sse = 0.0;
    for (const double r : scratch) {
        if (!std::isfinite(r)) {
            return kInf;
        }
        sse += r * r;
    }
    return sse;
}

}  // namespace

RandomRestartOptimizer::RandomRestartOptimizer(OptimizerConfig config)
    : config_(config) {}

std::string RandomRestartOptimizer::name() const {
    return "RandomRestartOptimizer";
}

OptimizationResult RandomRestartOptimizer::optimize(
    const OptimizationProblem& problem, const StopRequested& stop_requested) const {
    if (!problem.residuals) {
        throw std::invalid_argument(
            "RandomRestartOptimizer: problem.residuals must not be null");
    }

    const std::vector<double>& x0 = problem.initial_constants;
    const std::size_t k = x0.size();

    std::mt19937_64 rng(config_.seed);
    std::normal_distribution<double> gaussian(0.0, 1.0);
    std::vector<double> scratch(problem.num_residuals, 0.0);

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
        // Honour the search-loop deadline: stop between restarts and return the
        // best-so-far result (already tracked in `result`). See docs/20.
        if (stop_requested()) break;
        // Restart 0 starts exactly at x0; later restarts start from a perturbation of
        // x0 so the search explores different basins.
        std::vector<double> current =
            (restart == 0) ? x0 : perturb(x0, config_.perturbation_scale);
        double current_loss =
            evaluate(problem, current, scratch, result.evaluations);

        if (current_loss < result.loss) {
            result.loss = current_loss;
            result.constants = current;
        }

        for (std::size_t it = 0; it < config_.max_iterations; ++it) {
            if (stop_requested()) break;
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
                evaluate(problem, candidate, scratch, result.evaluations);

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
