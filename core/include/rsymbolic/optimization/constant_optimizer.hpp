#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rsymbolic {

// Objective: maps a vector of constant values to a scalar loss (lower is better).
//
// This abstraction deliberately decouples the optimizer from the expression-tree
// representation. Later, the tree + dataset + evaluator will produce this closure
// (constants -> mean loss over the data). For the walking skeleton it lets us build,
// run, and test optimizers before any tree code exists, and it is exactly the seam
// that keeps optimizer backends swappable.
using ObjectiveFunction = std::function<double(const std::vector<double>&)>;

// A constant-optimization problem: an objective plus a starting point. The size of
// `initial_constants` defines the dimensionality (k). k == 0 is valid (an expression
// with no tunable constants); the objective is then evaluated on an empty vector.
struct OptimizationProblem {
    ObjectiveFunction objective;            // required; must not be null
    std::vector<double> initial_constants;  // x0; size defines k
};

// Outcome of an optimization run.
struct OptimizationResult {
    std::vector<double> constants;  // best constants found
    double loss = 0.0;              // objective(constants)
    bool success = false;           // true iff a finite-loss solution was found
    std::size_t evaluations = 0;    // number of objective evaluations performed
};

// Common configuration shared by backends. Individual backends read the fields they
// need and ignore the rest. Keeping a single struct avoids a config-class explosion
// in the skeleton; it can be specialized later if a backend needs unique parameters.
struct OptimizerConfig {
    std::uint64_t seed = 0;             // RNG seed (reproducibility is required)
    std::size_t n_restarts = 4;         // number of restarts (including the initial)
    std::size_t max_iterations = 100;   // local-search iterations per restart
    double perturbation_scale = 0.5;    // relative scale of random perturbations
};

// Strategy interface. All constant-optimization backends implement this. The search
// loop depends only on this interface, never on a concrete optimizer, so backends
// (RandomRestart now; Eigen MINPACK LM, Ceres TinySolver, full Ceres later) are
// interchangeable.
class ConstantOptimizer {
public:
    virtual ~ConstantOptimizer() = default;

    // Optimize the constants of `problem`. Implementations must not throw on a merely
    // poor result; they return success == false instead. They may throw only on
    // misuse (e.g. a null objective).
    virtual OptimizationResult optimize(const OptimizationProblem& problem) const = 0;

    // Human-readable backend name, e.g. for logging and benchmark reporting.
    virtual std::string name() const = 0;
};

}  // namespace rsymbolic
