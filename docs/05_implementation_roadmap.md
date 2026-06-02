# Phased Implementation Roadmap: rsymbolic2

**Date:** 2026-06-03  
**Total estimated duration:** 18–24 months (solo developer)

---

## Overview

```
Phase 1: C++ Expression Engine          (months 1–4)
Phase 2: Evolutionary Search Core       (months 5–9)
Phase 3: R Interface + Parallelism      (months 10–14)
Phase 4: Advanced Features              (months 15–20)
Phase 5: Python Interface + Publication (months 21–24)
```

Each phase ends with a deployable artifact and a defined benchmark milestone.

---

## Phase 1: C++ Expression Engine
**Duration:** 4 months  
**Goal:** A correct, efficient, unit-tested expression tree library.

### Deliverables
- `core/include/rsymbolic/expression/` — complete node, tree, operator_set, eval headers
- Dual number implementation for forward-mode AD
- Batch evaluation function (scalar and dual)
- Operator registry with 20+ standard operators
- `core/tests/` — unit tests with ≥90% line coverage (Catch2)
- CMakeLists.txt building the core as a static library

### Key Implementation Tasks

**1.1 Node and Tree types (weeks 1–3)**
- Implement `Node` struct (postfix encoding)
- `Tree = std::vector<Node>`; value semantics (copy, move)
- `tree_size()`, `subtree_size()`, `subtree_begin()` utilities
- Implement `parent_distance` field population in tree construction

**1.2 Operator registry (weeks 2–4)**
- `OperatorSet`: stores arrays of unary and binary `std::function<double(double...)>`
- Each operator has: name, arity, function pointer, derivative function pointer
- Standard operators: `+`, `-`, `*`, `/`, `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`, `abs`, `tanh`, `sigmoid`
- Protected operators (safe versions): `safe_log`, `safe_sqrt`, `safe_div` (NaN/Inf guards)

**1.3 Scalar evaluation (weeks 3–5)**
- Stack-machine evaluator for single data point
- Batch evaluator: loop over data matrix rows, output to result vector
- Verify SIMD auto-vectorization: compile with `-O3 -march=native`, inspect assembly

**1.4 Dual number AD (weeks 4–7)**
```cpp
template<typename Real>
struct Dual {
    Real val, grad;
    // Full operator overload set
};
// Explicit instantiation of all standard math functions
template<> Dual<double> sin(Dual<double> x) {
    return {std::sin(x.val), x.grad * std::cos(x.val)};
}
```
- Instantiate `evaluate_typed<Dual<double>>()` for gradient computation
- Verify against numerical finite differences on 100+ random expressions

**1.5 Random tree generation (weeks 6–8)**
- Configurable max depth and size
- Random operator/variable/constant selection
- Generates valid postfix trees

**1.6 Tree serialization (weeks 8–12)**
- Human-readable string form: `(sin (* x1 (+ x2 3.14)))`
- Round-trip parse and serialize for debugging and logging
- Expression equivalence testing for unit tests

### Phase 1 Benchmark Milestone
- Batch evaluation of 1000 random trees (depth 6) × 1000 data points: < 100ms
- All unit tests pass on Linux (GCC 12), macOS (Clang 15), Windows (MSVC 2022)

---

## Phase 2: Evolutionary Search Core
**Duration:** 5 months  
**Goal:** A working single-threaded symbolic regression engine that finds equations on small problems.

### Deliverables
- Complete evolution loop (mutation, selection, population management)
- Ceres-based constant optimization
- Basic rule-based simplifier (~40 rules)
- Hall of Fame / Pareto front
- Adaptive parsimony pressure
- Command-line demo: `rsymbolic2_demo --input data.csv --output results.json`

### Key Implementation Tasks

**2.1 Ceres Solver integration (weeks 1–5)**
```cpp
// CostFunction wrapping expression evaluation
struct ExpressionCost : ceres::SizedCostFunction<N_RESIDUALS, N_PARAMS> {
    bool Evaluate(double const* const* params, double* residuals, double** jacobians) const override;
};
// Use AutoDiffCostFunction wrapping our evaluate_typed<Dual>
```
- Test on 10 known functions; verify optimizer converges to correct constants
- Implement restart logic: N_restarts with 50% perturbation, retain best

**2.2 PopMember and Population (weeks 3–6)**
- `PopMember`: tree + loss + birth_time + complexity
- `Population`: `std::vector<PopMember>` + sorted age index
- `replace_oldest()`: efficiently update the oldest member

**2.3 Mutation operators (weeks 4–10)**
Implement all 12 operators in priority order:
1. `mutate_constant` (week 4)
2. `mutate_operator`, `mutate_feature` (week 4)
3. `randomize_tree` (week 5)
4. `delete_random_op` (week 6)
5. `append_random_op`, `prepend_random_op` (weeks 6–7)
6. `insert_random_op` (week 8)
7. `swap_operands`, `rotate_tree` (week 9)
8. `crossover_trees` (week 10)
9. `backsolve_rewrite` — defer to Phase 4

**2.4 Regularized evolution cycle (weeks 8–11)**
```cpp
void run_cycle(Population& pop, const Options& opts, RNG& rng) {
    int n_cycles = pop.size() / opts.tournament_n;
    for (int i = 0; i < n_cycles; ++i) {
        auto& parent = best_of_sample(pop, opts.tournament_n, rng);
        if (uniform(rng) < opts.crossover_prob) {
            auto& parent2 = best_of_sample(pop, opts.tournament_n, rng);
            auto [child1, child2] = crossover(parent.tree, parent2.tree, rng);
            optimize_constants(child1, pop_data, opts);
            optimize_constants(child2, pop_data, opts);
            pop.replace_oldest(PopMember(child1, ...));
            pop.replace_oldest(PopMember(child2, ...));
        } else {
            auto child_tree = mutate(parent.tree, opts, rng);
            optimize_constants(child_tree, pop_data, opts);
            pop.replace_oldest(PopMember(child_tree, ...));
        }
    }
}
```

**2.5 Hall of Fame (weeks 11–13)**
- Non-dominated insertion
- Top-k extraction
- JSON serialization for logging

**2.6 Rule-based simplifier (weeks 13–17)**
Implement ~40 rewrite rules as bottom-up tree passes:
- Constant folding: `eval_if_constant_children()`
- Identity elimination: `x * 1 → x`, `x + 0 → x`, `x^1 → x`, `x^0 → 1`
- Double negation: `--x → x`, `log(exp(x)) → x`
- Commutativity normalization: canonical ordering of `+` and `*`
- Applied after each constant optimization call

**2.7 Adaptive parsimony (weeks 16–18)**
- `RunningSearchStats`: frequency table over complexity levels
- Window-based update (mutex-protected for Phase 3)
- Integration into loss penalty: `adjusted_loss = raw_loss * parsimony_factor(complexity)`

**2.8 Main search loop (weeks 17–20)**
```cpp
HallOfFame equation_search(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Options& opts
);
```
- Single population, single-threaded
- Iteration loop with convergence check and time limit
- Progress logging to stderr

### Phase 2 Benchmark Milestone
- Recover `y = sin(x1) + x2^2` from noisy data (σ=0.01) in < 30 seconds
- Recover 10/20 equations from internal regression suite (60-second limit each)
- Constant optimizer: converges on 90%+ of test functions with correct gradient

---

## Phase 3: R Interface + Parallelism
**Duration:** 5 months  
**Goal:** An installable R package with multi-island parallel search. Match PySR's quality on the 20-equation regression suite.

### Deliverables
- R package `rsymbolic2` installable via `install.packages()` from r-universe
- `sr_fit()`, `predict.rsymbolic2()`, `print.rsymbolic2()`, `plot.rsymbolic2()` functions
- TBB-parallel island model (N islands, configurable)
- Inter-island migration
- Full SRBench protocol runner (R script)

### Key Implementation Tasks

**3.1 cpp11 bridge (weeks 1–4)**
- `XPtr<SearchState>` for mutable C++ object lifetime in R
- Data marshalling: `cpp11::doubles_matrix` → `Eigen::MatrixXd`
- Result marshalling: `HallOfFame` → R data frame
- Error handling: C++ exceptions caught and rethrown as R errors

**3.2 R API design (weeks 3–6)**
```r
# Primary interface
result <- sr_fit(
    X, y,
    operators      = c("+", "-", "*", "/", "sin", "cos", "exp", "log"),
    n_populations  = 15,          # number of islands
    population_size = 100,        # members per island
    max_complexity = 20,
    n_iterations   = 40,
    timeout_seconds = 300,
    loss           = "mse",       # "mse", "mae", "custom"
    verbosity      = 1
)
```

**3.3 TBB island parallelism (weeks 4–8)**
- `N_islands` `Population` objects, each evolved by one TBB task
- Per-island RNG (seeded from master + island_id)
- Synchronization barrier every `migration_interval` outer iterations
- `migrate()`: copy top-k members from each island to a random other island

**3.4 Thread-safety audit (weeks 7–10)**
- Review all shared mutable state: `RunningSearchStats`, `HallOfFame`, progress counters
- Protect with `std::mutex` or `std::atomic` as appropriate
- Use thread sanitizer (TSan) to detect races during testing

**3.5 R package structure (weeks 8–12)**
- `DESCRIPTION`, `NAMESPACE`, `configure.ac`
- `Makevars.in` for linking Ceres and TBB
- `testthat` unit tests covering: data input validation, output format, reproducibility (fixed seed)
- `pkgdown` documentation site

**3.6 SRBench runner (weeks 13–17)**
- R script implementing SRBench protocol: PMLB datasets, 10-fold CV, 30-minute limit
- Outputs: R² per dataset, complexity, time; stored as CSV for comparison
- Compare against PySR reference results (pre-computed)

**3.7 Visualization (weeks 16–20)**
- `plot.rsymbolic2()`: Pareto front (loss vs. complexity), ggplot2
- `print.rsymbolic2()`: equation table with loss and complexity

### Phase 3 Benchmark Milestone
- Parallel speedup ≥ 0.7 × n_threads (measured on 8-thread machine)
- Match PySR recovery on internal 20-equation set (within 10% of PySR at same time budget)
- R package passes `R CMD check --as-cran` (no ERRORs, no WARNINGs)

---

## Phase 4: Advanced Features
**Duration:** 6 months  
**Goal:** Close the quality gap with PySR; add features required for scientific use cases.

### Key Features

**4.1 E-graph simplification (months 15–17)**
- Implement minimal C++ e-graph (union-find + rule engine) OR integrate egglog via C FFI
- 80+ rewrite rules covering algebraic identities, trigonometric identities, exponential laws
- Benchmark: compare Pareto front quality (hypervolume) with and without e-graph on Feynman set

**4.2 Dimensional analysis (months 15–18)**
- Unit type system: track SI dimensions (mass, length, time, current, temperature)
- Dimension constraint propagation: reject expressions that are dimensionally inconsistent
- Significantly narrows search space for physics applications
- Implementation: templated `Quantity<Unit>` type propagated through expression evaluation

**4.3 Adaptive parsimony improvements (month 15)**
- Per-island parsimony stats (reduces contention)
- Temperature schedule: higher parsimony pressure early, lower late in search

**4.4 Custom loss functions (month 16)**
- R API: `loss = function(y_pred, y_true) { ... }` accepted as R function
- Bridge: R function called from C++ via `cpp11::function` wrapper
- Performance note: R function callbacks add overhead; document this trade-off
- Gradient approximation: finite differences when no analytic gradient is provided

**4.5 Backsolve rewrite operator (months 17–18)**
- Port the experimental `backsolve_rewrite_random_node` from SymbolicRegression.jl
- Inverts a subtree: solves for the value the subtree should produce to minimize loss
- Fits a new random expression to those target values
- Expected impact: improved performance on equations with complex substructure

**4.6 Parametric expressions (months 19–20)**
- Fixed-structure expressions with optimized constants: `y = a * sin(b * x + c)`
- User specifies template; optimizer finds a, b, c
- Use case: physics fitting where functional form is known

**4.7 Multi-output regression (months 20)**
- `MultitargetSR`: search for equations for multiple output variables simultaneously
- Shared population (expressions can be shared across outputs)

### Phase 4 Benchmark Milestone
- Feynman recovery rate ≥ 65% (matching PySR) at equivalent time budget
- Dimensional analysis: reduces search time by ≥ 20% on dimensioned Feynman problems
- SRBench median R² ≥ PySR (Phase 3 gap closed)

---

## Phase 5: Python Interface + Publication
**Duration:** 4 months  
**Goal:** Python package; reproducible benchmark results for publication or public release.

### Key Tasks

**5.1 pybind11 Python bindings (months 21–22)**
```python
import rsymbolic2 as sr

result = sr.fit(
    X=np.array(...),
    y=np.array(...),
    operators=["+", "-", "*", "/", "sin", "cos"],
    n_iterations=40,
    timeout=300
)
print(result.pareto_front)    # DataFrame
print(result.best_equation)   # string: "sin(x1) + x2^2"
```
- scikit-learn compatible API (`fit`, `predict`, `score`)
- NumPy array input/output
- No Julia runtime dependency

**5.2 Performance profiling + optimization (months 21–23)**
- Profile full search with perf/VTune on the Feynman benchmark
- Identify top-3 hotspots; optimize
- Expected hotspots: evaluation batch loop, Ceres solver call overhead, memory allocation in mutation

**5.3 Full SRBench++ evaluation (months 23–24)**
- 30 runs × 252 PMLB datasets × 30-minute limit
- Compare: rsymbolic2, PySR, Operon
- Statistical analysis: Wilcoxon tests, critical difference diagrams
- Write-up: technical report or arXiv preprint

**5.4 Documentation and packaging (month 24)**
- pkgdown site for R package
- MkDocs or Sphinx site for Python package
- CRAN submission (requires bundled Ceres + TBB source)
- PyPI submission

### Phase 5 Benchmark Milestone
- Python package passes standard scikit-learn estimator check
- Full SRBench++ results published or archived

---

## Risk Register

| Risk                              | Probability | Impact | Mitigation                                          |
|-----------------------------------|-------------|--------|-----------------------------------------------------|
| Ceres not installable on Windows  | MEDIUM      | HIGH   | Bundle source; test on GitHub Actions Windows runner |
| egglog C FFI unstable             | HIGH        | MEDIUM | Implement minimal C++ e-graph as fallback           |
| Postfix mutation bugs             | MEDIUM      | HIGH   | Fuzz testing; compare against pointer-tree impl     |
| TBB not available on all R targets| MEDIUM      | MEDIUM | Fallback: OpenMP; serial mode always available      |
| Performance gap vs PySR (Julia JIT)| MEDIUM     | MEDIUM | Profile early (Phase 2); SIMD intrinsics if needed  |
| Constant optimizer divergence     | LOW         | MEDIUM | Multiple restarts + NaN guards; fallback to Nelder-Mead |

---

## Milestone Summary

| Phase | End Month | Deliverable                                   | Quality Gate                              |
|-------|-----------|-----------------------------------------------|-------------------------------------------|
| 1     | M4        | C++ expression library + tests                | 90% unit test coverage; eval speed target |
| 2     | M9        | Single-threaded SR engine; CLI demo           | 10/20 equations recovered < 60s each     |
| 3     | M14       | R package on r-universe; parallel search      | R² matches PySR within 10%; parallel speedup ≥ 0.7× |
| 4     | M20       | E-graph simplification; dimensional analysis  | Feynman recovery ≥ 65%                    |
| 5     | M24       | Python package; full SRBench++ results        | scikit-learn API; public results available|
