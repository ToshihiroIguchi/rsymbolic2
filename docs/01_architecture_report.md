# Architecture Report: rsymbolic2
## A Native C++ Symbolic Regression Library

**Date:** 2026-06-03  
**Scope:** Architecture analysis of PySR/SymbolicRegression.jl and proposed C++ redesign

---

## 1. Reference System Analysis: PySR and SymbolicRegression.jl

### 1.1 Overall Architecture

PySR is a two-layer system. The user-facing Python layer (PySR) handles configuration, data marshalling, and result formatting. The computation layer is SymbolicRegression.jl, a Julia package invoked via `juliacall`. Julia is required at runtime; the JIT compilation overhead (10–30 seconds on first run) is a known user-experience issue.

```
Python Layer (PySR)
  ├── Options validation
  ├── Data marshalling (numpy → Julia arrays)
  ├── Result extraction and formatting
  └── juliacall bridge
        │
        ▼
Julia Layer (SymbolicRegression.jl)
  ├── DynamicExpressions.jl  ← expression trees + SIMD evaluation
  ├── Optim.jl               ← constant optimization
  ├── DifferentiationInterface.jl  ← AD abstraction
  ├── Enzyme.jl / Mooncake.jl      ← AD backends (weak deps)
  ├── SymbolicUtils.jl             ← symbolic simplification (weak dep)
  └── DynamicQuantities.jl         ← dimensional analysis
```

### 1.2 Expression Tree: DynamicExpressions.jl Node Type

The fundamental data structure is a heap-allocated binary tree. Each `Node{T}` contains:

| Field       | Type    | Description                                       |
|-------------|---------|---------------------------------------------------|
| `degree`    | UInt8   | 0 = leaf, 1 = unary, 2 = binary                   |
| `constant`  | Bool    | true if leaf is a literal constant                |
| `val`       | T       | constant value (when degree==0, constant==true)   |
| `feature`   | UInt16  | variable index (when degree==0, constant==false)  |
| `op`        | UInt16  | operator index (when degree > 0)                  |
| `l`         | Node{T} | left child (when degree ≥ 1)                      |
| `r`         | Node{T} | right child (when degree == 2)                    |

Operators are stored in an `Options` struct as two arrays of function pointers: `binary_operators` and `unary_operators`. The `op` integer indexes these arrays at runtime. Julia's JIT can specialize the evaluation loop for a given operator set, enabling SIMD kernel fusion. Benchmark: 607 ns per evaluation vs. 526 ns for hand-written SIMD (approximately 15% overhead).

**Critical observation:** The pointer-chasing structure of a heap-allocated linked tree causes cache misses during evaluation. DynamicExpressions.jl compensates via Julia's JIT specialization. A C++ implementation without a JIT compiler must use a different representation.

### 1.3 Evolutionary Algorithm

SymbolicRegression.jl implements **regularized evolution** (also called age-fitness Pareto optimization):

1. Each `PopMember` carries a birth timestamp (iteration counter).
2. `best_of_sample()` selects the fittest member from a random subsample of size `tournament_selection_n` (default 10).
3. The *oldest* member is replaced by the offspring (not the worst-fitness member).
4. This age-based replacement prevents premature convergence while maintaining quality pressure.

**Cycle structure:**
```
for each cycle (pop_size / tournament_n iterations):
    parent = best_of_sample(population, tournament_n)
    if rand() < crossover_probability:
        parent2 = best_of_sample(population, tournament_n)
        child1, child2 = crossover(parent, parent2)
        replace oldest two members
    else:
        child = mutate(parent)
        optimize_constants(child)
        replace oldest member
```

**Mutation operators (12 total):**

| Operator              | Description                                              |
|-----------------------|----------------------------------------------------------|
| `mutate_constant`     | Perturb a constant by temperature-scaled random factor   |
| `mutate_operator`     | Replace an operator with another of same arity           |
| `mutate_feature`      | Reassign a leaf variable to a different feature          |
| `append_random_op`    | Attach new operator + subtree to a leaf                  |
| `insert_random_op`    | Insert new operator into interior, carrying child subtree|
| `prepend_random_op`   | Add new root operator, existing tree becomes one child   |
| `delete_random_op`    | Remove interior node, splice one child back in place     |
| `randomize_tree`      | Replace entire tree with new random tree                 |
| `swap_operands`       | Swap two children of a binary operator                   |
| `rotate_tree`         | Local parent-child rotation (restructuring)              |
| `crossover_trees`     | Exchange subtrees between two trees (interindividual)    |
| `backsolve_rewrite`   | Experimental: invert a node, fit replacement expression  |

### 1.4 Constant Optimization

File: `ConstantOptimization.jl`

- Library: **Optim.jl**
- Single-constant case: `Optim.Newton()` with backtracking line search
- Multi-constant case: configurable via `options.optimizer_algorithm` (default BFGS)
- Restarts: `options.optimizer_nrestarts` (default 2), with 50% perturbation of initial values
- Gradients: **DifferentiationInterface.jl** calling `value_and_gradient!()`
- AD backends: Enzyme.jl (source transformation, fastest), Mooncake.jl, Zygote.jl
- Acceptance criterion: only accept if `new_loss < baseline_loss`

**Known limitation:** Optimization runs with a fixed small iteration budget per candidate expression. Long optimization runs would dominate evaluation time and are not feasible inside an inner evolutionary loop.

### 1.5 Parallelism Architecture

Three modes, selected at search initialization:

| Mode              | Mechanism                    | Worker ID type  |
|-------------------|------------------------------|-----------------|
| `:serial`         | Direct function calls        | N/A             |
| `:multithreading` | `Threads.@spawn`             | `Task`          |
| `:multiprocessing`| `Distributed.@spawnat`       | `Future`        |

The `@sr_spawner` macro abstracts over these three modes. Workers are tracked in `WorkerAssignments`, a dictionary mapping `(output_idx, population_idx) → worker_id`. Load balancing: `next_worker()` selects the least-busy worker by job count.

Multiple populations exist per output dimension. Populations evolve independently; periodic **migration** (`Migration.jl`) exchanges top individuals between populations at a configurable interval.

### 1.6 Adaptive Parsimony Pressure

`AdaptiveParsimony.jl` maintains a sliding window of 100,000 recently observed expressions. The frequency distribution over complexity levels is tracked and normalized. When the window fills, frequencies are proportionally reduced. This normalized distribution is used to penalize complexity levels that are already well-explored, pushing the search toward underexplored complexity regions.

**Effect:** Prevents stagnation at a fixed complexity; biases search toward novel expression sizes over time.

### 1.7 Symbolic Simplification

Handled by `SymbolicUtils.jl` (weak dependency; only loaded if installed). The integration is post-hoc: after mutation and evaluation, an expression may be simplified before being added to the Hall of Fame. Simplification uses term-rewriting rules (associativity, commutativity, constant folding, identity elimination).

**Known limitation:** SymbolicUtils.jl applies rules sequentially in a fixed order. This greedy approach may miss simplifications reachable only through non-obvious rule sequences.

---

## 2. Proposed C++ Architecture for rsymbolic2

### 2.1 Design Philosophy

The proposed design eliminates the Julia dependency entirely, targeting a self-contained C++17 library with:
- Predictable performance without JIT warmup
- Direct R integration via cpp11 (no inter-process communication)
- A layered API: core C++ library → R package → future Python package
- Operon-style linear tree encoding for cache efficiency

### 2.2 Module Map

```
rsymbolic2/
├── core/                          # C++17 core library (header + source)
│   ├── include/rsymbolic/
│   │   ├── expression/
│   │   │   ├── node.hpp           # Node type (linear encoding)
│   │   │   ├── tree.hpp           # Tree = std::vector<Node>
│   │   │   ├── operator_set.hpp   # Operator registry
│   │   │   └── eval.hpp           # Tree evaluation (scalar + dual)
│   │   ├── evolution/
│   │   │   ├── population.hpp     # Population + PopMember
│   │   │   ├── selection.hpp      # Tournament selection
│   │   │   ├── mutation.hpp       # All 12 mutation operators
│   │   │   ├── crossover.hpp      # Subtree crossover
│   │   │   └── migration.hpp      # Inter-population migration
│   │   ├── optimization/
│   │   │   ├── constant_opt.hpp   # Ceres/NLopt wrapper
│   │   │   └── loss.hpp           # Loss function types
│   │   ├── simplification/
│   │   │   ├── rewrite.hpp        # Rule-based simplifier (Phase 1)
│   │   │   └── egraph.hpp         # E-graph interface (Phase 4)
│   │   ├── search/
│   │   │   ├── options.hpp        # Search hyperparameters
│   │   │   ├── hall_of_fame.hpp   # Pareto front
│   │   │   ├── search_state.hpp   # Global state
│   │   │   └── main_search.hpp    # Top-level search function
│   │   └── utils/
│   │       ├── complexity.hpp     # Complexity calculation
│   │       ├── parsimony.hpp      # Adaptive parsimony
│   │       └── rng.hpp            # Thread-safe RNG
│   ├── src/                       # .cpp implementations
│   └── tests/                     # Catch2 unit tests
├── r/                             # R package (rsymbolic2)
│   ├── R/                         # R wrapper functions
│   ├── src/                       # cpp11 bridge code
│   ├── tests/testthat/
│   └── DESCRIPTION
├── python/                        # Future Python bindings
│   └── (placeholder)
├── benchmarks/                    # Benchmark suite
│   ├── feynman/
│   ├── pmlb/
│   └── timing/
└── CMakeLists.txt
```

### 2.3 Expression Tree: Linear Encoding

Inspired by Operon's approach. Trees are stored as `std::vector<Node>` in **postfix order** (left subtree, right subtree, operator). This ensures all children appear before their parent in memory, enabling a simple stack-based evaluator with no pointer indirection.

```cpp
// core/include/rsymbolic/expression/node.hpp

enum class NodeType : uint8_t {
    CONSTANT  = 0,
    VARIABLE  = 1,
    UNARY_OP  = 2,
    BINARY_OP = 3,
};

struct Node {
    NodeType type;
    uint16_t op_or_feature;   // operator index OR variable index
    uint16_t parent_distance; // distance to parent in postfix array (for tree ops)
    float    value;           // constant value (if type == CONSTANT)
};

// Tree is a value type; copying is O(n) in tree size
using Tree = std::vector<Node>;
```

**Evaluation (stack-based, no recursion):**
```cpp
// Pseudocode
double evaluate(const Tree& tree, const double* row) {
    double stack[MAX_DEPTH];
    int top = -1;
    for (const auto& node : tree) {
        if (node.type == CONSTANT) { stack[++top] = node.value; }
        else if (node.type == VARIABLE) { stack[++top] = row[node.op_or_feature]; }
        else if (node.type == UNARY_OP) { stack[top] = unary_ops[node.op_or_feature](stack[top]); }
        else { double r = stack[top--]; stack[top] = binary_ops[node.op_or_feature](stack[top], r); }
    }
    return stack[0];
}
```

**Dual number evaluation for AD (forward mode):**
```cpp
template<typename T>  // T = double or Dual<double>
T evaluate_typed(const Tree& tree, const T* row, const T* constants);
```

### 2.4 Constant Optimization

This decision is deliberately revised from an earlier draft that named Ceres
Solver as the primary optimizer. Per the project priorities (portability and
simplicity outrank performance), the default is now a self-contained
implementation with no heavy external dependency.

**Primary:** A self-implemented Levenberg-Marquardt / Gauss-Newton solver built on
Eigen (already a required, header-only dependency).
- Rationale: SR expressions typically contain 1–5 constants. This is a very small,
  dense nonlinear least-squares problem. A compact LM implementation (a few hundred
  lines) is sufficient, adds zero new dependencies, and avoids the per-call setup
  overhead of a general-purpose solver invoked millions of times in the inner loop.
- Gradients: forward-mode dual numbers via `evaluate_typed<Dual>()` (see 2.5).
- This aligns with "pursue performance only with measured evidence": a heavy solver
  is not justified before measurement.

**Conditional fallback — Ceres Solver:** Considered only if the self-implemented LM
is *measured* to be insufficient (poor convergence or robustness on degenerate
Jacobians). Ceres is a strong solver (NIST nonlinear-regression benchmark winner)
but pulls in Eigen + glog + gflags and raises Windows maintenance cost, so it is a
fallback, not the default. The decision must be evidence-driven.

**Fallback for non-sum-of-squares losses:** NLopt with L-BFGS-B, or a
derivative-free method (Nelder-Mead), for user-defined losses (MAE, Huber, etc.).
Whether NLopt is worth its dependency cost is itself deferred until custom losses
are actually implemented (Phase 4).

**Restart strategy:** Identical to SymbolicRegression.jl — N restarts with 50% perturbation, retain best.

### 2.5 Automatic Differentiation

For constant optimization, **forward-mode AD via dual numbers** is used:

```cpp
template<typename Real>
struct Dual {
    Real value;
    Real grad;
    // operator overloads for +, -, *, /, sin, cos, exp, log, ...
};
```

Rationale: SR expressions contain few constants (typically 1–5). Forward mode computes one directional derivative per evaluation pass; for k constants, k passes are needed. For k ≤ 10, this is competitive with reverse mode and has no tape allocation overhead.

For k > 10 (rare), a reverse-mode tape (Adept-2) would be more efficient. This can be added as a backend option in Phase 4.

### 2.6 Parallelism Architecture

This decision is also revised from an earlier draft that assumed Intel TBB. The
island model is coarse-grained and its per-island load is approximately uniform, so
TBB's main advantage (work-stealing for uneven loads) is largely unused. A simpler,
dependency-free mechanism is preferred.

**Primary:** OpenMP `parallel for` over island populations. OpenMP is built into
GCC, Clang, and MSVC, so it adds no external dependency and no Windows maintenance
cost. Each island runs its evolutionary cycle independently; migration is
synchronized at fixed intervals.

```cpp
// Parallel island evolution (OpenMP)
#pragma omp parallel for schedule(dynamic)
for (int i = 0; i < num_islands; ++i) {
    run_evolution_cycle(islands[i], options, rng_per_thread[i]);
}
// Synchronization point: migrate top members between islands
migrate(islands, options.migration_rate);
```

`schedule(dynamic)` covers the modest load imbalance from islands converging to
different tree sizes. C++17 `std::thread` with a small thread pool is an equally
acceptable dependency-free alternative.

**Conditional fallback — Intel TBB:** Considered only if OpenMP is *measured* to
scale poorly (load imbalance below the 0.7 efficiency target). Not adopted
speculatively.

**Within-population:** Each island's cycle is single-threaded (mutation + evaluation are fast for small trees). SIMD vectorization within `evaluate_batch()` via auto-vectorization hints or explicit intrinsics.

**Thread-safety:** Each thread owns its `pcg64` RNG instance (seeded from master seed + thread_id). No shared mutable state during evolution cycles.

### 2.7 Symbolic Simplification

**Phase 1 (rule-based):** A hand-written set of ~40 algebraic rewrite rules applied bottom-up on the linear tree. Rules cover: constant folding, identity elimination (x*1→x, x+0→x), double-negation (--x→x), basic factoring. Applied after constant optimization.

**Phase 2 (e-graph):** Interface to `egglog` (the successor to `egg`, written in Rust with C FFI). E-graphs guarantee finding the simplest equivalent expression within a ruleset, unlike greedy sequential rewriting. Requires a C FFI bridge to the Rust library.

See `simplification/egraph.hpp` for the planned interface. This is a Phase 4 deliverable.

### 2.8 Complexity and Pareto Front

**Complexity:** Node count by default. Configurable: operators/variables/constants can have individual weights. Implementation: single linear scan of the tree vector (O(n)).

**Hall of Fame (Pareto front):** A sorted list of non-dominated `(loss, complexity)` pairs. On insertion, dominated members are removed. At most one member per complexity level.

```cpp
struct HallOfFame {
    std::map<int, PopMember> pareto_front;  // complexity → best member at that complexity
    void update(const PopMember& candidate);
    std::vector<PopMember> get_pareto_frontier() const;
};
```

### 2.9 R Interface

Using **cpp11** (not Rcpp) for the R binding layer:
- Modern C++11/17 semantics, lower overhead for scalar operations
- Better ownership model (no unnecessary copies of large matrices)
- External pointer wrapping for `SearchState` object lifetime management

```r
# R usage (proposed API)
result <- sr_fit(
  X = matrix(...),
  y = numeric(...),
  operators = c("+", "-", "*", "/", "sin", "cos", "exp", "log"),
  max_complexity = 20,
  n_iterations = 40,
  n_populations = 15,
  timeout_seconds = 60
)

print(result)           # Pareto front table
plot(result)            # complexity vs loss
predict(result, newX)   # prediction with best equation
```

---

## 3. Key Differences from PySR / SymbolicRegression.jl

| Aspect                  | SymbolicRegression.jl        | rsymbolic2 (proposed)              |
|-------------------------|------------------------------|------------------------------------|
| Language                | Julia (JIT)                  | C++17 (AOT compiled)               |
| Tree representation     | Linked heap nodes (pointer)  | Contiguous vector (postfix order)  |
| Evaluation performance  | ~25% slower than hand SIMD   | Target: ≤20% slower than hand SIMD |
| Constant optimizer      | Optim.jl (BFGS/Newton)       | Self-implemented LM on Eigen; Ceres as measured fallback |
| AD for constants        | Enzyme/Mooncake/Zygote       | Forward-mode dual numbers (k≤10)   |
| Simplification          | SymbolicUtils.jl (greedy)    | Phase1: rules; Phase4: e-graphs    |
| Parallelism             | Julia Distributed / Threads  | OpenMP over islands; TBB as measured fallback |
| Startup time            | 10–30s (JIT)                 | <1s (no JIT)                       |
| R interface             | Via Python (PySR → juliacall)| Direct cpp11 binding               |
| Dimensional analysis    | DynamicQuantities.jl         | Phase 4 (unit type system)         |
| Parametric expressions  | Yes (ParametricExpression)   | Phase 4                            |
| Template expressions    | Yes (TemplateExpression)     | Phase 4                            |
