# Migration Difficulty Assessment: SymbolicRegression.jl → C++

**Date:** 2026-06-03

---

## Assessment Framework

Each component is rated on two axes:
- **Translation difficulty:** How hard is it to re-implement the logic in C++?
- **Strategy:** Direct migration | Requires redesign | Replace with newer approach

Ratings: LOW / MEDIUM / HIGH / VERY HIGH

---

## 1. Direct Migration (LOW–MEDIUM difficulty)

These components have clear algorithmic logic that maps directly to C++ without architectural changes.

### 1.1 Complexity Calculation
**Source:** `Complexity.jl`  
**Difficulty:** LOW  
**Description:** Single pass over tree nodes, summing weights. No Julia-specific dependencies.

```julia
# Julia
function compute_complexity(tree::Node, options)
    sum = 0
    for node in traverse(tree)
        sum += complexity_weight(node, options)
    end
    return sum
end
```

```cpp
// C++ equivalent: single linear scan
int compute_complexity(const Tree& tree, const Options& opts) {
    int total = 0;
    for (const auto& node : tree)
        total += opts.complexity_weight(node);
    return total;
}
```

**Migration notes:** None. Direct translation.

---

### 1.2 Tournament Selection
**Source:** `RegularizedEvolution.jl` (`best_of_sample()`)  
**Difficulty:** LOW  
**Description:** Select k members at random; return the one with lowest loss.

**Migration notes:** Replace Julia's `randperm` with Fisher-Yates shuffle on local index array. No architectural changes needed.

---

### 1.3 Hall of Fame / Pareto Front
**Source:** `HallOfFame.jl`  
**Difficulty:** LOW  
**Description:** Insert candidate; remove any member dominated by it; at most one member per complexity level.

**Migration notes:** Julia's Dict → C++ `std::map<int, PopMember>`. Trivial translation.

---

### 1.4 Loss Function Evaluation
**Source:** `LossFunctions.jl`  
**Difficulty:** LOW  
**Description:** MSE, weighted MSE, MAE, custom loss. Evaluated over batch of data points.

**Migration notes:** Batch evaluation loops over `(X, y)` pairs. The C++ template-based `evaluate_typed<T>()` handles both scalar and dual number evaluation uniformly.

---

### 1.5 Mutation Operators (11 of 12)
**Source:** `MutationFunctions.jl`  
**Difficulty:** MEDIUM  
**Description:** Tree structure modifications. Most are straightforward tree surgery.

**Individual operator difficulty:**

| Operator           | Difficulty | Notes                                             |
|--------------------|------------|---------------------------------------------------|
| mutate_constant    | LOW        | Random perturbation of float value                |
| mutate_operator    | LOW        | Replace op index with another of same arity       |
| mutate_feature     | LOW        | Replace variable index                            |
| swap_operands      | LOW        | Swap two indices in postfix array                 |
| delete_random_op   | MEDIUM     | Splice parent_distance; need postfix index math   |
| randomize_tree     | LOW        | Generate new random tree                          |
| append_random_op   | MEDIUM     | Extend vector; update parent distances            |
| insert_random_op   | MEDIUM     | Insert into middle of postfix vector              |
| prepend_random_op  | MEDIUM     | Prepend new root; wrap existing                   |
| rotate_tree        | MEDIUM     | Local restructure; postfix array shifts           |
| crossover_trees    | MEDIUM     | Identify subtree boundaries by parent_distance    |
| backsolve_rewrite  | HIGH       | Experimental; inverse function fitting            |

**Migration notes:** The pointer-based Julia tree makes subtree operations easy (swap pointers). The postfix linear encoding requires careful index arithmetic to identify subtree boundaries. Each node stores `parent_distance` (offset to its parent in the array); subtree size can be computed bottom-up. This is solved by Operon's implementation.

**Recommendation:** Study Operon's `Subtree()` function for postfix subtree boundary identification before implementing mutation operators.

---

### 1.6 Population Initialization
**Source:** `Population.jl`  
**Difficulty:** LOW  
**Description:** Generate N random trees of varying sizes; initialize PopMember with loss = Inf.

**Migration notes:** Random tree generation in postfix form requires a recursive builder that flattens to vector. Direct translation.

---

### 1.7 Migration Between Populations
**Source:** `Migration.jl`  
**Difficulty:** LOW  
**Description:** Copy top-k members from one island to replace worst-k of another.

**Migration notes:** Direct translation. In C++, use `std::partial_sort` to find top/bottom-k.

---

## 2. Requires Redesign (HIGH difficulty)

These components have the right algorithmic intent but require significant changes due to Julia-specific features.

### 2.1 Expression Tree Data Structure
**Source:** `DynamicExpressions.jl` Node type  
**Difficulty:** HIGH  
**Reason:** Julia uses GC-managed heap nodes with pointer-based trees. Julia's JIT enables runtime operator specialization for SIMD.

**C++ redesign:**
- Use Operon-style linear postfix encoding (`std::vector<Node>`)
- Tree operations require postfix-aware subtree identification
- SIMD: rely on compiler auto-vectorization of batch evaluation loops; test with `-O3 -march=native`
- No runtime JIT: all operators compiled in; new operators require recompilation

**Effort:** 3–6 weeks for core data structure + all mutation operators correctly handling postfix form.

---

### 2.2 Constant Optimization
**Source:** `ConstantOptimization.jl`  
**Difficulty:** HIGH  
**Reason:** Uses Julia-specific `Optim.jl` and `DifferentiationInterface.jl` with pluggable AD backends (Enzyme, Mooncake, Zygote). Enzyme performs source-level transformation on Julia IR.

**C++ redesign:**
- Replace Optim.jl with Ceres Solver (LM for least-squares, L-BFGS-B via NLopt for general loss)
- Replace Enzyme/Mooncake with forward-mode dual numbers for k ≤ 10 constants
- Ceres `AutoDiffCostFunction` using dual numbers for gradient computation
- Restart logic: direct translation

**Critical decision:** Ceres LM requires the residuals to be a vector (one per data point), not a scalar loss. This means MSE and related losses integrate naturally, but custom scalar losses require wrapping as a single residual block. This is a design constraint.

**Effort:** 4–6 weeks including Ceres integration, dual number implementation, and correctness verification.

---

### 2.3 Parallel Search Architecture
**Source:** `SearchUtils.jl`, `@sr_spawner` macro  
**Difficulty:** HIGH  
**Reason:** Julia's `Distributed` and `Threads` are first-class language features with seamless data sharing via SharedArrays and remote references.

**C++ redesign:**
- TBB `parallel_for` over islands (coarse-grained parallelism)
- Thread-local state: each thread has its own RNG (`pcg64`) and a local copy of operator function pointers
- Population objects are value types (copied into workers, results copied back); avoids shared mutable state
- No `Distributed` equivalent needed: all threads in-process

**Key difference from Julia:** Julia's multiprocessing mode spawns separate OS processes communicating via Julia RPC. In C++, all parallelism is intra-process (threads via TBB). This is simpler but limits scaling to a single machine. Multi-machine distribution (if needed) would require MPI or a custom solution and is deferred.

**Effort:** 3–5 weeks including thread-safety analysis and testing.

---

### 2.4 Adaptive Parsimony Pressure
**Source:** `AdaptiveParsimony.jl`  
**Difficulty:** MEDIUM  
**Reason:** The algorithm is clear; the complexity is in thread-safe update of shared running statistics.

**C++ redesign:**
- `RunningSearchStatistics` protected by a mutex or atomic operations
- Frequency array update on each new expression evaluated
- Window-based proportional reduction is trivially translatable

**Effort:** 1–2 weeks.

---

## 3. Replace with Newer Approach

These components should not be migrated as-is; newer methods offer clear advantages.

### 3.1 Symbolic Simplification
**Current (SymbolicRegression.jl):** SymbolicUtils.jl — sequential rule application, greedy, may miss simplifications requiring non-obvious sequences.

**Replacement: E-graph (equality saturation)**

**Why:** E-graphs maintain all equivalent forms simultaneously. The extraction step selects the minimum-cost form. This guarantees finding the simplest equivalent expression within the ruleset, unlike greedy sequential rewriting. Applied to symbolic regression expressions (Folivetti et al., 2023), e-graph simplification reduces expression overparametrization and improves interpretability.

**Concrete plan:**
- **Phase 1:** Custom rule-based simplifier (~40 rules, C++ implementation). Covers constant folding, identity elimination, double negation. Fast, no external dependencies.
- **Phase 4:** egglog integration via C FFI. Provides completeness guarantee. Build complexity is the main barrier.

**Performance note:** The `egg` library reported up to 3000× speedups over sequential rule application on complex expressions. For SR expressions (typical depth 4–8), the speedup is more modest but still meaningful.

---

### 3.2 Expression Evaluation Architecture
**Current (SymbolicRegression.jl):** Julia JIT fuses operators into SIMD kernels at runtime based on the specific expression. Achieved via Julia's metaprogramming.

**Replacement: Compile-time operator dispatch + auto-vectorization**

The JIT approach is fundamentally unavailable in pre-compiled C++. Alternatives:
1. **Template specialization:** Generate evaluation code for fixed operator sets at compile time. Impractical for user-defined operators.
2. **Auto-vectorization:** Write the batch evaluation loop in a SIMD-friendly way; rely on `-O3 -march=native`. Compiler converts to SIMD instructions for simple operators.
3. **SIMD intrinsics (AVX2/AVX-512):** Manually write vectorized kernels for the 10–20 most common operators. Only worthwhile if profiling shows evaluation as the bottleneck.
4. **Operon's approach:** Contiguous postfix array + stack machine; modern compilers auto-vectorize this well.

**Recommendation:** Start with approach 2 (auto-vectorization). Profile against PySR before investing in approach 3.

---

### 3.3 Python Interface Architecture
**Current (PySR):** juliacall bridge — Python calls Julia at runtime. Requires Julia installation, 10–30s JIT warmup.

**Replacement:** pybind11 direct C++ binding

Eliminates runtime JIT dependency entirely. The Python package is a thin wrapper around the C++ core, similar to how scikit-learn's tree methods work. First-call performance: < 1 second.

---

## 4. Summary Table

| Component                    | Strategy          | Difficulty  | Estimated Effort |
|------------------------------|-------------------|-------------|------------------|
| Complexity calculation       | Direct migration  | LOW         | 2 days           |
| Tournament selection         | Direct migration  | LOW         | 1 day            |
| Hall of Fame / Pareto front  | Direct migration  | LOW         | 3 days           |
| Loss function evaluation     | Direct migration  | LOW         | 3 days           |
| Mutation operators (simple)  | Direct migration  | LOW-MEDIUM  | 2 weeks          |
| Population initialization    | Direct migration  | LOW         | 3 days           |
| Inter-island migration       | Direct migration  | LOW         | 2 days           |
| Adaptive parsimony           | Redesign          | MEDIUM      | 1–2 weeks        |
| Expression tree data struct  | Redesign (linear) | HIGH        | 3–6 weeks        |
| Constant optimization        | Redesign (Ceres)  | HIGH        | 4–6 weeks        |
| Parallel search arch         | Redesign (TBB)    | HIGH        | 3–5 weeks        |
| Symbolic simplification      | Replace (e-graph) | HIGH        | 2w (rules) + 6w (egglog) |
| Expression evaluation SIMD   | Replace (auto-vec)| MEDIUM      | 2–3 weeks        |
| Python interface             | Replace (pybind11)| MEDIUM      | 3–4 weeks        |
| Dimensional analysis         | New (Phase 4)     | HIGH        | 6–8 weeks        |
| Backsolve rewrite operator   | Defer             | VERY HIGH   | Deferred to Phase 4 |

**Total estimated effort (core library + R interface, phases 1–3):** ~32–42 developer-weeks
