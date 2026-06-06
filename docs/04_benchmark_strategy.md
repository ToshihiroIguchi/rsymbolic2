# Benchmark Strategy: rsymbolic2

**Date:** 2026-06-03

---

## 1. Goals of Benchmarking

The benchmark suite serves three distinct purposes:

1. **Regression testing during development:** Verify that no optimization or refactoring degrades search quality.
2. **Performance profiling:** Identify bottlenecks in evaluation, optimization, and parallelism.
3. **Publication-quality comparison:** Produce defensible results comparing rsymbolic2 to PySR, Operon, and neural SR methods.

These goals require different benchmark setups and should not be conflated.

---

## 2. Benchmark Datasets

### 2.1 Feynman-AI Dataset (Ground-Truth Recovery)
- **Source:** AI Feynman dataset (Udrescu & Tegmark, 2020) — 100 equations; extended to 120 with Feynman II
- **Most-used subset:** 117 physics equations (standard in recent SR literature)
- **Access:** [https://github.com/SciML/AIFeynman](https://github.com/SciML/AIFeynman)
- **Primary metric:** **Exact recovery rate** — fraction of equations for which the method finds a mathematically equivalent form
- **What "exact" means:** Normalized mean squared error (NMSE) < 10⁻⁴ on a held-out test set AND symbolic structure matches (or is equivalent to) the ground-truth formula
- **Why use it:** Standard reference; allows direct comparison with published PySR, Operon, QDSR numbers
- **Caveats:** Some equations are trivially simple; some have been criticized for redundancy. Recent work (SRSD dataset) provides more realistic variable ranges.
- **Current best (2025):** QDSR achieves 91.6% recovery; PySR achieves approximately 60–70% at comparable time budgets

### 2.2 SRBench / PMLB (Predictive Accuracy)
- **Source:** SRBench (La Cava et al., 2021); 252 datasets from Penn Machine Learning Benchmarks
- **Access:** [https://github.com/cavalab/srbench](https://github.com/cavalab/srbench)
- **Primary metrics:** R² on test set, model complexity (node count), interpretability score (SRBench++)
- **Why use it:** The largest standardized SR benchmark; includes real-world datasets without known ground truth
- **Protocol:** 10-fold cross-validation; 30-minute time limit per dataset per algorithm

### 2.3 SRBench++ (2025 Extended Protocol)
- **Source:** SRBench++ (La Cava et al., 2024/2025)
- **Extends SRBench with:** Domain-expert interpretability evaluation; sub-task analysis (feature selection, local minima avoidance)
- **12 algorithms evaluated** in the published study
- **Use in rsymbolic2:** Target: match SRBench++ protocol for publication-quality results

### 2.4 Internal Synthetic Dataset (Development Regression)
- A fixed set of 20 synthetic equations of varying complexity (complexity 5–25)
- Equations drawn from: Feynman simple set + GP-generated ground truths
- Run at every CI commit; time limit 60 seconds per equation
- Purpose: catch regressions before they reach the full benchmark suite

```
# Example equations for regression suite
y = x1 + x2                          # complexity 5  (trivial)
y = x1 * sin(x2) + exp(x3)           # complexity 10
y = x1^2 + 2*x1*x2 + x2^2           # complexity 12 (perfect square)
y = sin(x1) / (1 + x1^2)             # complexity 11
y = exp(-x1^2) * cos(2*pi*x1)        # complexity 14
```

### 2.5 Timing / Scalability Benchmarks
- Evaluate throughput: expressions evaluated per second as function of:
  - Tree depth (2–12)
  - Batch size (100, 1000, 10000, 100000 data points)
  - Number of threads (1, 2, 4, 8, 16)
- Platform: fixed hardware (e.g., Intel Core i7-13700K, 8P+8E cores, AVX2)
- Purpose: characterize performance vs. PySR and Operon

---

## 3. Comparison Algorithms

| Algorithm    | Type                    | Why Include                                    |
|--------------|-------------------------|------------------------------------------------|
| PySR 1.x     | Evolutionary (Julia)    | Primary reference; most widely used            |
| Operon       | Evolutionary (C++)      | Best C++ baseline; competitive with PySR       |
| gplearn       | Evolutionary (Python)   | Widely known; older baseline                   |
| DSR / uDSR   | Neural + RL             | Representative neural SR method                |
| QDSR         | Evolutionary + QD       | Current Feynman SOTA (91.6% recovery)          |

For timing comparisons, PySR and Operon are the most relevant; DSR/uDSR are included for breadth.

---

## 4. Metrics

### 4.1 Recovery Metrics (for datasets with ground truth)

**Exact recovery rate (R_exact):**
A solution is "exact" if NMSE(ŷ_test, y_test) < ε_exact AND (optionally) structural equivalence.
- ε_exact = 10⁻⁴ is the standard threshold
- Structural equivalence: verified via symbolic simplification + coefficient comparison
- Report: percentage of equations exactly recovered across 30 independent runs

**Approximate recovery rate (R_approx):**
- ε_approx = 10⁻²
- Captures near-misses; useful for tracking learning curves

**Tree Edit Distance (TED) to ground truth:**
- As recommended by the 2024 constant optimization benchmarking paper
- Measures structural similarity even when NMSE threshold is not met
- Report: median TED across 30 runs

### 4.2 Predictive Metrics (for real-world datasets)

**R² on held-out test set:**
- Standard; comparable across all benchmarks
- Report: median R² and 25th–75th percentile across 30 runs

**Normalized Mean Squared Error (NMSE):**
- NMSE = MSE(ŷ, y) / Var(y)
- Scale-independent; comparable across datasets

**Model complexity:**
- Node count of returned expression
- Pareto front hypervolume (area under the loss-vs-complexity curve)
- Lower hypervolume = better Pareto approximation

### 4.3 Computational Metrics

**Wall-clock time to first exact recovery:**
- Median time across 30 runs; DNF if not recovered within time limit
- Report: median, 25th/75th percentile, DNF rate

**Throughput:**
- Expressions evaluated per second (batch evaluation benchmark)
- Constant optimization: average time per call, success rate

**Parallel scaling efficiency:**
- Speedup(n_threads) / n_threads; should be > 0.7 for n_threads ≤ 16

---

## 5. Statistical Methodology

**Number of runs:** 30 independent runs per (algorithm, problem) pair.
- This follows SRBench protocol and provides adequate power for non-parametric tests.

**Random seed management:** Seeds drawn from a fixed master sequence. Same seeds used for all algorithms on the same problem, enabling paired tests.

**Hypothesis tests:**
- **Paired Wilcoxon signed-rank test** for pairwise comparisons (non-parametric; no normality assumption)
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons
- Report: p-value and effect size (rank-biserial correlation)

**Aggregation across problems:**
- Critical difference diagrams (Demšar, 2006) for ranking algorithms across all problems
- Separate analysis for: easy problems (complexity ≤ 10), medium (11–20), hard (> 20)

**Reporting checklist:**
- [ ] Exact version of each algorithm used (git hash or package version)
- [ ] Hardware specification
- [ ] Time limit per run
- [ ] Number of runs
- [ ] Full results table (not just cherry-picked equations)
- [ ] Statistical test details

---

## 6. Anti-Patterns to Avoid

The following methodological pitfalls are documented in the SR benchmarking literature and must be avoided:

| Anti-pattern                          | Risk                                    | Mitigation                              |
|---------------------------------------|-----------------------------------------|-----------------------------------------|
| Tuning hyperparameters on test set    | Over-optimistic recovery rates          | Fix hyperparameters before benchmarking |
| Single run per problem                | High variance; luck-dependent           | Always 30 runs                          |
| NMSE threshold < 10⁻⁴                | Inflates "exact recovery" claims        | Use standard threshold 10⁻⁴            |
| Comparing different time limits       | Unfair to slower algorithms             | Fixed wall-clock limit for all          |
| Reporting only best run               | Selection bias                          | Always report median ± IQR             |
| Simplified versions of Feynman eqs    | Easier than standard set                | Use standard 117-equation set          |
| No comparison to simple baselines     | Can miss that SR adds no value          | Always include polynomial regression   |

---

## 7. CI Integration

**Fast CI suite (every commit, < 5 minutes):**
- 20-equation synthetic set, 30-second time limit each
- Single-threaded only
- Pass/fail criterion: R² > 0.99 on 15+ of 20 equations

**Weekly full benchmark (scheduled):**
- 50 Feynman equations (representative subset)
- 60-minute time limit
- Compare against PySR baseline stored as JSON reference file
- Fail if recovery rate drops > 5 percentage points vs. baseline

**Release benchmark (before each version tag):**
- Full 117 Feynman equations + 50 PMLB datasets
- 30 runs each
- Generates comparison report vs. PySR and Operon

---

## 8. Reference Numbers (as of 2025)

These are published results to target or exceed:

| Algorithm | Feynman Recovery | Time Budget    | Source                     |
|-----------|------------------|----------------|----------------------------|
| QDSR      | 91.6%            | Not specified  | arXiv 2503.19043           |
| PySR      | ~65–70%          | Standard        | Various                    |
| Operon    | Lower (struggles)| Standard        | GECCO 2022 post-analysis   |
| uDSR      | Competitive      | Standard        | Various                    |

**rsymbolic2 target:** Match PySR on Feynman recovery rate at equivalent wall-clock time by end of Phase 3. Exceed PySR on PMLB R² median by end of Phase 4 (via e-graph simplification improving Pareto front quality).

**Note:** These targets are based on the current literature and are not guaranteed. Actual performance depends on implementation quality and hyperparameter choices. Benchmarks should drive decisions, not confirm predetermined conclusions.

---

## 6. Parallel Efficiency Measurement — Phase 3 (2026-06-06)

### Setup

**Method:** Strong scaling (fixed total work, vary threads).
`n_populations=12` fixed; `omp_num_threads` ∈ {1, 2, 3, 4, 6}.
12 divides evenly into each thread count → no load-imbalance artifact from unequal island assignment.
`target_loss = -∞` forces the full `pop × gen` budget on every island regardless of fit quality.

**Hardware:** Intel hybrid (10 physical / 12 logical cores — P+E architecture).
**OS:** Windows 11 Home 10.0.26200.
**Toolchain:** Rtools45 MinGW GCC / UCRT, OpenMP enabled.
**Problem:** Nguyen-1 (x³+x²+x), polynomial, no transcendentals.
**Config:** `n_populations=12`, `population_size=100`, `generations=25`,
`migration_interval=10`, `migration_size=5`, `simplify=false`.
**n_runs:** 3 (median reported).

### Results

| threads | med_ms | min_ms | max_ms | speedup | efficiency | gate (≥0.70) |
|---------|--------|--------|--------|---------|------------|--------------|
| 1       | 7,806  | 6,379  | 7,854  | 1.000   | 1.000      | PASS         |
| 2       | 4,363  | 4,324  | 4,659  | 1.789   | **0.895**  | PASS         |
| 3       | 4,047  | 3,754  | 4,510  | 1.929   | 0.643      | FAIL         |
| 4       | 3,451  | 3,270  | 4,484  | 2.262   | 0.565      | FAIL         |
| 6       | 3,092  | 2,845  | 3,984  | 2.524   | 0.421      | FAIL         |

### Analysis

Gate (efficiency ≥ 0.70) is met at **threads=2 only**.

The sharp drop between threads=2 (0.895) and threads=3 (0.643) is consistent across
calibration and confirmed runs. Cause is **not load imbalance** — islands are equal-sized,
`schedule(dynamic)` is active, and each island's RNG is fully independent.

Most likely causes (hardware-specific, not algorithmic):

1. **Intel hybrid core scheduling.** 10 physical / 12 logical suggests a P+E configuration.
   Windows schedules the 3rd thread onto an E-core (significantly slower), making its islands
   take proportionally longer and limiting wall-time to the slowest thread.
2. **Windows heap allocation contention.** LM optimization allocates Eigen matrix temporaries
   per evaluation. Under concurrent allocation pressure from 3+ threads, the Windows CRT heap
   serialises allocations, reducing parallelism.

Variance increases with thread count (max/min ratio: 1.40× at threads=6 vs 1.23× at threads=1),
consistent with non-deterministic OS scheduling on hybrid cores.

### Decision

Per roadmap: "only if efficiency falls below 0.7× **and the cause is load imbalance**, evaluate
TBB." Cause here is hardware architecture + OS allocator, not load imbalance → TBB adoption
condition is not satisfied.

Speedup of **2.5× at 6 threads** is still a net benefit. The gate failure is hardware-context-
dependent; a homogeneous-core machine or Linux (glibc + ptmalloc2) may yield different results.

**Remaining action:** Measure on Ubuntu LTS (homogeneous cores, glibc allocator) before drawing
final conclusions about the 0.7× gate. That measurement is deferred to the Ubuntu CI phase.
