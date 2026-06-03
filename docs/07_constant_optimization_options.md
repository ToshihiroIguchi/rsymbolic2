# Constant Optimization: Options, Feature Analysis, and Phase 2 Measurement Plan

**Date:** 2026-06-03
**Status:** Investigation for a Phase 2 decision. **Nothing here is decided.** Neither
"avoid Ceres" nor "adopt a self-implemented LM" is a conclusion. Phase 2 will measure
candidates and decide.

**Scope of context:** All maintainability/distribution statements are specific to the
**R-package distribution environment built with Rtools/MinGW + UCRT**, not to Windows
in general (where Ceres's MSVC build is mature and well supported).

---

## 0. The decision space is not a dichotomy

An earlier framing ("self-implemented LM vs full Ceres") was a false dichotomy. There
are at least **four realistic options**, two of which are header-only and avoid the
R/MinGW distribution problems of full Ceres entirely:

| Option | What it is | External/system dep for R | Precedent |
|--------|------------|---------------------------|-----------|
| **A. Eigen MINPACK LM** | `unsupported/Eigen/NonLinearOptimization` (MINPACK-derived Levenberg-Marquardt: `lmpar`, `qrsolv`, `dogleg`, `covar`, `fdjac1`) | **None** — shipped inside RcppEigen | **Operon uses exactly this** |
| **B. Ceres TinySolver** | Single header `tiny_solver.h`, header-only LM | **None** — vendor one BSD-3 file | Part of Ceres, designed to be vendored standalone |
| **C. Bespoke LM from scratch** | Hand-written LM | None (Eigen for linear algebra) | This is the high-cost option the user rightly flagged |
| **D. Full Ceres Solver** | The complete library | Abseil (ABI-unstable), MinGW/UCRT build, no CRAN precedent | See `06_windows_dependency_risk.md` |

Primary sources:
- RcppEigen ships `inst/include/unsupported/Eigen/NonLinearOptimization`
  (github.com/RcppCore/RcppEigen). Eigen's Non-Linear Optimization module is a port of
  MINPACK's `lmder`/`lmdif`. Source: Eigen-unsupported docs (eigen.tuxfamily.org).
- Operon "employs a C++ port of the well-known MINPACK routines, as provided by the
  Eigen library" for Levenberg-Marquardt constant tuning. Source: Operon / SRBench
  paper (cavalab.org/assets/papers/de Franca et al. 2023).
- Ceres TinySolver header states: "This code has no dependencies beyond Eigen,
  including on other parts of Ceres, so it is possible to take this file alone and put
  it in another project without the rest of Ceres." BSD-3. Source:
  github.com/ceres-solver/ceres-solver/blob/master/include/ceres/tiny_solver.h.

**Consequence for the user's concern:** the worry that *writing an LM from scratch*
(Option C) underestimates damping/singular-matrix/restart/ill-conditioning work is
**correct** — and is largely **avoidable**, because Options A and B provide a mature,
header-only LM that is R-distributable with zero system dependency. The realistic
"self" path is *integration + glue around a vetted LM*, not *re-deriving MINPACK*.

---

## 1. What Ceres actually implements (full feature inventory)

Source: Ceres Solver docs, *Solving Non-linear Least Squares* (`nnls_solving`) and
*Modeling Non-linear Least Squares* (`nnls_modeling`).

### 1.1 Trust-region methods
- **Levenberg-Marquardt** with diagonal damping; both "exact step" (Cholesky/QR) and
  "inexact step" (truncated/iterative) variants.
- **Dogleg** and **Subspace Dogleg** (2D subspace of Gauss-Newton + Cauchy; fewer
  linear solves per iteration than LM in some regimes).
- **Trust-region radius update** driven by ρ (actual/predicted decrease), configurable.
- **Jacobian column scaling** to improve conditioning.

### 1.2 Line-search methods (alternative to trust region)
- STEEPEST_DESCENT, NONLINEAR_CONJUGATE_GRADIENT (Fletcher-Reeves / Polak-Ribière /
  Hestenes-Stiefel), **BFGS**, **L-BFGS**.
- Armijo and strong-Wolfe line searches; bisection/quadratic/cubic interpolation.

### 1.3 Linear solvers
- Dense: **DENSE_QR**, **DENSE_NORMAL_CHOLESKY**.
- Sparse: SPARSE_NORMAL_CHOLESKY (SuiteSparse/Accelerate/Eigen).
- Iterative: CGNR, ITERATIVE_SCHUR, SCHUR_POWER_SERIES_EXPANSION.

### 1.4 Preconditioners (for iterative solvers)
- IDENTITY, JACOBI, SCHUR_JACOBI, CLUSTER_JACOBI, CLUSTER_TRIDIAGONAL, SUBSET, etc.

### 1.5 Numerical extras
- **Mixed-precision** factorization with iterative refinement.
- Multiple **convergence criteria**: function tolerance, gradient tolerance, parameter
  tolerance, iteration/time limits, trust-region radius bounds.

### 1.6 Modeling features
- Derivatives: **AutoDiff** (Jet), **NumericDiff** (forward/central/Ridders),
  **Analytic**.
- **Robust loss functions**: Huber, SoftL1, Cauchy, Arctan, Tolerant, Tukey.
- **Bounds constraints**: `SetParameterLowerBound/UpperBound`.
- **Manifolds / local parameterizations**: Euclidean, Quaternion, Sphere, Line.
- **Covariance estimation** post-optimization.

---

## 2. What a minimal Eigen-based implementation "loses"

Two sub-cases must be distinguished, because they lose very different things.

### 2.1 Using Eigen MINPACK LM (Option A) or TinySolver (Option B)
These already provide: LM with damping, pivoted-QR / normal-equations dense solve,
trust-region/`lmpar` step control, column scaling, and standard convergence tests.
Relative to full Ceres they **lack**:
- Dogleg / subspace dogleg (alternative to LM — not additive value for tiny problems).
- Line-search family (BFGS/L-BFGS/NCG) — alternative strategies, not needed alongside LM.
- Sparse and iterative linear solvers, preconditioners, Schur machinery — irrelevant to
  a 1–5 parameter dense problem.
- Built-in robust loss functions (Huber/Cauchy/…) — would be hand-added if needed.
- Bounds constraints and manifolds — bounds are occasionally useful; manifolds are not
  (constants are plain Euclidean scalars).
- Covariance estimation — only relevant to final reporting, not the inner loop.
- Mixed-precision factorization — a large-problem micro-optimization.

### 2.2 Writing LM from scratch (Option C)
Here the "lost" items are the things one must **re-implement and re-validate**: the LM
damping-parameter update (`lmpar`), pivoted QR (`qrsolv`), trust-region radius logic,
column scaling, and the convergence tests. This is precisely the underestimated cost.
Options A/B exist specifically to avoid paying it.

---

## 3. Feature necessity for k ≤ 5 constant optimization

The problem is **low-dimensional, dense, called very many times in the inner loop**.
The SR literature notes constant optimization is typically run with a small fixed
budget per expression (e.g. BFGS with ~10 iterations), i.e. it is deliberately
lightweight, not a precision-critical solve. Source: "Benchmarking symbolic regression
constant optimization schemes" (arXiv 2412.02126).

| Feature | Necessity for k≤5 | Rationale |
|---------|-------------------|-----------|
| LM with diagonal damping | **Essential** | Core method; handles mild ill-conditioning via regularization |
| Dense linear solve (QR or normal Cholesky) | **Essential** | Tiny dense system |
| Convergence criteria (grad/step/cost tol, max iter) | **Essential** | Needed to stop cheaply in the inner loop |
| Gradient via forward-mode autodiff (dual numbers) | **Essential** | Already implemented in rsymbolic2 core |
| Random-restart from perturbed x0 | **Essential** | SR-specific; mitigates initial-value dependence; added by us regardless of backend |
| NaN/Inf guards (safe operators, reject non-finite steps) | **Essential** | Expressions routinely evaluate to non-finite values |
| Jacobian column scaling | **Desirable** | Cheap; improves conditioning; provided by MINPACK/TinySolver |
| Pivoted QR / rank-revealing solve | **Desirable** | Helps redundant-constant (rank-deficient) skeletons; in MINPACK LM |
| Numeric-diff fallback | **Desirable** | Safety net for operators without a dual-number rule |
| Bounds constraints | **Desirable** | Occasionally keeps constants in valid domains (e.g. log) |
| Robust loss (Huber/Cauchy) | **Desirable (later)** | Useful for noisy/outlier data; default SR loss is MSE; defer |
| Dogleg / subspace dogleg | **Unnecessary** | Alternative to LM; no clear benefit at this size |
| Line-search BFGS/L-BFGS/NCG | **Unnecessary** | Alternative family; LM suffices for dense NNLS |
| Sparse linear solvers (SuiteSparse, etc.) | **Unnecessary** | Problem is tiny and dense |
| Iterative solvers + preconditioners | **Unnecessary** | For large/sparse systems |
| Schur complement / bundle-adjustment | **Unnecessary** | Not applicable |
| Manifolds / local parameterizations | **Unnecessary** | Constants are Euclidean scalars |
| Covariance estimation | **Unnecessary (inner loop)** | Optional for final reporting only |
| Mixed-precision factorization | **Unnecessary** | Large-problem micro-optimization |

**Reading:** every feature classified Essential is provided by Eigen MINPACK LM and by
Ceres TinySolver. The Desirable items are mostly present (scaling, pivoted QR) or
cheaply addable (bounds, numeric-diff fallback, robust loss). The Unnecessary list is
the bulk of what full Ceres adds — i.e. full Ceres's marginal value for *this* problem
is low, which is the crux the user identified.

---

## 4. Expected code volume

Estimates are for the rsymbolic2-side code (functor adapter to dual-number evaluation,
restart loop, NaN/Inf guards, acceptance test, unit tests). They exclude the LM core
itself for Options A/B (that code is the library's).

| Option | Optimistic | Realistic | Pessimistic | Notes |
|--------|-----------:|----------:|------------:|-------|
| **A. Eigen MINPACK LM** | ~150 | ~300–500 | ~800 | Glue + `Functor` adapter + restart + guards + tests. LM core is Eigen's. |
| **B. Ceres TinySolver** | ~150 | ~300–600 | ~800 | Similar glue; vendor one header. Slightly different API/autodiff adapter. |
| **C. Bespoke LM from scratch** | ~400 | ~800–1200 | ~2000+ | Must implement+debug+validate damping, pivoted QR, scaling, convergence, edge cases. Highest ongoing correctness/maintenance risk. |
| **D. Full Ceres** | ~200 (integration) | ~200 | ~200 | LOC is small; the real cost is build/distribution (Abseil/MinGW/UCRT), not code. |

The LOC figures deliberately show that Option C is the expensive *and* risky one, while
A/B are modest glue. The earlier "self-implemented LM" framing implicitly meant C;
A/B are the better-evidenced realizations of the same "no heavy dependency" goal.

---

## 5. Phase 2 Measurement Plan

**Objective:** decide among {sufficient with a header-only LM} / {adopt full Ceres} /
{support both} on evidence, for the Rtools/MinGW/UCRT R-distribution context.

### 5.1 Candidates to measure
1. **Eigen MINPACK LM** (Option A) — primary candidate to characterize.
2. **Ceres TinySolver** (Option B) — header-only comparison.
3. **Full Ceres LM** (Option D) — reference for solution quality/robustness ceiling.
   Build it however is convenient for the *measurement* (it need not be the R build).
4. **Random-restart-only baseline** (no gradient solver) — lower bound / sanity check.

(Option C bespoke is not built for measurement; if A/B suffice, C is moot.)

### 5.2 Benchmark problems (constants are known)
- **Feynman** subset (representative equations with non-trivial constants).
- **Nguyen-1 … Nguyen-10** (standard GP-SR targets; several have constants).
- **Keijzer-1 … Keijzer-15** (include constants and ill-scaled targets).
Each target supplies a known constant vector `c*` for accuracy checks.

### 5.3 Two measurement layers

**Layer 1 — Optimizer microbenchmark (isolates the solver):**
For a fixed library of expression *skeletons* with known optimal constants:
- **Convergence success rate:** fraction with `||ĉ − c*|| / ||c*|| < ε` (e.g. ε=1e-3).
- **Final loss** vs known optimum.
- **Per-call latency:** median and p95 wall-clock per optimization call.
- **Iterations to convergence.**
- **Ill-conditioning robustness** on a stress set (see 5.4): success rate there.

**Layer 2 — End-to-end search (the figure that actually matters):**
Run the full rsymbolic2 search with each backend, identical seeds and time budget:
- **Equation recovery rate** on Feynman/Nguyen/Keijzer.
- **Total wall-clock** to first recovery.
- **Fraction of runtime spent in constant optimization** (profile).

### 5.4 Ill-conditioning / robustness stress set
Construct skeletons that exercise the hard cases the user listed:
- **Rank-deficient / redundant constants:** `a + b` (both additive), `a*b*x`
  (multiplicative collinearity) — singular/near-singular Jacobian.
- **Initial-value sensitivity:** `a*exp(b*x)` with poor x0; multi-basin targets.
- **Ill-scaling:** constants differing by many orders of magnitude.
- **Near-non-finite regions:** `log`, `1/x`, `sqrt` near domain edges.
Measure: success rate, divergence/NaN rate, sensitivity to restart count.

### 5.5 Distribution-cost evaluation (qualitative, but recorded)
For each candidate, record concretely in the R/MinGW/UCRT context:
- Does it build under Rtools (MinGW) for an R package without system libraries?
- Added install/`R CMD check` time and any CRAN-policy friction.
- Number of external/system dependencies introduced.
(Options A/B are expected to add none; D adds Abseil and the issues in doc 06. This is
to be confirmed, not assumed.)

### 5.6 Decision rule (applied at the Phase 2 gate)
Choose the option that minimizes maintenance/distribution cost **among those whose
measured convergence and speed are within an acceptable margin** of the best candidate.
Concretely:
- If a header-only LM (A or B) is within a small margin of full Ceres on recovery rate
  and robustness, **prefer it** (priorities rank portability/simplicity above
  performance). 
- Adopt **full Ceres** only if it shows a **material, measured** advantage that the
  header-only options cannot reach, accepting its R-distribution cost knowingly.
- **Support both** only if no single option is adequate across the problem range and
  the maintenance of two backends is justified by the measured spread.

The margin thresholds (e.g. acceptable recovery-rate gap, acceptable latency ratio)
will be fixed **before** measurement to avoid post-hoc bias.

---

## 6. Position (explicit)

- Ceres is **not "dangerous."** TBB is **not "dangerous."** Their adoption cost in the
  R/MinGW/UCRT context is real but is to be **weighed against measured benefit**, not
  assumed.
- A header-only Eigen-based LM (Option A or B) is the **leading candidate** precisely
  because it sidesteps the distribution cost while reusing a vetted LM — but its
  numerical adequacy on the stress set is **unverified** and must be measured.
- **No adoption is decided.** Phase 2 runs the plan above and the gate decides among
  "header-only LM sufficient," "adopt Ceres," or "support both."

## Sources
- Ceres *Solving NNLS*: https://ceres-solver.readthedocs.io/latest/nnls_solving.html
- Ceres *Modeling NNLS*: https://ceres-solver.readthedocs.io/latest/nnls_modeling.html
- Ceres TinySolver header: https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/tiny_solver.h
- RcppEigen NonLinearOptimization: https://github.com/RcppCore/RcppEigen/blob/master/inst/include/unsupported/Eigen/NonLinearOptimization
- Eigen-unsupported Non-Linear Optimization module: https://eigen.tuxfamily.org/dox/unsupported/group__NonLinearOptimization__Module.html
- Operon / SRBench (MINPACK via Eigen for LM): http://cavalab.org/assets/papers/de%20Franca%20et%20al.%20-%202023%20-%20Interpretable%20Symbolic%20Regression%20for%20Data%20Science.pdf
- Constant optimization schemes benchmark: https://arxiv.org/abs/2412.02126
