# Feynman Benchmark Plan

**Date:** 2026-06-08

---

## 1. Why now — prerequisites met

docs/18 added the `square` (unary) and `pow` (binary, safe semantics) operators and
was committed on 2026-06-08 (6747cf8), verified on Windows 11 (Rtools45 MinGW/UCRT)
and Ubuntu LTS. The Nguyen no-regression run is archived at
`benchmarks/results/nguyen_gate_20260608.csv` (10/10 problems, no new failures).

The Feynman dataset is power-heavy: inverse-square laws, relativistic denominators
(`sqrt(1-v^2/c^2)`), Planck/Bose-Einstein energy with `omega^3`, Larmor radiation with
`c^3`. Without `pow`/`square`, a large fraction of equations was structurally unreachable;
any Feynman run before 6747cf8 would have measured operator coverage, not search quality.
That measurement would have been misleading and would have needed to be re-done after
adding the operators — a wasted cycle. The operators are now settled; benchmarking can
begin.

**Full operator set (as of 6747cf8):**
- Unary: `neg`, `exp`, `log`, `sin`, `cos`, `sqrt`, `tanh`, `abs`, `square`
- Binary: `add`, `sub`, `mul`, `div`, `pow`

---

## 2. Dataset

**Source:** Udrescu & Tegmark (2020), "AI Feynman: A physics-inspired method for
symbolic regression", *Science Advances* 6(16) eaay2631.
Supplementary table S2 defines 100 equations (Feynman I) with variable names,
physical domains, and evaluation functions.
Extended set (Feynman II, 20 equations): Udrescu et al. (2020), *NeurIPS 2020*.
Combined 120 equations; the literature standard is **117 equations** (3 excluded
for ambiguous or trivially degenerate forms; exact exclusions documented below in §3).

**Access for variable ranges:** https://github.com/SciML/AIFeynman

**Data generation:** R only, no Julia or external download at runtime.
Uniform random sampling within the published variable ranges per equation.

| Split | Size | Seed |
|-------|------|------|
| Training | 1 000 | `DATA_SEED = 42L` |
| Test | 500 | `DATA_SEED + 1L = 43L` |

Training data is what `symbolic_regression()` sees. Test data is used for the final
NMSE gate (held-out evaluation).

**Current limitation:** `symbolic_regression()` returns the training `loss` (SSR) but
no `predict()` method for new data yet exists in the R API. For Stages 0–1 (development
gates), in-sample NMSE on the training split is used as a proxy — the same protocol as
the Nguyen gate (which also uses in-sample loss). For Stage 2 (publication quality),
implementing `predict.rsymbolic2()` or evaluating the returned expression string on the
test split is a prerequisite before recording the final recovery rate.

Output directory: `benchmarks/data/feynman_I_<id>.csv`, `benchmarks/data/feynman_II_<id>.csv`
Columns: `x1, x2, ..., xp, y` (consistent with Nguyen data format).

---

## 3. Operator set and reachability analysis

**Classification method:** An equation is *reachable* if every function it requires
appears in the operator set; *reachable-by-composition* if it can be expressed via
operator combinations (e.g. `1/x = div(1, x)`, `x^3 = pow(x, 3)`,
`exp(-x) = exp(neg(x))`); *unreachable* if it requires an operator outside the set
(e.g. `arcsin`, `Gamma`, `erf`, `Bessel`).

The **official recovery-rate denominator** is the count of reachable + reachable-by-
composition equations. Unreachable equations are excluded from the count and listed
explicitly below so the denominator is transparent. Equations are never silently
dropped to inflate the rate.

**Operators required but NOT available:**
- `arcsin` / `arccos` / `arctan`: needed by several equations in Feynman II
- `Gamma` function: needed by ~2 equations
- `erf`: needed by ~1 equation
- `Bessel` (J₀, J₁): needed by ~2 equations

Equations requiring these operators are marked `skip = TRUE` in `feynman_datasets.R`
with the `skip_reason` field populated. The exact list is recorded here as equations
are processed in Stage 0.

**Dev subset (Stages 0–1):** `benchmarks/feynman_datasets.R` defines 25 equations
selected to cover: (a) pow-heavy cases for the smoke test, and (b) diversity of
operators and variable counts for the dev gate. See §5 and the file itself for the
full list.

**Full 117-equation set:** Processed in Stage 2. The reachability classification for
all 117 equations will be completed as part of Stage 0 analysis and appended to this
document.

---

## 4. Staged run plan

| Stage | Scope | Runs/problem | Purpose | Gate criterion |
|-------|-------|-------------|---------|---------------|
| **0 — smoke** | 10 pow-heavy equations (see §5) | 3 | No NaN/Inf crash; pipeline works end-to-end on both OSes | All 3 runs return finite NMSE; no crash |
| **1 — dev gate** | 25 dev-subset equations | 5 | Baseline recovery rate; catch regressions before full run | ≥ 18/25 problems recovered (majority of 5 seeds) |
| **2 — full** | All reachable equations (≥ 90 expected) | 30 | Paper-quality results per docs/04 | Recovery rate documented; no floor set until results arrive |
| **PySR comparison** | Same full set as Stage 2 | 30 | Fair head-to-head with SymbolicRegression.jl | Per docs/04 §7: gap < 5 pp or explained |

Stage 0 must pass before Stage 1 begins. Stage 1 must complete before Stage 2 is
launched (to avoid wasting the full 30-run compute on a broken pipeline or bloated
operator set).

**Time budget per problem:**

| Stage | Per-problem limit | Rationale |
|-------|------------------|-----------|
| Stage 0 | 60 s | Smoke test; pipeline-health check only, not a recovery measurement (see §6 Stage 0 profile) |
| Stage 1 | 5 min | Dev gate; consistent with Nguyen gate experience |
| Stage 2 | 10 min | Publication quality; Feynman equations are harder than Nguyen |
| PySR comparison | 10 min | Match Stage 2 budget; equalization on rsymbolic2 side only |

---

## 5. Dev subset equations

The 25 equations used in Stages 0–1, organized by stage assignment and variable count.
All are from Feynman I unless noted.

### Stage 0 — smoke test (pow-heavy)

| Key | ID | Formula | Vars | Domains | Required new ops |
|-----|----|---------|------|---------|------------------|
| gaussian | I.6.20a | exp(−θ²/2)/√(2π) | 1 | θ∈[1,3] | square |
| rel_mass | I.10.7 | m₀/√(1−v²/c²) | 3 | m₀∈[1,5], v∈[1,2], c∈[3,10] | square |
| coulomb | I.12.2 | q₁q₂/(4πεr²) | 4 | all∈[1,5] | square |
| spring_pe | I.14.4 | ½kx² | 2 | k,x∈[1,5] | square |
| lorentz_x | I.15.1 | (x−ut)/√(1−(u/c)²) | 4 | x∈[5,10], u,t∈[1,2], c∈[3,10] | square |
| rel_mom | I.15.10 | mv/√(1−v²/c²) | 3 | m∈[1,5], v∈[1,2], c∈[3,10] | square |
| harmonic_ke | I.24.6 | ¼m(ω²+ω₀²)x₁² | 4 | all∈[1,5] | square |
| interference | I.37.4 | I₁+I₂+2√(I₁I₂)cos(δ) | 3 | I₁,I₂∈[1,5], δ∈[1,5] | — (sqrt pre-existed) |
| bohr_radius | I.38.12 | 4πε₀ħ²/(mq²) | 4 | all∈[1,5] | square |
| larmor_rad | I.32.17 | q²a²/(6πε₀c³) | 4 | all∈[1,5] | square, pow(c,3) |

### Stage 1 additions — dev gate (15 more equations)

| Key | ID | Formula | Vars | Domains |
|-----|----|---------|------|---------|
| planck | I.41.16 | ħω³/(π²c²(exp(ħω/kT)−1)) | 5 | ħ,ω,c,k,T∈[1,3] |
| center_mass | I.18.4 | (m₁r₁+m₂r₂)/(m₁+m₂) | 4 | all∈[1,5] |
| torque | I.18.12 | rF sin(θ) | 3 | all∈[1,5] |
| larmor_freq | I.34.8 | qvB/r | 4 | all∈[1,5] |
| doppler_rel | I.34.14 | ω₀(1+v/c)/√(1−v²/c²) | 3 | ω₀∈[1,5], v∈[1,2], c∈[3,10] |
| einstein_smol | I.43.31 | mob·k_B·T | 3 | all∈[1,5] |
| driven_osc | I.50.26 | x₁(cos(ω₀t)+α cos(ωt)) | 5 | all∈[1,3] |
| heat_conduct | II.2.42 | κ(T₂−T₁)/d | 4 | all∈[1,5] |
| boltzmann_dist | II.11.3 | n₀ exp(−mgx/(k_BT)) | 6 | all∈[1,5] |
| clausius_moss | II.11.27 | n₀α/(1−n₀α/3) | 2 | n₀,α∈[0.5,1.5] |
| clausius_moss2 | II.11.28 | 1+n₀α/(1−n₀α/3) | 2 | n₀,α∈[0.5,1.5] |
| bohr_magneton | II.34.29a | qħ/(2m) | 3 | all∈[1,5] |
| bose_einstein | III.4.33 | ħω/(exp(ħω/k_BT)−1) | 4 | all∈[1,3] |
| lens_eq | I.27.18 | d₁d₂/(d₂+nd₁) | 3 | all∈[1,5] |
| newtons_grav | I.9.18s | Gm₁m₂/(dx²+dy²) | 5 | G,m₁,m₂∈[1,5], dx,dy∈[1,3] |

---

## 6. Protocol (from docs/04)

**Exact recovery:** NMSE < 10⁻⁴ on the evaluation split (in-sample for Stages 0–1;
held-out test for Stage 2 once `predict.rsymbolic2()` is available).

**Runs:** 30 per problem in Stage 2; see §4 for Stage 0–1.

**Seed management:** Seeds drawn from a fixed master sequence: `seed_i = i` for
`i = 1, ..., n_runs`. Same seed sequence given to both tools, enabling paired tests.

**Hyperparameters:** Fixed before benchmarking; not tuned on Feynman results.

```r
BENCH_PARAMS <- list(
  population_size       = 500L,
  n_populations         = 4L,
  generations           = 500L,       # more generations than Nguyen (harder problems)
  tournament_size       = 5L,
  unary_ops             = c("neg", "exp", "log", "sin", "cos",
                            "sqrt", "tanh", "abs", "square"),
  binary_ops            = c("add", "sub", "mul", "div", "pow"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5
  # optimize_probability and parsimony: shipped defaults (0.1 / 1e-3)
)
```

These are the defaults for Stage 1–2. If Stage 1 shows systematic failures on
hard equations, generation count may be increased before Stage 2 — but only with
the Stage 1 failure pattern as evidence (not speculatively).

**Stage 0 smoke profile (lighter).** Stage 0 only checks that the pipeline runs
end-to-end and returns a finite NMSE — it does **not** measure recovery. Running it
at the full `BENCH_PARAMS` above is wasteful: a single bloated `pow`-tree `fit()` can
take minutes (the deadline is checked at step boundaries but cannot interrupt one
in-flight LM optimisation; see `evolutionary_search.cpp`), so a full-budget smoke run
overshoots its time limit several-fold. The smoke test therefore uses a reduced budget:

```r
SMOKE_PARAMS <- modifyList(BENCH_PARAMS, list(
  population_size = 200L,   # vs 500L
  n_populations   = 2L,     # vs 4L
  generations     = 100L,   # vs 500L
  timeout_seconds = 60L     # per-problem wall-clock cap (overshoots to one fit() call)
))
```

Lower recovery under `SMOKE_PARAMS` is expected and irrelevant: the Stage 0 gate is
"all runs return a finite NMSE", not a recovery threshold. Recovery is measured at
Stage 1 onward with the full `BENCH_PARAMS`.

**Statistical reporting (Stage 2 only):**
- Median recovery rate ± IQR across all reachable equations
- Paired Wilcoxon signed-rank vs. PySR on per-problem recovery fraction (0 or 1)
- Significance level α = 0.05, Bonferroni correction for multiple comparisons
- Separate breakdowns: easy (≤ 2 vars), medium (3–4 vars), hard (≥ 5 vars)

**Archive:** All result CSVs in `benchmarks/results/feynman_stage<N>_<YYYYMMDD>.csv`
with columns `problem_id, key, seed, elapsed_sec, loss, nmse, recovered, timed_out,
n_vars, expression`.

---

## 7. PySR comparison

**Tool:** SymbolicRegression.jl (same as docs/15, docs/16; the PySR Python bridge is
broken on this machine). Version pinned and recorded per the docs/04 reporting checklist.

**PySR settings:** Strict documented defaults — no modifications.

**Shared input:** Identical operator set given to both tools:
```julia
# SymbolicRegression.jl operators (analogous to rsymbolic2 BENCH_PARAMS above)
binary_operators = [+, -, *, /, ^]           # ^ = safe_pow in SRjl
unary_operators  = [neg, exp, log, sin, cos, sqrt, tanh, abs, x->x^2]
```

**Thread equalization:** rsymbolic2's `n_populations` is set so wall-clock compute
matches PySR's thread budget, on rsymbolic2's side only (per CLAUDE.md and docs/16).

**safe_pow semantics check (from docs/18 §2.2):** Before the comparison run, confirm
that SymbolicRegression.jl `^` operator handles `x ≤ 0` with the same guard semantics
as rsymbolic2 `safe_pow`. Any divergence is documented as a caveat of the shared-input
assumption — **never** compensated by altering PySR settings.

**Comparison harness:** `benchmarks/05_feynman_pysr_comparison.jl` (new, mirrors
`benchmarks/03_sr_comparison.jl`). Created when Stage 2 rsymbolic2 run is ready.

---

## 8. Files

| File | Created by this plan | Purpose |
|------|---------------------|---------|
| `benchmarks/feynman_datasets.R` | YES | Equation definitions, variable ranges, data generation |
| `benchmarks/export_feynman_data.R` | YES | Materialise CSVs to `benchmarks/data/` |
| `benchmarks/02_feynman_gate.R` | NO (Stage 1 work) | Dev gate runner (25 equations, 5 runs) |
| `benchmarks/03_feynman_full.R` | NO (Stage 2 work) | Full run (all reachable, 30 runs) |
| `benchmarks/05_feynman_pysr_comparison.jl` | NO (Stage 2 work) | PySR comparison harness |

Runner scripts are listed here for reference but not created until the stage begins,
to prevent stub scripts from drifting out of sync.

---

## 9. Acceptance criteria ("done" for this plan)

- [ ] `benchmarks/feynman_datasets.R` written; `source()` loads cleanly; `feynman_dataset()` returns consistent train/test splits
- [ ] `benchmarks/export_feynman_data.R` written; `Rscript benchmarks/export_feynman_data.R` produces `benchmarks/data/feynman_I_*.csv` files with correct columns and row counts
- [ ] Stage 0 smoke test passed: 10 pow-heavy equations, 3 runs each, all return finite NMSE (no NaN/Inf crash)
- [ ] Stage 0 verified on both Windows and Ubuntu (consistent with CLAUDE.md Platform Constraints)

Full benchmark completion (Stage 1→2→PySR) is tracked in docs/05 and future plan documents.

---

## 10. Open risks

**safe_pow semantics drift vs. SymbolicRegression.jl.** Mitigation: verify before Stage 2 comparison; document residual differences in this file §7 rather than papering over them.

**Search-space inflation.** `pow` with a free exponent enlarges the search space and can
slow convergence or increase bloat. The Nguyen no-regression run (6747cf8) shows no
timing regression at Nguyen scale; Feynman with longer equations may behave differently.
If Stage 1 timings show systematic slowdown vs. Nguyen, increase the time limit for
Stage 2 (with Stage 1 data as evidence) before committing to the full run.

**In-sample NMSE proxy (Stages 0–1).** Using training-set NMSE instead of held-out
NMSE means the recovery criterion is slightly more lenient than the Stage 2 paper standard.
This is acceptable for a development gate but must not be reported as a Feynman recovery
rate without qualification. The gate and the publication metric are distinct.

**6-variable equation (II.11.3).** This problem has 6 input variables, which is
near the practical recovery limit at the current search budget. It is included in the
dev gate to characterise capability, not as a required pass. If it consistently fails,
it is moved to a "hard / open" bucket rather than blocking the gate.
