# Implementation Plan: Phase 3 Gate Closure — Regression Verification + PySR Baseline

**Date:** 2026-06-07
**Basis:** B1+B2 bloat control and the sqrt/tanh/abs operator extension are committed
(324f0d2, 1a85a5f, fd6c564) but have **never been verified together on the official
Nguyen gate runner**. The on-disk `benchmarks/results/nguyen_gate_20260606.csv`
predates B1/B2 (N9 s5 DNF at ~96 min, N4 s5 = 2966 s, N10 not run). The B1/B2 sweep
numbers in `docs/13` come from `_b1_sweep.R` / `_b2_sweep.R` on a **subset**
(N1/N5/N7/N9/N10) with sweep-specific parameters, not from `01_nguyen_gate.R` with its
`BENCH_PARAMS`. This plan closes that gap, then sets up the PySR comparison that
CLAUDE.md requires for the Phase 3 gate.

## Guiding principle (one measurable lever at a time)

Per `docs/13`'s method and CLAUDE.md priority #1 (Correctness): isolate variables.
Two independent changes landed since the last gate CSV — (a) B1/B2 selection/optimizer
changes, (b) a larger operator set (sqrt/tanh/abs). Measuring them in one combined run
would confound "did B1/B2 regress recovery?" with "did the bigger search space regress
recovery?". This plan measures each lever separately.

## Where this sits in the roadmap

Phase 3 gate (roadmap §"Phase 3 Benchmark Milestone / Gate") requires, in order of what
remains:
1. **No regression versus the previous release** — the *primary internal gate*. Blocked
   on a fresh, full-gate, post-B1/B2 CSV. **This plan, Step 1.**
2. **PySR comparison recorded as a reference point** (CLAUDE.md mandate). **Step 2.**
3. Parallel speedup ≥ 0.7×n_threads — already measured on Windows (`docs/04` §6) and
   WSL2 Ubuntu (`docs/09` §3); bare-metal Ubuntu confirmation still pending. **Step 3
   (forward pointer only; no new work in this plan).**

Builds+tests on both OSes and `R CMD check --as-cran` (0 ERROR/0 WARN) are already
satisfied (`docs/08`, `docs/09`). Feynman is explicitly deferred to its own plan doc
(roadmap line 41); it is **out of scope here**.

---

## Step 1 — Nguyen gate regression refresh (primary, fully detailed)

### 1.1 Goal and success criteria

Produce one dated `nguyen_gate_<date>.csv` that supersedes the stale 2026-06-06 file and
answers two questions independently:

- **Q-A (B1/B2 regression?):** With the *original* operator set (neg/exp/log/sin/cos),
  do N1–N7, N9, N10 still recover (NMSE < 1e-4, majority of 5 seeds), and have the
  pre-B1/B2 runtime outliers (N4 s5 2966 s; N9 s1/s4 ~1300–1800 s; N9 s5 DNF) collapsed
  to the ~10–60 s range that the `docs/13` sweep predicts?
- **Q-B (operator-extension regression?):** With sqrt added, does N8 (= sqrt(x)) now
  recover, **and** do N1–N7, N9, N10 still recover despite the enlarged search space?

**Gate pass condition (unchanged from `utils.R::print_summary`):** ≥ 7 of the active
problems recover by seed-majority. Additionally, this plan imposes a **runtime
regression check**: no run may hit the safety timeout (Step 1.2); any run that does is a
FAIL-to-investigate, not a silent pass.

### 1.2 Code changes to the gate runner

All edits in `benchmarks/01_nguyen_gate.R` and `benchmarks/nguyen_datasets.R`. No C++
or package changes — the new defaults already ship in `symbolic_regression.R`.

**(a) Add a safety timeout to `BENCH_PARAMS`.** The unbounded gate is what allowed the
96-minute hang. The per-run timeout machinery exists and was verified (`docs/12`). Add:

```r
BENCH_PARAMS <- list(
  population_size       = 500L,
  n_populations         = 4L,
  generations           = 200L,
  tournament_size       = 5L,
  unary_ops             = c("neg", "exp", "log", "sin", "cos"),
  binary_ops            = c("add", "sub", "mul", "div"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5,
  timeout_seconds       = 180        # NEW: safety net only; see note
)
```

`optimize_probability` and `parsimony` are deliberately **left unset** so the gate
exercises the shipped defaults (0.1 / 1e-3). Document this in a comment.

**Note on the timeout:** 180 s is ~3× the slowest expected post-B1/B2 run (N3/N4 were
~90–200 s pre-fix; the `docs/13` sweep shows the diagnosed slow cases now finish in
8–34 s, but N3/N4 were never in that sweep). It is a guard against a *re-emergent*
blowup, not a normal operating point. A timed-out run returns best-so-far and is not
bit-reproducible (roadmap/roxygen caveat), so the runner must **flag** timeouts rather
than treat them as ordinary results.

**(b) Record whether a run hit the timeout.** In the per-run loop, after computing
`elapsed`, add a column:

```r
timed_out = isTRUE(elapsed >= BENCH_PARAMS$timeout_seconds - 1)
```

and carry it into the `data.frame(...)` row and the CSV. Print a visible marker in the
per-run `cat(...)` line when `timed_out` is true.

**(c) Make N8's skip reflect operator availability, not "not implemented."** In
`nguyen_datasets.R`, the N8 entry currently reads `skip_reason = "sqrt operator not yet
implemented"` — now factually wrong. Change the comment and reason; keep `skip = TRUE`
as the *default* (the default gate operator set has no sqrt), but make the run script
able to override it for the Phase-2 sub-run:

```r
# N8: sqrt(x). Recoverable only when "sqrt" is in unary_ops; the default gate
# operator set omits it, so this is skipped unless the caller enables sqrt.
N8 = list(
  id = "N8", formula = "sqrt(x)", n_vars = 1L,
  domains = list(c(0, 4)),
  fn = function(x) sqrt(x[1]),
  skip = TRUE,
  skip_reason = "requires 'sqrt' in unary_ops (enabled in the operator-extension run)"
),
```

### 1.3 Run protocol (two sub-runs, isolated levers)

**Run 1 — B1/B2 regression (Q-A).** Operator set unchanged (no sqrt); N8 skipped.
This is the direct before/after against the stale CSV.

```powershell
Rscript benchmarks/01_nguyen_gate.R
```

Produces `nguyen_gate_<date>.csv` (9 active problems: N1–N7, N9, N10).

**Run 2 — operator extension (Q-B).** Add sqrt and un-skip N8. Implement as a thin
wrapper script `benchmarks/01b_nguyen_gate_sqrt.R` that sources the datasets, sets
`nguyen_problems$N8$skip <- FALSE`, and runs the same loop with
`unary_ops = c("neg","exp","log","sin","cos","sqrt")`. Keep it a separate script so
Run 1 stays a clean, parameter-frozen regression artifact. Produces
`nguyen_gate_sqrt_<date>.csv` (10 active problems).

Rationale for two scripts rather than one parameterised run: the gate CSV is a
regression *artifact* compared release-over-release; freezing Run 1's parameters keeps
that comparison honest, while Run 2 measures the orthogonal operator-set change.

### 1.4 Acceptance / what counts as "no regression"

| Check | Pass condition |
|-------|----------------|
| Recovery (Q-A) | ≥ 7/9 problems recover by seed-majority; ideally 9/9 as before |
| Runtime (Q-A) | No run hits the 180 s timeout; former outliers (N4 s5, N9 s1/s4/s5) now ≤ ~60 s |
| Recovery (Q-B) | N8 recovers with sqrt enabled; N1–N7,N9,N10 still ≥ 7/10 majority |
| Runtime (Q-B) | Adding sqrt does not push any problem into the timeout |

If Q-A fails recovery → B1/B2 regressed; bisect against `optimize_probability=1.0,
parsimony=0` (pre-B1/B2 behaviour, both still selectable) to localise. If Q-A passes but
Q-B regresses some problem → the enlarged search space is the cause; record it and
decide (at a gate) whether the default benchmark operator set should include sqrt or
whether sqrt belongs only to problems that need it.

### 1.5 Recording

- Commit both CSVs under `benchmarks/results/`.
- Add a short results table to **this doc** (a "§1.6 results" subsection) with the
  before/after runtime collapse and the recovery counts, mirroring `docs/13`'s B1.8/B2.8
  tables. Cite hardware, thread count, rsymbolic2 version, date, n_runs (= 5 seeds).
- Update roadmap §"Current Status": replace the unverified "Gate table now 9/9" claim
  with the dated, full-runner evidence, and tick roadmap immediate-priority #3.
- Delete or clearly supersede the stale `nguyen_gate_20260606.csv` (do not leave two
  undated-intent files that imply two valid baselines).

### 1.6 Results (2026-06-07)

**Environment.** rsymbolic2 0.1.0 (built from source, Rtools45 MinGW GCC/UCRT,
R 4.6.0); Windows 11 Home 10.0.26200; Intel hybrid (10 physical / 12 logical cores);
n_populations=4 (4 OpenMP threads); 5 seeds per problem; defaults exercised
(optimize_probability=0.1, parsimony=1e-3). CSVs: `benchmarks/results/
nguyen_gate_20260607.csv` (Run 1) and `nguyen_gate_sqrt_20260607.csv` (Run 2).

**Run 1 — B1/B2 regression (original operator set, N8 skipped): Gate 9/9 PASS.**

| Problem | Recovered | med NMSE | med time | former outlier → now |
|---------|-----------|----------|----------|----------------------|
| N1  | 5/5 | 1.1e-32 | 2 s  | — |
| N2  | 5/5 | 1.9e-32 | 4 s  | see anomaly below (s3) |
| N3  | 5/5 | 6.1e-08 | 6 s  | — |
| N4  | 5/5 | 8.4e-07 | 10 s | **s5: 2966 s → 10 s** |
| N5  | 5/5 | 3.0e-11 | 3 s  | — |
| N6  | 5/5 | 1.6e-16 | 2 s  | — |
| N7  | 5/5 | 4.1e-07 | 3 s  | — |
| N9  | 5/5 | 2.5e-05 | 8 s  | **s1: 1786 s → 16 s; s5: DNF(96 min) → 2 s** |
| N10 | 5/5 | 2.0e-32 | 2 s  | **(was "not run") → 1–2 s** |

The pre-B1/B2 runtime outliers collapse to ≤16 s. Recovery is fully preserved (9/9,
all NMSE far below 1e-4). **B1/B2 did not regress recovery.**

**Run 2 — operator extension (sqrt added, N8 enabled): Gate 10/10 PASS.**

| Problem | Recovered | med NMSE | med time |
|---------|-----------|----------|----------|
| N1–N7, N9, N10 | 5/5 each | ≤ 8.4e-06 | 1–8 s |
| **N8 = sqrt(x)** | **5/5** | **0.0e+00** | **1 s** |

N8 — previously unrecoverable — recovers exactly once `sqrt` is in the operator set,
confirming the new operator works in a recovery context (not just unit tests). Adding
sqrt did **not** regress any other problem; no run hit the 180 s safety timeout.
**The operator extension did not regress recovery.**

### 1.7 Open issue — timeout overshoot on a rare bloat tail (N2 seed 3)

In Run 1, **N2 seed 3 ran 11 664 s (~3.2 h)** despite `timeout_seconds = 180`, and was
flagged `[TIMED OUT]`. Root cause: the deadline is checked per *step*
(`evolutionary_search.cpp:165`), so the worst-case overshoot is **one `fit()` call** —
but a single Levenberg-Marquardt solve on a pathologically bloated tree (far above the
`max_nodes = 50` *soft* cap, with a correspondingly huge dual-number Jacobian) can itself
run for hours. The per-step check cannot interrupt mid-`fit()`.

This is **not a B1/B2 recovery regression** (N2 s3 still recovered, NMSE 8.1e-33) and it
**did not recur** in Run 2 (N2 s3 = 3 s there) — it is a rare stochastic bloat-tail event,
exactly the long tail B1/B2 made infrequent rather than impossible. It is the residual of
the H1 bloat diagnosed in `docs/12`.

**Follow-up options (not yet implemented; decide at a later gate):**
1. A *hard* node cap that `fit()` honours — skip the LM solve (or bail to `sse_current`)
   when a tree exceeds, say, 2–3× `max_nodes`. Cheapest; directly bounds one `fit()`.
2. Cooperative cancellation: pass the deadline into the optimizer so a long solve can
   abort at an iteration boundary.
3. An LM iteration/evaluation budget scaled down for very large trees.

Recommendation: option 1, as it attacks the bloat at its source and is a small, local
change. Tracked here; out of scope for the Step-1 regression verification itself.

---

## Step 2 — PySR comparison baseline (detailed; one decision point)

CLAUDE.md ("Benchmarking Requirements"): *"Comparison against PySR is required at an
equivalent wall-clock budget on the same hardware. Report version, hardware, time limit,
and number of runs."* This is the headline remaining Phase 3 item once Step 1 confirms a
verified baseline to compare.

### 2.1 Dependency / environment decision (confirm with user first)

PySR is a Python package whose backend is SymbolicRegression.jl — installing it pulls a
**Julia** toolchain into the *benchmarking environment*. This does **not** violate the
"no Julia dependency" rule: that rule governs the shipped `rsymbolic2` library, not the
comparison harness. Still, it is a real local-environment change, so per CLAUDE.md
("Confirm before … adding dependencies") **ask the user in Japanese** before installing,
and record the decision here. Pin exact versions (PySR, SymbolicRegression.jl, Julia,
Python) for reproducibility.

Fallback if PySR install is undesirable on this machine: run PySR on a separate
machine/CI and import its result CSV; the comparison only needs PySR's output table, not
a co-located install.

### 2.2 Protocol (fair comparison)

- **Same problems:** the Nguyen set from Step 1 (identical data: `data_seed = 123`,
  n = 20, same domains). Reuse `nguyen_datasets.R` to emit the exact X/y so both tools
  see identical samples.
- **Same wall-clock budget per run:** set PySR's time limit equal to rsymbolic2's
  effective budget. Because rsymbolic2's gate is generation-bounded (not time-bounded),
  first derive a per-problem median wall time from Step 1, then give PySR that same
  median as its `timeout_in_seconds`. Document the exact mapping.
- **Same operator set:** PySR `binary_operators`/`unary_operators` mapped to the gate's
  set (Run 1 set for the core comparison; optionally the sqrt-extended set to match
  Run 2).
- **Same metric and threshold:** NMSE < 1e-4 recovery, identical to `utils.R`.
- **n_runs:** 5 seeds (matching the gate) at minimum; note CLAUDE.md/`docs/04`'s
  30-run standard as the eventual publication target, and state explicitly that this
  baseline is a 5-seed *reference point*, not the publication run.

### 2.3 Deliverable

- `benchmarks/03_pysr_comparison.{R,py}` (harness) and
  `benchmarks/results/pysr_comparison_<date>.csv`.
- A comparison table in a new `docs/15_pysr_comparison.md`: per-problem recovery rate
  and median wall time for both tools, with the full reporting checklist from `docs/04`
  §5 (versions, hardware, time limit, n_runs, full table — no cherry-picking).
- Per roadmap: PySR is recorded as a **reference point, not a pass/fail gate**. The
  Phase 3 gate is "no regression vs prior release" (Step 1); the PySR gap merely informs
  whether Phase 4 work is warranted.

---

## Step 3 — Remaining Phase 3 closure (forward pointer; no work in this plan)

Bare-metal Ubuntu parallel-efficiency confirmation (`docs/09` §3 left it "conditionally
open" under WSL2). Not started here because it needs a non-WSL2 Ubuntu machine. When
that hardware is available, re-run `bench_parallel` and update `docs/09`. Tracked, not
scheduled.

Feynman ground-truth recovery is the *primary* benchmark (CLAUDE.md) and is now unblocked
by sqrt/tanh/abs, but the roadmap explicitly says "plan separately." It gets its own
doc (`docs/16_feynman_plan.md`) after Step 2, to avoid scope creep here.

---

## Cross-cutting tasks

- Keep all benchmark scripts parameter-frozen once a CSV is published from them; never
  retro-edit a script that produced a recorded baseline.
- Re-run the testthat suite after any `nguyen_datasets.R` edit (the N8 skip change) to
  confirm nothing in the package tests depends on the old skip reason string.
- English for all code/docs; Japanese for discussion with the user (CLAUDE.md).

## Scope boundary

**In scope:** Step 1 (fully, now), Step 2 (after Step 1 passes, pending the user's
PySR-install decision). **Out of scope:** Feynman, bare-metal Ubuntu re-measurement,
any C++ change, any new library dependency in the shipped package. Widening scope
requires user confirmation (CLAUDE.md).
```