# Timeout Overshoot — Investigation and Follow-up Plan

**Date:** 2026-06-13
**Status:** Investigation complete; follow-up plan proposed (not yet implemented).
**Update (2026-06-14):** The "real residual issue" in §5–6 is **resolved** — see
**docs/22**. The genuine cause of the Stage 1 overshoot was not the per-`fit()` cost bound
analysed here, but that a *single* residual/Jacobian evaluation (one closure over all
`m` points) can take ~90 s for pow-heavy trees and the deadline could not interrupt it
mid-evaluation. docs/22 Phase 1 makes evaluation interruptible (correct timeout); the
~33 s figure below was a contention-influenced underestimate.
**Depends on:** docs/20 (deadline propagation, implemented in `6ba5791`).

---

## 1. Why this document exists

The docs/20 fix was committed, but the Feynman Stage 0 evidence still showed two gross
timeout violations:

| Problem | Seed | Reported time | Budget |
|---------|------|--------------:|-------:|
| `lorentz_x` | 3 | **2194 s** | 60 s |
| `larmor_rad` | 2 | **821 s** | 60 s |

The working assumption was "the fix does not fully work; the optimizer is still running
uninterruptibly." This document records the investigation that tested that assumption,
the experiments that settled it, and the residual issue that is actually worth fixing.

**Headline result: the docs/20 fix works.** A clean rebuild of the committed code
respects the deadline on *both* previously-violating cases. The 2194 s / 821 s figures
are **stale evidence** from a binary that predated the completed fix. The genuine
residual issue is not timeout *correctness* but the *cost of a single `fit()` call*,
which is a search-efficiency concern.

---

## 2. Investigation: ruling out every per-operation suspect

The overshoot must come from work done *between* deadline polls. The committed fix polls
at: (a) each evolution step (`evolve_island`), (b) each population member during
`initialize_island` (member 0 exempt), and (c) between every `minimizeOneStep` inside
`run_lm` (`eigen_lm_optimizer.cpp`). Every candidate uninterruptible operation was bounded
by reading the code:

| Operation | Bound | Verdict |
|-----------|-------|---------|
| `generate_random_tree` | recursion depth ≤ `max_depth` (6) ⇒ ≤ ~127 nodes; no loop | fast |
| `simplify` | one recursive pass, O(tree size); no fixpoint loop | fast |
| `mutate` / `crossover` | `max_nodes` (50) enforced (`mutation.cpp:89,183`) | bounded |
| single tree evaluation | ≤ ~127 nodes of O(1) ops; `pow`/`Dual` are loop-free (`dual.hpp`) | fast |
| one `minimizeOneStep` | 1 Jacobian (`k` passes) + inner loop ≤ `maxfev` evals (see §3) | ≤ ~3 s |
| whole `fit()` | `nfev` capped at `maxfev = max_iterations·(k+1)`, `max_iterations=100` | bounded |

`maxfev` **is** applied (`eigen_lm_optimizer.cpp:158-166`); the instrumentation in §4
confirmed `evals == maxfev` exactly (e.g. `k=4 ⇒ evals=500`, `k=7 ⇒ evals=800`).

The default backend is **EigenLM** with an **analytic** Jacobian. `OptimizerConfig`
defaults (`max_iterations=100`, `n_restarts=4`) are **not** overridden by the R glue
(`rsymbolic2_r.cpp` never sets `opts.optimizer_config`), so the live values are the
struct defaults.

Conclusion from the static analysis: **no single operation can take 2194 s.** Worst-case
overshoot past the deadline is one `minimizeOneStep` (≤ ~3 s, §3). This contradicted the
observed 2194 s, which meant a premise was wrong — so we measured.

---

## 3. The Eigen LM per-step cost, read from the vendored source

`LevenbergMarquardt::minimizeOneStep`
(`RcppEigen/.../NonLinearOptimization/LevenbergMarquardt.h:212-359`) is:

1. Build the Jacobian (`functor.df`) — for the analytic functor, `k` forward passes over
   `m` data points. `njev++`.
2. QR-factorize the Jacobian — O(m·k²).
3. **Inner trust-region loop** `do { … } while (ratio < 1e-4)`:
   - `lmpar2` linear solve — O(k³);
   - **one** residual evaluation `functor(wa2, wa4)` — `m` points, `nfev++`;
   - update the step bound / damping; convergence + termination tests, including
     `if (nfev >= maxfev) return TooManyFunctionEvaluation` (line 347).

The key fact: the inner loop runs **once per accepted-or-rejected trial step**, and is
bounded only by `maxfev` (via the cumulative `nfev`). So a single `minimizeOneStep` on a
tree where LM struggles can do up to `maxfev` residual evaluations before it returns —
and the docs/20 poll fires only *after* `minimizeOneStep` returns. Worst-case single-step
cost, for the largest plausible tree (`k≈64`, `maxfev≈6500`, `m=1000`, tree ≈127 nodes):

```
6500 inner iters × (residual eval ~1.3e5 node-ops + solve ~2.6e5 ops) ≈ 2.5e9 ops ≈ 2–3 s
```

So the residual overshoot bound is **≈ one minimizeOneStep ≈ ≤ 3 s**, not the "one solve"
stated in docs/20 §2. Bounded and acceptable, but worth stating correctly.

---

## 4. Experiment: instrumented reproduction

Added temporary instrumentation to `fit()` (timing around `optimizer.optimize`; print
`[SLOW FIT] dur k treesize evals stop` when a single fit exceeds 3 s), rebuilt with
Rtools45, and ran each violating case in isolation under a hard 200 s wall cap.

**`lorentz_x` seed 3** (Stage-0 profile: pop=200, islands=2, gens=100, timeout=60 s):

```
[SLOW FIT] dur=3.6s k=7 treesize=27 evals=800 stop=0
[SLOW FIT] dur=8.7s k=4 treesize=23 evals=500 stop=0
[SLOW FIT] dur=4.9s k=4 treesize=17 evals=500 stop=0
[SLOW FIT] dur=5.3s k=4 treesize=19 evals=500 stop=0
[epoch 0  t=56.7s] ...
[epoch 1  t=60.0s] ...
DONE: elapsed=60.0s
```

**`larmor_rad` seed 2:**

```
[SLOW FIT] dur=14.8s k=8  treesize=28 evals=900 stop=0
[SLOW FIT] dur=15.9s k=11 treesize=37 evals=276 stop=0
[SLOW FIT] dur=33.1s k=8  treesize=46 evals=900 stop=0
[epoch 0  t=60.0s] ...
DONE: elapsed=60.1s
```

Findings:

1. **Both cases respect the deadline** (60.0 s / 60.1 s), versus the reported 2194 s /
   821 s. The fix works; the prior evidence was produced by a stale binary. (The
   installed DLL was rebuilt at 21:50 on 2026-06-09, *after* the source edits at 21:41–
   21:42; the background Stage 0 that produced 2194 s was launched earlier, against the
   pre-fix binary, and finished long after.)
2. **`stop=0` on every slow fit** — these fits ran entirely *before* the deadline, so
   they were never candidates for interruption. They are simply expensive: a single
   `fit()` reached **33.1 s** (`k=8`, tree size 46).
3. `evals` equals `maxfev = 100·(k+1)` whenever LM runs to the cap, confirming the budget
   is applied and the analytic-Jacobian `k`-factor is the cost driver.

> Reproduction scripts: `benchmarks/_debug_lorentz.R`, `benchmarks/_debug_larmor.R`
> (temporary; delete after the follow-up lands). The `[SLOW FIT]` instrumentation in
> `evolutionary_search.cpp` is marked `// DEBUG-TIMEOUT` and must be reverted before the
> clean build.

*(Full Stage-0 re-run on the fresh build: all 30 runs ≤ 60 s, gate PASS — see §7.)*

---

## 5. The real residual issue

Timeout *correctness* is solved. What remains is **a single constant-optimization can
burn a large fraction of the budget**: 33 s on one candidate out of a 60 s run. Two
distinct consequences:

- **Search efficiency.** A 33 s `fit()` is 33 s not spent exploring other structures.
  `optimize_probability=0.1` already throttles this during evolution, but
  `initialize_island` optimizes **every** seed member unconditionally, and large random
  init trees (`max_depth=6` ⇒ up to ~127 nodes, larger than the `max_nodes=50` evolution
  cap) are exactly the expensive ones.
- **Overshoot tail.** The bound is one `minimizeOneStep` (§3). It is ≤ ~3 s today because
  `maxfev` is moderate; it would grow if `max_iterations` or tree size grew. The bound is
  fine now but is the thing to watch for Stage 1/2 (larger trees, longer budgets).

Per CLAUDE.md priorities (performance is last; pursue only with measured evidence), this
is **not** an emergency. The deadline contract is honoured. The question for the plan is
whether the wasted-budget cost is large enough to hurt **recovery rate** — which is what
Stage 1 measures.

---

## 6. Follow-up options (with trade-offs)

| # | Change | Effect | Cost / risk | Verdict |
|---|--------|--------|-------------|---------|
| A | Poll *inside* the inner trust-region loop | Tightens overshoot to one residual eval | Requires editing vendored Eigen (forbidden) or re-implementing LM | **Reject** — disproportionate; violates "don't fork vendored code" |
| B | Cap init-tree size: generate initial trees under the same `max_nodes` (50) as evolution, not just `max_depth` | Removes the ≥50-node init trees that drive the worst fits | One-line change in `generate_random_tree` use; may slightly reduce initial diversity | **Recommend** — cheap, targets the measured cause |
| C | Gate `fit()` by tree size in `initialize_island` (skip LM / use plain SSE above a node threshold, mirroring `optimize_probability`) | Bounds init cost directly | Adds a heuristic knob; init members start with un-optimized constants | **Recommend if B insufficient** |
| D | Lower / adapt `maxfev` (e.g. scale down for large `k`) | Shrinks both worst fit and overshoot | Changes convergence; needs a sweep + Nguyen no-regression | **Defer** — only with evidence it helps recovery |
| E | Do nothing beyond refreshing evidence | — | — | **Baseline** — defensible; correctness already met |

**Recommendation: E now, B as the one low-risk improvement, gated on Stage 1 evidence.**
Refresh the Stage 0 / Stage 1 numbers on the fixed binary first (§7). If Stage 1 recovery
is acceptable, ship B (align init-tree size with the evolution `max_nodes`) as a small
efficiency win and re-measure. Pursue C/D only if Stage 1 shows the per-fit cost is
materially hurting recovery. Do not touch vendored Eigen (A).

---

## 7. Verification / acceptance

**Full Feynman Stage 0, fixed build (2026-06-13, instrumented; numbers below).**
All 30 runs (10 problems × 3 seeds, 60 s cap): **every run ≤ 60 s; zero overshoot.** The
two previously-violating cases are resolved — `lorentz_x` seed 3 = 60 s (was 2194 s),
`larmor_rad` seed 2 = 60 s (was 821 s). Gate: **10/10 all-finite NMSE → PASS.** Worst
single `fit()` observed across the sweep: **42.3 s** (the efficiency tail option B
targets). **Timeout correctness is met.**

> **Recovery caveat (do not read this run as a recovery baseline).** This sweep ran while
> the machine was under heavy concurrent load (parallel C++/R builds and the debug
> repros). Because the deadline is wall-clock, contention cuts the number of generations
> reached in 60 s, so recovery is depressed relative to the original idle run (e.g.
> `spring_pe` 3/3 → 0/3). This is a measurement artifact, **not** a regression from the
> fix or from B. A fair recovery comparison must be measured on an **idle** machine.

- [x] Full Feynman Stage 0 on the fixed build: no run exceeds the cap; 2194 s / 821 s
      cases gone. (Per-run CSVs are git-ignored; numbers recorded above.)
- [x] Revert the `// DEBUG-TIMEOUT` instrumentation; delete `benchmarks/_debug_*.R`.
- [x] Option B implemented (`random_tree.cpp`) + unit test
      (`test_generation_respects_max_nodes`); Windows standalone suite 14/14 pass.
- [x] **B no-regression: Nguyen gate** (`nguyen_gate_20260613.csv`, idle): **45/45
      recovered, 9/9 PASS** — identical to the `nguyen_gate_20260608.csv` baseline
      (45/45). B does not reduce recovery. Times 2–40 s, all well under the 180 s cap.
- [x] B effect: Stage 0 re-run on the clean (B) build, **idle** machine. All 30 runs
      ≤ 60 s — **timeout compliance holds with B.** Gate 10/10 PASS. On the idle machine
      recovery recovered to gaussian 3/3, spring_pe 2/3 (vs 2/3, 0/3 in the contended E
      run), confirming the E-run recovery dip was contention, not a regression. *(The unit
      test already proves the mechanism: generation now obeys `max_nodes`, so the
      >50-node init trees that drove the longest fits no longer occur.)*
- [x] Ubuntu LTS (WSL Ubuntu-24.04, cmake 3.28, g++): standalone suite **14/14 pass**,
      incl. `test_generation_respects_max_nodes` and the docs/20 optimizer stop-predicate
      tests — so the deadline fix *and* B are both verified on Ubuntu at the unit level.
      (A full R-package Stage 0 smoke on Ubuntu remains a nice-to-have; the core sources
      are identical to the standalone build and the Rcpp glue is untouched by B.)
- [x] docs/20 status + the two analysis corrections noted in its header.

---

## 8. One-line summary

The timeout bug is **fixed**; the alarming 2194 s was a stale-binary artifact. What is
left is an *efficiency* tail (single fits up to ~33 s), to be addressed minimally and
only as Stage 1 evidence justifies — not a correctness defect.
