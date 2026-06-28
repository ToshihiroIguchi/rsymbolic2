# Timeout Overshoot: Root Cause and the Interruptible-Evaluation Fix

**Date:** 2026-06-14 (updated 2026-06-15)
**Status:** Phase 1 implemented (`7a10757`). **Phase 2 (the `safe_pow`/libm guard)
is withdrawn — its premise was wrong (see Update below).** The remaining overshoot was
an OpenMP thread-oversubscription bug plus external CPU contention, fixed/mitigated as
described in the Update.
**Supersedes the open question in:** docs/21 §5–6 (the "real residual issue").
**Depends on:** docs/20 (deadline propagation), docs/21 (timeout investigation).

---

## Update (2026-06-15): the actual root cause, and why Phase 2 is withdrawn

A measurement-driven re-investigation found the Phase 2 premise — "a value-dependent slow
`libm` path inside `safe_pow`" — to be **incorrect**. The real causes:

1. **A stale build masqueraded as a 6× overshoot.** Phase 1's core change is in the header
   `least_squares_problem.hpp`. R's package build does **not** track header dependencies, so
   `make` reported "Nothing to be done" and relinked stale `.o` files that predated Phase 1.
   The running binary therefore lacked the interruptible-evaluation fix. A forced clean
   rebuild (delete `src/*.o,*.dll`, reinstall) dropped `rel_mass`'s worst overshoot from
   ~1660 s to ~50 s. **Always clean-rebuild after editing a header.**

2. **The residual overshoot was OpenMP thread oversubscription, not a slow op.** The island
   loop used `#pragma omp parallel for` with the default team size (= core count, 12 here)
   for only `n = n_populations` (= 4) iterations. The 8 surplus threads busy-wait at the
   loop barrier and starve the island workers. Under that starvation, the **wall time** of a
   single ordinary node evaluation balloons to tens of seconds — instrumentation caught a
   single `std::exp(23.0)` taking **187 s** and a `std::log(5.07)` taking **67 s**, both with
   perfectly normal arguments (no libm value can be intrinsically that slow → the thread was
   simply not scheduled). The deadline poll runs on the starved worker, so it cannot fire.
   - **Proof:** `OMP_NUM_THREADS=4` (one thread per island) → 15/15 runs stop exactly at the
     budget. Single-threaded → no overshoot. The fix is to cap the team at the island count:
     `num_threads(min(n, omp_get_num_procs()))` on the island parallel-for
     (`evolutionary_search.cpp`). Islands are the only unit of parallelism, so this loses no
     concurrency.

3. **Any remaining overshoot is external CPU contention, not a code defect.** With the fix,
   on an **idle** machine the timeout is honoured exactly (`rel_mass` 300 s budget: runs land
   at 300.0 s, process CPU ≈ 895 s ⇒ cpu/wall ≈ 3.0). An overshooting run showed wall = 764 s
   but **total process CPU = 660 s (lower than a clean run's 895 s)** and cpu/wall = 0.86 —
   i.e. the process was *denied* CPU (a busy internal stall would *raise* CPU, not lower it).
   A cooperative in-thread deadline cannot beat external starvation; the benchmark-level
   **fail-fast** (abort the gate when a run exceeds `timeout × 1.5`) bounds the wasted time.

**Net fixes delivered:** clean-rebuild discipline; `num_threads` cap on the island loop
(`evolutionary_search.cpp`); benchmark fail-fast + `overshoot_sec` column
(`benchmarks/02_feynman_gate.R`). Phase 1 (interruptible residual/Jacobian) is retained and
correct. Phase 2 (§6 below) is **withdrawn**: there is no slow `safe_pow` path to guard.

The diagnostic detail in §§5–6 below is kept for the record but read it in light of this
update; the `cpu/wall` accounting is the decisive evidence and supersedes the "~94 ms/point
slow libm" reading in §1.

---

## 1. Executive summary

Feynman Stage 1 runs overshot their per-problem wall-clock budget grossly (individual
runs of 1000–4000 s against a 120–300 s budget). The investigation in docs/21 bounded
every *per-operation* suspect and concluded the residual issue was merely "a single
`fit()` up to ~33 s." That conclusion was **incomplete**.

**Proven root cause (2026-06-14):** the search deadline is polled only *between*
`minimizeOneStep` calls (`run_lm`, `eigen_lm_optimizer.cpp`). A single residual or
Jacobian evaluation — one closure call that loops over all `m = 1000` data points —
can take **~90+ seconds** for certain `pow`-heavy trees whose constants wander into an
extreme range during Levenberg–Marquardt iteration (a value-dependent slow `libm` path
inside nested `safe_pow = exp(y·log(x))`). Because that evaluation runs inside a single
`minimizeOneStep`, the deadline cannot interrupt it.

**Decisive measurement.** Instrumenting `EigenLMOptimizer::optimize` to time the wall
clock of the whole fit versus the wall time spent strictly inside the residual/Jacobian
closures (`func_secs`):

```
[LM] wall=1129.5s  func_secs=1129.5s  steps=3  func=9  jac=3  k=9  m=1000  nfev=9
[SLOW TREE] expr=((cos(tanh((-1.29431*x0)*0.481422)) ^ (0.879168 ^ ((-0.0891583*(x0*x0)) ^ 1.69071)))
                  * (0.0010357 * ((123.267+x0) + (((x0/-0.0829474)*(x0^x0)) ^ (0.615322-sqrt(x0))))))
```

`func_secs ≈ wall` ⇒ **100 % of the time is inside node evaluation**, not Eigen's linear
algebra and not thread blocking. With only 9 functor + 3 Jacobian calls, that is
**~94 s per single evaluation over 1000 points**. The tree has nested `pow`.

**Important caveat on the headline numbers.** The most extreme figures seen during the
investigation (e.g. 13 908 s, with `steps=1 func=4`) were *amplified by CPU contention
from the investigation harness itself* (concurrent monitor loops, a `Get-Counter`
sampler, frequent polling) oversubscribing the OpenMP worker threads. On an **idle**
machine the genuine pathology is ~1000–1650 s of overshoot, occurring in roughly 1 of
10–15 runs (it is timing-dependent: the wall-clock deadline changes how many generations
each island completes, which changes which trees are generated). The contention is not
the bug — the uninterruptible long evaluation is — but it explains why the raw numbers
varied so wildly and why the issue is intermittent.

Two independent fixes follow, in priority order (CLAUDE.md: correctness before
performance):

- **Phase 1 — correctness.** Make residual/Jacobian evaluation honour the deadline so the
  timeout contract holds to within a fraction of one evaluation, *regardless of which
  node operation is slow*. Op-agnostic; no vendored-Eigen changes.
- **Phase 2 — performance.** Eliminate the value-dependent slow evaluation so no single
  evaluation is pathologically expensive (so the search does not waste its budget and
  Phase 1's abort almost never fires). Requires a short diagnosis to pin the exact
  `safe_pow`/`exp`/`log` argument range, then an O(1) argument guard.

---

## 2. Evidence trail (so the conclusion is auditable)

All runs below are on Windows 11 / Rtools45 (MinGW, UCRT), the toolchain the R package
actually uses.

| Experiment | Result | Conclusion |
|---|---|---|
| coulomb/rel_mass **isolated** at 30/120/300 s budgets | every run stops at *exactly* the budget; worst single fit ≤ ~32 s | deadline mechanism is correct between steps; isolated runs do **not** overshoot |
| FTZ/DAZ subnormal micro-bench | subnormal penalty ~3× (mults), none for `exp`/`pow` | **not** a subnormal-flush issue |
| sleep/resume event log during the overshoot window | no Kernel-Power 42/107 events | **not** system suspend |
| R process WS across a 15-run sequence | flat ~76 MB | **not** a memory leak / paging |
| `simplify()` source review | one post-order pass, O(tree); no tree-growing rewrite | **not** a simplify blow-up |
| `sin/cos/exp/log/pow` micro-bench at extreme fixed args | ≤ 283 ns worst (`sin(1e300)`) | no *single fixed-arg* op explains ~94 ms/point |
| vendored Eigen `minimizeOneStep` review | inner loop bounded by `maxfev`; deadline polled only *between* steps | overshoot = one in-flight `minimizeOneStep` |
| `func_secs` vs `wall` split (clean machine) | `func_secs ≈ wall = 1129 s`, 12 evals | **100 % in node evaluation** → the slow work is in the residual/Jacobian closures |
| clean vs contended sequence | gaussian overshoots (13908 s) vanished clean; rel_mass still 1654 s clean | extreme numbers were contention-amplified; a genuine ~1000–1650 s pathology remains |

The reproduction harness (delete after Phase 2): `benchmarks/_debug_*.{R,cpp}` and the
`DEBUG-TIMEOUT`-marked instrumentation in `eigen_lm_optimizer.cpp` /
`evolutionary_search.cpp`.

---

## 3. Relevant code map

- `r-package/rsymbolic2/src/rsymbolic/expression/least_squares_problem.hpp`
  `make_least_squares_problem(...)` builds two `std::function` closures stored on the
  `OptimizationProblem`:
  - `residuals(params, r)` — loops `i = 0..m-1`, `r[i] = evaluate<double>(tree, X[i], params) - y[i]`.
  - `jacobian(params, jac)` — `k` passes, each looping `i = 0..m-1`, `evaluate<Dual>(...)`.
  These loops are where ~94 s is spent and where the deadline check must go.
- `r-package/rsymbolic2/src/eigen_lm_optimizer.cpp`
  - `fill_residuals(...)` calls `residuals(...)` then copies into Eigen's `fvec`
    (non-finite → `kLargeResidual = 1e10`).
  - `AnalyticFunctor::operator()` / `::df` wrap `fill_residuals` / `jacobian`.
  - `run_lm(...)` is `minimizeInit; do { minimizeOneStep } while(Running && !stop())`.
  - Eigen's `minimizeOneStep` returns `UserAsked` when `functor(...) < 0` or
    `functor.df(...) < 0` (vendored `NonLinearOptimization/LevenbergMarquardt.h:201,223,287`).
    **This is the interruption point Phase 1 uses — no Eigen edit required.**
- `r-package/rsymbolic2/src/rsymbolic/expression/dual.hpp`
  `pow(double)` and `pow(Dual)` implement `safe_pow`: `x>0 ⇒ exp(y·log(x))`,
  `x<0 ∧ y≈int ⇒ std::pow(x, round(y))`, else `0`. The Dual derivative adds
  `y·exp((y-1)·log(x))` and `p·log(x)`. This is the suspect for Phase 2.
- `r-package/rsymbolic2/src/rsymbolic/optimization/constant_optimizer.hpp`
  defines `ResidualFunction`, `JacobianFunction`, `StopRequested`, `OptimizationProblem`,
  `OptimizationResult`, `OptimizerConfig`.

---

## 4. Phase 0 — cleanup (do first, on a clean HEAD baseline)

1. Restore `eigen_lm_optimizer.cpp` and `evolutionary_search.cpp` to `HEAD` (removes all
   `DEBUG-TIMEOUT` instrumentation). The working-tree `M` on these two files is **only**
   that instrumentation (verified: `git diff` is +66/-0, all `DEBUG-TIMEOUT`); the
   session-start `M` was CRLF line-ending noise. There are **no** pre-existing content
   changes to preserve.
2. Delete debug artifacts: `benchmarks/_debug_*.{R,cpp,exe}` and
   `benchmarks/results/_debug_*.log`, `benchmarks/results/_stage1_*.log`.
3. Confirm `git status` is clean except intended new files (this doc).

---

## 5. Phase 1 — correctness: interruptible residual/Jacobian evaluation

**Goal.** No `symbolic_regression(..., timeout_seconds = T)` call exceeds `T` by more than
the time of a small fixed slice of one evaluation (target: ≤ a few seconds), on both
Windows and Ubuntu, with no change to search results when the deadline is not hit.

### 5.1 Design

Thread the search deadline (`StopRequested`) into the least-squares closures and check it
*inside* their per-point loops; on trigger, signal an abort that propagates to Eigen via a
negative functor return.

Recommended, minimally-invasive shape (the implementer may choose an equivalent clean
form, but must meet the constraints in §5.3):

1. `make_least_squares_problem(tree, X, y, initial, stop)` gains a trailing
   `StopRequested stop = {}` parameter (a `std::function<bool()>`; empty ⇒ never aborts).
2. Add an abort signal to `OptimizationProblem`, e.g. a
   `std::shared_ptr<std::atomic<bool>> aborted` (created in the factory, captured by the
   closures, and readable by the optimizer). Alternatively change `ResidualFunction` /
   `JacobianFunction` to return `int` (`0` ok, `<0` aborted) — pick whichever keeps the
   call sites cleanest; the shared-flag form avoids touching the function typedefs.
3. In the residual closure: every `kStride = 256` points, if `stop && stop()`, set
   `aborted = true`, fill the remaining `r[i..]` with `kLargeResidual`, and return. Same
   pattern in each Jacobian pass.
4. In `AnalyticFunctor::operator()` / `::df` (and the Numeric variants): after the closure
   call, if `aborted` is set, return `-1`. Eigen's `minimizeOneStep` then returns
   `UserAsked`; `run_lm`'s loop exits.
5. `fit()` (in `evolutionary_search.cpp`) already holds the `stop_requested` predicate and
   already calls `make_least_squares_problem`; pass `stop_requested` through to it.

### 5.2 Why the reported loss/constants stay correct on abort

In Eigen's `minimizeOneStep`, the iterate `x` is updated **only** on a successful step,
and `fvec` holds the residual at the last accepted `x`. An abort returns before any
update, so `result.constants` = last accepted iterate and `result.loss = fvec.norm²` =
its loss. If the abort happens on the very first evaluation (`minimizeInit`), `fvec` is the
partially-filled vector whose remaining entries we set to `kLargeResidual`, giving a
defined, large loss — i.e. the candidate is correctly treated as a poor fit and discarded
by selection. Either way the result is finite and selection-safe; an interrupted fit is
never a winning candidate.

### 5.3 Constraints / invariants (must hold)

- **No behavioural change when the deadline is not reached.** With `stop` empty (or never
  true), the closures and the optimizer must be byte-for-byte equivalent to today
  (the `stop()` check is the only added work, guarded by the stride counter).
- **Serial fallback intact.** Works with and without OpenMP. The `stop` closure already
  reads only `t_start` + wall clock and is called from worker threads today.
- **No vendored-Eigen edits.** Use only the `functor(...) < 0 ⇒ UserAsked` contract.
- **`kStride` is a compile-time constant** (256) — not a new public tuning knob.
- Default optimizer is `EigenLM` with an analytic Jacobian; the analytic path is the one
  that must be interruptible. Keep the `NumericFunctor` path consistent.

### 5.4 Tests (Phase 1)

Add to the standalone suite (`standalone/tests/…`, run via cmake on both OSes):

1. `test_optimizer_honours_stop`: build a least-squares problem; pass a `stop` predicate
   that returns `true` after the first invocation; assert `optimize()` returns promptly
   with `nfev` far below `maxfev`, `success == true`, and a finite loss.
2. `test_optimizer_no_stop_is_unchanged`: with an empty `stop`, assert identical result
   (constants, loss, nfev) to a reference run without the new parameter — pins §5.3's
   no-behavioural-change invariant.
3. Keep existing optimizer tests green (Eigen LM known-answer fits).

### 5.5 Acceptance (Phase 1)

- Standalone suite passes on Windows and Ubuntu (expected counts per
  `reference-r-test-commands`: standalone 14/14 +2 new; testthat `FAIL 0`).
- A clean (idle, no concurrent monitors) 3-problem × 5-seed sequence
  (`gaussian, rel_mass, coulomb`, `timeout=120`) shows **every run ≤ ~125 s** (budget +
  one short eval slice). Compare against the pre-fix clean baseline where `rel_mass seed2`
  overshot to 1654 s.
- Nguyen gate no-regression (recovery unchanged vs `nguyen_gate_20260613.csv`).

---

## 6. Phase 2 — performance: remove the value-dependent slow evaluation

**Goal.** No single residual/Jacobian evaluation is pathologically slow, so the search
spends its full budget exploring (and Phase 1's abort is a safety net, not a routine
event).

### 6.1 Diagnosis (required first — do not guess the guard)

The fixed-argument micro-benchmarks did **not** reproduce ~94 ms/point, so the slow path
depends on a specific constant configuration reached mid-fit. Capture it:

1. Temporary instrumentation: in `AnalyticFunctor::operator()` (and `df`), time each call;
   when one call exceeds ~2 s, dump `params` (the `k` constants) and the data key.
2. Reproduce on an **idle** machine (no concurrent monitors — that contention confounds
   the timing) until one slow call is captured.
3. Replay deterministically in a minimal standalone (single-threaded, no R, no OpenMP):
   rebuild that exact tree + captured constants, evaluate over the dataset, and **bisect**
   which node/operator and which argument range consumes the time (instrument
   `safe_pow`/`exp`/`log` call counts and per-call timing).

### 6.2 Likely fix (confirm against 6.1 before coding)

Most probable: nested `safe_pow` drives `y·log(x)` (or the derivative's
`exp((y-1)·log(x))`) into a range where `std::exp`/`std::pow` on this `libm` takes a slow
high-precision path. Guard the argument so the result saturates in O(1):

- In `pow(double)` and `pow(Dual)`: compute `t = y·log(x)` for the `x>0` branch; if
  `t > kPowExpMax` (≈ 709, `exp` overflow) return `+inf`/clamp; if `t < kPowExpMin`
  (≈ −745, underflow to 0) return `0` — **without** calling `std::exp`. Apply the
  symmetric guard in the derivative terms.
- If 6.1 instead implicates the `x<0, y≈int` branch (`std::pow(neg, round(y))` with a
  huge integer exponent), guard `|round(y)|` to a sane cap and return `0`/clamped beyond
  it.

The guard must preserve `safe_pow` semantics in the normal range and keep the value path
and the AD path in agreement (they share the branch logic by design — dual.hpp §"so the
value path and the AD path always agree").

### 6.3 Tests (Phase 2)

1. Finite-difference AD check: for the guarded `pow`, verify the Dual derivative matches a
   central finite difference across normal and near-threshold arguments (this is the
   project's standing AD-correctness requirement).
2. Regression: the captured pathological (tree, params) now evaluates in O(ms), asserted
   by a standalone timing test with a generous ceiling.
3. Nguyen gate no-regression (recovery + timing).
4. `safe_pow` semantics tests still green (x>0, x=0, x<0 integer/non-integer).

### 6.4 Acceptance (Phase 2)

- Captured pathological evaluation drops from ~90 s to ≤ ~1 s (standalone).
- Clean Stage-1-style sequence: no run relies on Phase 1's abort (all fits finish well
  inside budget); recovery not reduced.
- Both OS green; AD finite-difference check passes.

---

## 7. Verification protocol (both phases)

Per CLAUDE.md Platform Constraints, "done" = builds and tests pass on **Windows and
Ubuntu**.

- **Windows (Rtools45):** `install.packages('r-package/rsymbolic2', repos=NULL, type='source')`;
  standalone cmake suite; `testthat` via `test_dir(...)` (PowerShell + full-path Rscript —
  see `reference-r-test-commands`).
- **Ubuntu (WSL Ubuntu-24.04):** copy sources to `/tmp`, strip `.o`/`.so`, build the
  standalone cmake suite and (optionally) the R package; same expected pass counts.
- **Overshoot check:** the clean 3-problem × 5-seed sequence at `timeout=120`, idle
  machine, no concurrent monitors (the monitors are what perturbed earlier runs).
- Report medians/spreads where recovery is measured; never weaken thresholds.

---

## 8. Commit / upload plan

- Phase 0 + Phase 1 as one commit (cleanup + the correctness fix + tests + this doc),
  message describing the root cause and the interruptible-evaluation mechanism.
- Phase 2 as a separate commit (guard + tests) once 6.1 pins the operator.
- Push only after both-OS green. (CLAUDE.md: confirm before pushing — the user has
  authorised upload for this task.)
- Update docs/21 header to point here (its §5–6 "real residual issue" is now resolved),
  and docs/20 if its overshoot-bound wording needs the correction.

---

## 9. One-line summary

The timeout could not interrupt a single residual/Jacobian evaluation, and a `pow`-heavy
tree can make one evaluation take ~90 s; Phase 1 makes evaluation interruptible (correct
timeout, op-agnostic), Phase 2 removes the slow evaluation itself (performance). The
4000 s+ headline numbers were contention-amplified; the genuine pathology is ~1000–1650 s
and intermittent.
