# Constant-Optimizer Deadline Propagation (timeout fix)

**Date:** 2026-06-09
**Status:** **Implemented** (commit `6ba5791`, 2026-06-13). Verified on Windows; Ubuntu
pending. Post-implementation investigation and the residual-overshoot follow-up are in
**docs/21**.

> **Two corrections to the analysis below, found during implementation (see docs/21):**
> 1. §1 says EigenLM runs "with no analytic Jacobian, NumericalDiff multiplies that by
>    `k`." That is wrong: `make_least_squares_problem` **does** supply an analytic
>    Jacobian (dual numbers), so the `AnalyticFunctor` branch runs. The factor of `k` is
>    real but comes from the analytic Jacobian's `k` forward passes, not NumericalDiff.
> 2. §2's overshoot table understates the Eigen LM worst case. One `minimizeOneStep` is
>    **not** "one Jacobian + one solve": its inner trust-region loop (`while ratio<1e-4`)
>    can run up to `maxfev` residual evaluations before returning. The poll fires only
>    *between* `minimizeOneStep` calls, so the true per-step overshoot bound is one
>    Jacobian + up to `maxfev` inner evals ≈ a few seconds, not one solve. Still bounded,
>    still acceptable; quantified in docs/21.

---

## 1. Problem

`symbolic_regression(..., timeout_seconds = T)` does not respect `T`. In the Feynman
Stage 0 smoke run (docs/19), with `timeout_seconds = 60`, problem `interference`
seed 2 ran for **2404 seconds** — 40× over budget. Other runs also overshot (77–195 s
against a 60 s cap).

**Root cause.** The deadline is checked only at *evolution-step boundaries*
(`evolutionary_search.cpp:170` and `:178`, inside `evolve_island`). But each step
calls `fit()` → `ConstantOptimizer::optimize()`, and that call is opaque to the
deadline.

The active backend is **EigenLM** — `SearchOptions::optimizer_type` defaults to
`OptimizerType::EigenLM` (`evolutionary_search.hpp:23`), so the 2404 s came from
`EigenLMOptimizer`, **not** RandomRestart. `EigenLMOptimizer` runs Levenberg–Marquardt
up to `maxfev = max_iterations·(k+1)` function evaluations
(`eigen_lm_optimizer.cpp:139`, `max_iterations = 100`). With no analytic Jacobian,
`NumericalDiff` multiplies that by another factor of `k`. The whole `lm.minimize(x)`
is **one opaque call** with no time check. On a bloated tree (here,
`interference = I1+I2+2·sqrt(I1·I2)·cos(delta)`) with a large constant count `k`, one
such `optimize()` call dominated wall-clock time and could not be interrupted.

`RandomRestartOptimizer` (the only other backend) has the same structural gap — its
`n_restarts = 4` × `max_iterations = 100` loop is un-time-checked — but it is **not the
default and was not in the path that produced 2404 s**. It is fixed here only for
interface consistency (§4.1), not because the current benchmark exercises it.

A second, related gap: **`initialize_island` (`evolutionary_search.cpp:126`) calls
`fit()` `population_size` times with no deadline at all.** Island initialisation can
therefore overshoot before the generation loop (which does check) is even entered.

This is a **correctness** problem (CLAUDE.md priority 1), not performance: a
user-supplied `timeout_seconds` is an API contract. It also blocks the Feynman
benchmark — Stage 1 (5 min × 25 eq × 5 runs) would balloon if a single problem can
consume 40 min.

---

## 2. Design

Propagate a lightweight **stop predicate** from the search loop, which owns the
deadline, down into every optimizer. The optimizer polls it at its own coarse
boundaries and returns the best result found so far when it fires.

```cpp
// constant_optimizer.hpp
using StopRequested = std::function<bool()>;
```

The search loop builds the closure once per deadline context:

```cpp
const auto stop = [&]{ return has_deadline && elapsed_sec(t_start) >= deadline_sec; };
```

### Why a callback, not a time value

- **Keeps `<chrono>` out of the optimizer interface.** The optimizer never learns
  *how* "stop" is decided — wall-clock, evaluation budget, or a test stub. The search
  loop already owns `t_start`/`deadline_sec`; it captures them in the closure.
- **Testable.** Unit tests pass `[]{ return true; }` (stop immediately) or a counter
  that trips after N polls, with no clock dependency or sleeps.
- **Backend-agnostic.** The same predicate works for RandomRestart, Eigen LM, and any
  future Ceres backend.

### Rejected alternatives

- *`deadline_sec` field on `OptimizationProblem`.* Leaks timing semantics and a clock
  type into a data struct that is otherwise a pure math problem; harder to unit-test.
- *Time budget in `OptimizerConfig`.* `OptimizerConfig` is fixed at optimizer
  construction (per island), but the *remaining* budget shrinks every step. Wrong
  lifetime.
- *Lower `max_iterations` / `n_restarts` only (the "simple" fix).* Reduces but does not
  bound overshoot: one bloated-tree `minimize()` with a large `k` still blows the
  budget. Does not honour `timeout_seconds`. Rejected as incomplete.

### Overshoot bound after the fix

The deadline cannot interrupt work *between* poll points, so the residual overshoot is:

| Backend | Poll granularity | Worst-case overshoot past deadline |
|---------|-----------------|-----------------------------------|
| RandomRestart | top of each restart and each inner iteration | one residual evaluation (one forward pass over the data) |
| Eigen LM | after each `minimizeOneStep` | one LM step = one Jacobian build (`k+1` residual evals for NumericalDiff) + one linear solve |

Both are bounded by a single cheap unit of work, versus the current unbounded
`minimize()` / full restart sweep. This is the same "overshoots to one fit() call"
guarantee the step-loop comment already promises (`evolutionary_search.cpp:174-177`) —
we are extending it one level deeper, into `fit()` itself.

---

## 3. Interface changes

`constant_optimizer.hpp`:

```cpp
class ConstantOptimizer {
public:
    virtual ~ConstantOptimizer() = default;

    // Stop-aware optimize. Implementations poll stop_requested() at their own
    // boundaries; when it returns true they stop early and return the best result
    // found so far (success reflects whether that result is finite).
    virtual OptimizationResult optimize(const OptimizationProblem& problem,
                                        const StopRequested& stop_requested) const = 0;

    // Convenience overload: no deadline (used by tests and any caller that does not
    // impose a time limit). Non-virtual; forwards with an always-false predicate.
    OptimizationResult optimize(const OptimizationProblem& problem) const {
        return optimize(problem, [] { return false; });
    }

    virtual std::string name() const = 0;
};
```

The non-virtual overload preserves every existing call site that passes no predicate
(including unit tests), so the change is additive at those sites.

---

## 4. Per-backend changes

### 4.1 RandomRestartOptimizer (`random_restart_optimizer.cpp`)

**Not the default backend** (EigenLM is), so this is not on the path that caused the
2404 s overshoot. It is updated because the interface change in §3 is `virtual` — every
backend must match the new signature — and because `optimizer_type` is a user-selectable
API knob, so `timeout_seconds` must be honoured regardless of which backend is chosen.
The poll itself is a few lines; this is consistency work, not scope expansion.

Add a poll at the top of the restart loop and the inner iteration loop. On stop,
break out and let the function return the best-so-far `result` (already tracked in
`result.loss`/`result.constants`).

```cpp
for (std::size_t restart = 0; restart < n_restarts; ++restart) {
    if (stop_requested()) break;
    ...
    for (std::size_t it = 0; it < config_.max_iterations; ++it) {
        if (stop_requested()) break;
        ...
    }
}
```

`result` is already initialised to `x0` with `loss = +Inf` and updated on every
improvement, so an early stop after zero completed restarts still returns a valid
(if poor) result — exactly the existing contract.

### 4.2 EigenLMOptimizer (`eigen_lm_optimizer.cpp`)

Replace the opaque `lm.minimize(x)` with the explicit init + step loop that Eigen's
`LevenbergMarquardt` exposes, polling between steps:

```cpp
using Status = Eigen::LevenbergMarquardtSpace::Status;
Status status = lm.minimizeInit(x);
if (status == Status::ImproperInputParameters) { finalize(lm, status, x, result); return result; }
do {
    status = lm.minimizeOneStep(x);
} while (status == Status::Running && !stop_requested());
finalize(lm, status, x, result);
```

This is applied to both the `AnalyticFunctor` and `NumericFunctor` branches.

**Equivalence — confirmed in source, not assumed.** In the vendored Eigen actually
included here (`unsupported/Eigen/NonLinearOptimization`, via RcppEigen),
`LevenbergMarquardt::minimize()` is literally (`LevenbergMarquardt.h:157-166`):

```cpp
status = minimizeInit(x);
if (status == ImproperInputParameters) return status;
do { status = minimizeOneStep(x); } while (status == Running);
return status;
```

`minimizeInit` and `minimizeOneStep` are public (`:85-86`). The decomposed form is
therefore the **same code path** as `minimize()`, not a re-implementation: with
`stop_requested()` always false it is bit-identical by construction. The Nguyen
no-regression run (§6) is a belt-and-braces check, not the sole guarantee. No version
fallback is needed; the prior concern is resolved.

### 4.3 Header signatures

`eigen_lm_optimizer.hpp` and `random_restart_optimizer.hpp` override declarations
updated to the new `optimize(problem, stop_requested)` signature.

---

## 5. Search-loop wiring (`evolutionary_search.cpp`)

1. **`fit()` gains a stop parameter** and forwards it:

   ```cpp
   double fit(Tree& tree, const std::vector<std::vector<double>>& X,
              const std::vector<double>& y, const ConstantOptimizer& optimizer,
              const StopRequested& stop_requested) {
       ...
       const OptimizationResult result = optimizer.optimize(problem, stop_requested);
       ...
   }
   ```

2. **`initialize_island` gains the deadline** (currently has none) and passes the
   stop predicate into each `fit()`. It also breaks out of the population-fill loop
   when stop fires, leaving a partially-initialised-but-valid island (members already
   created keep their fitted losses; the loop simply stops adding more). *Decision:*
   an island that runs out of time during init is acceptable — it just starts smaller;
   correctness of the search is unaffected.

   **Empty-population guard (required).** The downstream code assumes a non-empty
   population: `tournament_best`/`tournament_worst` call `sample_index(pop.size(), …)`,
   which computes `uniform_int_distribution(0, size-1)` and is **undefined for
   `size == 0`**. If the deadline fires before the very first member is added, the
   island would be empty and the subsequent generation loop would read out of bounds.
   Mitigation: guarantee at least one member by checking the stop predicate at the
   *top* of the fill loop but only after `i > 0` (always create member 0), **or** skip
   `evolve_island` entirely for an island whose population ended up empty. The plan
   takes the first option — always admit the initial member — so every island that
   exists is valid. This is a real crash risk the first draft missed; it must be in the
   implementation and covered by a test (init with an immediately-true predicate yields
   a 1-member island, not a crash).

3. **`evolve_island`** already owns `t_start`/`has_deadline`/`deadline_sec`. It builds
   the `stop` closure once and passes it to every `fit()` call. The existing
   step-boundary checks (`:170`, `:178`) remain as the coarse outer guard.

4. The deadline closure threads through any other `fit()` call sites (e.g. children
   evaluated during a generation).

No change to the OpenMP structure: each island already runs `evolve_island`
independently, and `stop` reads only `t_start` (shared, immutable after start) and the
wall clock — both safe to read concurrently.

---

## 6. Tests and verification

**Unit (new), `tests/` for the optimizers:**
- `optimize(problem, []{return true;})` returns immediately with `result.constants == x0`
  (or the single evaluated point) and finite-or-Inf loss, never a crash.
- A predicate that trips after N polls stops within a bounded number of
  evaluations (assert `result.evaluations` is small).
- With an always-false predicate, results are **identical** to the current
  `optimize(problem)` on a fixed known-answer problem (regression guard for the Eigen
  decomposition).

**Integration:**
- **Nguyen no-regression** (`benchmarks/01_nguyen_gate.R`): recovery rate and per-seed
  losses unchanged vs. `benchmarks/results/nguyen_gate_20260608.csv`. This is the
  primary guard that the Eigen LM decomposition is behaviour-preserving.
- **Feynman Stage 0 re-run** (`benchmarks/02_feynman_gate.R`): every run's
  `elapsed_sec` ≤ `timeout_seconds` + one-step slack (target: no run exceeds ~75 s for
  a 60 s cap; the 2404 s case must not recur). Record the new max overshoot.

**Both OSes** (CLAUDE.md Platform Constraints): build and run the unit tests and the
Nguyen gate on Windows 11 (Rtools45 MinGW/UCRT) and Ubuntu LTS before marking done.

---

## 7. Risks and trade-offs

- **Eigen decomposition behaviour drift.** *Resolved at planning time* (§4.2): the
  decomposed loop is the same code path as `minimize()` in the vendored Eigen source,
  so drift is not possible with an always-false predicate. Nguyen no-regression remains
  as a confirmation. (Original fallback — tighter `maxfev` — no longer needed.)
- **Poll cost.** One `std::function` call per restart-iteration / per LM step. Negligible
  against a residual evaluation over 1000 data points.
- **Partial island init.** An init-time timeout yields a smaller island. Acceptable for
  a development gate; for Stage 2 the per-problem budget (10 min) is large enough that
  init should always complete, so this path should not trigger there.
- **Overshoot is bounded, not zero.** One LM step on a huge tree with large `k` can
  still be a few seconds. This is inherent to not interrupting a linear solve mid-flight
  and is acceptable; the pathological 40× case is eliminated.

---

## 8. Acceptance criteria

- [ ] `ConstantOptimizer::optimize(problem, stop_requested)` added; no-deadline overload
      preserves existing call sites.
- [ ] RandomRestart and Eigen LM poll the predicate; Eigen LM uses the init/step loop.
- [ ] `fit()` and `initialize_island` thread the deadline through; `evolve_island`
      builds and passes the closure.
- [ ] New optimizer unit tests pass (immediate-stop, bounded-stop, identical-when-never-stop).
- [ ] Nguyen gate: no regression vs. `nguyen_gate_20260608.csv`.
- [ ] Feynman Stage 0: every run within `timeout_seconds` + one-step slack; 2404 s case
      gone. Recorded in docs/19.
- [ ] Verified on Windows 11 and Ubuntu LTS.
