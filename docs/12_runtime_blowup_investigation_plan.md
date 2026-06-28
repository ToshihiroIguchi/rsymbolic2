# Implementation Plan: Bound and Diagnose Runtime Blowup

**Date:** 2026-06-06
**Trigger:** Phase 2 gate benchmark (`docs/11`) passed, but some seeds ran 20–96×
the median (N4 s5: 2966s; N9 s5: DNF at ~96 min). All affected runs still *recovered* —
this is a wall-time problem, not a correctness problem.

## Guiding principle

Do **not** pre-commit to a cause. The earlier draft asserted "bloat" with no
measurement; that was intuition stated as fact. This plan first makes runtime
**bounded** (valuable regardless of cause) and **observable** (turns intuition into
evidence), then chooses the real fix by a pre-registered rule. Adaptive parsimony is
a *candidate* fix, not a decided one.

Competing hypotheses for the blowup (to be discriminated by the logs in Step 2):
- **H1 Bloat:** populations fill with large trees; per-generation cost grows.
- **H2 LM ill-conditioning:** high-degree polynomial bases (N4 = x⁶+…+x over [-1,1])
  are near-collinear → ill-conditioned normal equations → many LM iterations.
- **H3 Constant-count cost:** Jacobian is k forward passes (k × m × tree_size); a few
  trees with many constants dominate, independent of overall tree size.

---

## Step 1 — Per-run wall-clock timeout (bounded runtime)

**Goal:** No single `sr_fit()` call can run unbounded. Default off (0) to preserve
current deterministic behavior and existing tests.

### 1a. `SearchOptions` (`evolutionary_search.hpp`)
Add:
```cpp
double timeout_seconds = 0.0;  // 0 = no limit; otherwise stop search after this wall time
int    verbosity       = 0;    // 0 = silent; >=1 = per-epoch diagnostics to stderr (Step 2)
```

### 1b. `run_evolution` (`evolutionary_search.cpp`)
- `#include <chrono>`.
- Before the epoch loop: capture `const auto t_start = std::chrono::steady_clock::now();`
  and compute a deadline when `timeout_seconds > 0`.
- Pass the deadline (and a "has deadline" flag) into `evolve_island` so each island
  stops at **generation granularity** — checked at the top of the generation loop,
  before the population sweep.
- The outer epoch loop also stops launching new epochs once the deadline has passed
  (extend the existing `reached_target` break with a `timed_out` condition).

**Thread-safety:** the deadline check inside `evolve_island` only *reads*
`steady_clock::now()`; each island returns from its own invocation; no shared state is
written. Safe under the existing `#pragma omp parallel for`. No atomics needed.

**Determinism note (document in roxygen):** a run that *completes within budget* stays
bit-reproducible for a fixed seed. A run whose timeout *fires* is wall-clock dependent
and therefore not reproducible — this is inherent and acceptable. Default 0 keeps the
existing reproducibility test (`test-recovery.R`) deterministic.

---

## Step 2 — Per-epoch instrumentation (evidence)

**Goal:** Discriminate H1/H2/H3 by observing how population shape evolves on a slow run.

### `run_evolution`, serial section after `migrate()`, gated by `verbosity >= 1`
Compute across **all islands' populations** combined:
- global best loss so far,
- median and max tree size (`PopMember.complexity`),
- median and max constant count (`count_constants(m.tree)`),
- elapsed seconds since `t_start`.

Emit one line per epoch to `std::cerr` (portable; appears in the R console's stderr;
matches the roadmap's "progress logging to stderr"). Keep the core free of any R
dependency. Cost is O(total population) per epoch — negligible.

Example line:
```
[epoch 7  t=41.2s] best=3.1e-06  size med/max=12/47  nconst med/max=3/14
```

**What each hypothesis predicts:**
- H1 true → `size med` climbs steadily across epochs while `best` stalls.
- H2 true → `size`/`nconst` stay modest but each epoch is slow; concentrated on N4-type
  high-degree polynomials (cross-reference per-problem timing).
- H3 true → `size med` modest but `nconst max` large; the slow runs are those with a
  high `nconst max`.

---

## Step 3 — R bridge + wrapper

### 3a. `rsymbolic2_r.cpp`
Add `double timeout_seconds` and `int verbosity` parameters to
`symbolic_regression_cpp(...)`; wire to `opts.timeout_seconds` / `opts.verbosity`.

### 3b. `symbolic_regression.R`
Add `timeout_seconds = 0` and `verbosity = 0L` arguments; pass through; document both
in roxygen (including the determinism caveat for `timeout_seconds`).

### 3c. Regenerate bindings
`Rcpp::compileAttributes()` to refresh `RcppExports.cpp` / `RcppExports.R`;
`roxygen2::roxygenise()` (or `devtools::document()`) for the man page.

---

## Step 4 — Re-run the pathological cases with diagnostics

New script `benchmarks/02_blowup_probe.R`:
- Run **N9 seed 5** and **N10 (all 5 seeds)** — the only unfinished gate cells —
  with `verbosity = 1` and `timeout_seconds = 300`.
- Tee per-epoch stderr logs to `benchmarks/results/blowup_probe_YYYYMMDD.log`.
- Append the completed rows to `nguyen_gate_20260606.csv` (or write a companion CSV),
  finishing the gate table.

This single run both **completes the gate table** and **collects the H1/H2/H3 evidence
on the exact slow case** — no separate diagnostic run needed.

---

## Step 5 — Decide the real fix (pre-registered rule)

Read the Step 4 logs and apply:

| Observation | Conclusion | Fix to implement next |
|-------------|-----------|------------------------|
| `size med` grows monotonically, `best` stalls | H1 bloat | Adaptive parsimony (Phase 2.7) and/or tighter `max_nodes` |
| size/nconst modest, slow epochs, N4-class only | H2 conditioning | Cap LM `max_iterations`; investigate Jacobian conditioning / basis scaling |
| `nconst max` large on slow runs only | H3 const-count | Cap constants per tree; skip LM above a constant-count threshold |

If two hold simultaneously, address the one the logs show contributing most wall-time.
Record the decision and evidence in this file before implementing the fix.

---

## Testing (Step 1–3 changes)

`tests/testthat/test-timeout.R` (new):
- A run with a hard problem and `timeout_seconds = 1` returns within a few seconds
  (assert `elapsed < ~5s`) and still returns a valid result list.
- `timeout_seconds = 0` (default) is unchanged.

Regression guards (must still pass):
- `test-recovery.R` "same seed gives identical result" — confirms default path stays
  deterministic (timeout off).
- All existing `test-output.R` / `test-validation.R` / `test-island.R`.

---

## Scope boundary

In scope: timeout, instrumentation, R wiring, the probe run, the decision rule.
**Out of scope (deliberately):** implementing adaptive parsimony or any blowup *fix* —
that is Step 5's outcome, chosen by evidence, and planned separately once the logs exist.
This avoids building a fix for a cause we have not yet confirmed.

---

## RESULTS (2026-06-06) — Steps 4 & 5 executed

Probe run: `benchmarks/02_blowup_probe.R`, N9 s5 + N10 s1–s5, timeout 300s, verbosity 1.
Data: `benchmarks/results/blowup_probe_20260606.{csv,log}`.

### Evidence (epoch-0 snapshot: time vs population shape)

| run    | elapsed | size med/max | nconst med/max | recovered | shape |
|--------|---------|--------------|----------------|-----------|-------|
| N10 s1 | 36s     | 23/51        | 6/18           | yes       | compact correct form |
| N10 s5 | 14s     | 23/51        | 5/15           | yes       | compact correct form |
| N10 s2 | 29s     | 29/50        | 7/18           | yes       | |
| N10 s4 | 73s     | 42/51        | 10/14          | yes       | bloated |
| N10 s3 | 493s    | 41/51        | 10/15          | yes       | bloated |
| N9  s5 | 1127s   | 45/51        | 11/17          | yes       | heavily bloated |

Wall time tracks `size med` and `nconst med` tightly. Fast runs (14–36s) have size
~23 and found the compact true form (e.g. N10 s5: `-2*cos(x1)*sin(x0)`). Slow runs
(73–1127s) drift to the `max_nodes=50` cap (size 41–45) and carry ~10–11 constants.

### Verdict: **H1 (bloat), dominant. H3 is its mechanical consequence. H2 rejected.**

- **H2 rejected here:** N9/N10 are trigonometric, not polynomial, yet they blow up —
  so the cause is not high-degree-polynomial LM conditioning. (N4 earlier may have had
  a conditioning component, but it is not the general mechanism.)
- **H1 confirmed:** runs that fail to find a compact form early let trees grow to the
  node cap; population-wide bloat makes every generation expensive.
- **H3 is downstream of H1:** larger trees carry more constants, so each LM solve costs
  more (Jacobian = k forward passes). Not an independent cause.

**Decided fix (next work):** adaptive parsimony (Phase 2.7) — a complexity penalty in
the selection loss so the search prefers compact expressions and does not drift to the
node cap. Secondary lever: a lower default `max_nodes`. To be planned and implemented
separately; this document's scope ends at the decision.

### Defect found: timeout granularity (FIXED 2026-06-06)

The probe also revealed that the just-added timeout overshot badly: N9 s5 ran 1127s
and N10 s3 ran 493s against a 300s limit. Cause: the deadline was checked only at
**generation** boundaries, but one bloated generation (500 steps × LM on large trees)
can take >1000s, so the check could not fire until the generation finished. Fixed by
also checking the deadline inside the per-step loop, bounding overshoot to one `fit()`
call. (`evolve_island`, `evolutionary_search.cpp`.)
