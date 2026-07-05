# 49. Evaluation accounting and the duplicate-evaluation cache (eval_cache): Feynman wall-clock screen

**Date:** 2026-07-06
**Status:** COMPLETE. **Verdict: `eval_cache` default stays OFF.** The pre-registered
bar required all three criteria; criterion (2) failed. (1) median per-problem wall
change **-9.71%** (bar: <= -5%) — MET. (2) no problem > 2% slower — **NOT MET**:
driven_osc +6.8%, heat_conduct +2.9%, torque +2.8%. (3) all non-timeout expressions
identical between arms — MET (25/25). The option remains available as an opt-in speed
knob (measured overall wall -15.8%, hit rate 0.19-0.34); the default is unchanged.

This documents two related changes and the measurement that decides the cache's
default:

- **Evaluation accounting** (commit `99cdd3c`, unconditional, always on).
- **Duplicate-evaluation cache** (commit `6d0d683`, opt-in `eval_cache`, default OFF).

Both are implementation-method changes under CLAUDE.md's allowed divergences
(behaviour-neutral evaluation accounting and caching): they change *how* a result is
computed, never *which* settings define the search, and are permitted only while the
returned values and the search trajectory remain bit-identical with the mechanism on
or off. Catalogued in `docs/29` §A #15-16.

## 1. Design summary

### 1.1 Evaluation accounting (99cdd3c)

Per-island counters, maintained unconditionally and summed into `SearchResult`:

- **Counting unit** = `max_evals` units: `n_evals = n_forward_evals + n_lm_resid_evals`
  (forward-pass loss evaluations plus the residual evaluations consumed by
  constant-optimisation fits). This holds by construction.
- **LM Jacobian builds** (`lm_jac`) are reported for honest compute accounting but
  **never charged** to `n_evals` or the `max_evals` budget (unchanged budget
  semantics).
- Surfaced as `n_evals` + the `eval_counts` breakdown in the R result list and the
  Python result object. Counting is integer-only and RNG/float-free, so the search
  trajectory is bit-identical with or without it. Motivation: Operon-style honest
  compute accounting for fair budget comparisons against PySR.

### 1.2 Duplicate-evaluation cache (6d0d683, `eval_cache`, default OFF)

A memoisation of the full-data forward-pass loss (`sse_current`), which is a pure
function of (tree, dataset):

- **Structure:** per-island, fixed **1024-slot direct-mapped** table
  (`kEvalCacheSlots`, `evolutionary_search.cpp`; `eval_cache.hpp`). Slot =
  `key & (slots-1)`; `store()` overwrites unconditionally (no probing, no eviction
  policy, no growth). Island-local, share-nothing; a few hundred KB per island.
- **Key:** strict FNV-1a 64-bit hash over the postfix tree's evaluation-relevant
  fields (node kind; constant slot index AND value **bit pattern**; variable index;
  operator ids). A hit additionally requires **field-wise tree equality**
  (`tree_eval_equal`, constants compared bitwise so NaN-valued constants match
  themselves), so a slot collision or even a 64-bit hash collision is reported as a
  miss and re-evaluated — collisions can only lower the hit rate, never change a
  returned value.
- **Hit-charging rule:** a cache hit is charged to `n_evals`/`n_forward_evals`
  exactly like a real evaluation, so even `max_evals`-budgeted runs are bit-identical
  ON vs OFF. The `cache_hits`/`cache_misses` counters exist to disambiguate real
  compute: `hits + misses` is the number of forward passes routed through the cache,
  and `misses` is the real evaluation work done.
- **Scope:** only full-data `sse_current` calls through `score_sse` are memoised.
  Inactive under `batching = TRUE` (each iteration's random subsample makes a cached
  SSE unreusable). LM residual/Jacobian evaluations are not cached (constants change
  every step). OFF means no table is allocated and the code path is byte-identical.
- Standalone gates at commit time: ON-vs-OFF result identity, 1-vs-4-thread
  determinism with the cache ON.

## 2. Pre-registered decision criteria

Fixed **before** the measurement (and not adjusted after): flip the default to ON iff
**all** of

1. median per-problem wall-clock reduction >= 5% on the Stage-1 set,
2. no problem > 2% slower,
3. all non-timeout expressions identical between arms.

Otherwise the default stays OFF and the measured hit rate is recorded as the reason
the option still exists.

## 3. Setup

- **Arms.** `benchmarks/02_feynman_gate.R stage=1 runs=1` (OFF) vs
  `stage=1 runs=1 cache` (ON; the new `cache` flag passes `eval_cache = TRUE` and
  changes nothing else — see the flag comment in the script). 25 dev-set problems x
  1 seed, 2800 generations, 300 s per-run timeout, PySR-parity settings pinned in the
  script. Arms run strictly sequentially (never concurrently), each under a 3 h hard
  wall-clock bound; neither came close (OFF 1200 s, ON 1011 s total).
- **Machine.** Windows 11, R 4.6.0, Rtools45/GCC 14.3.0, package rebuilt from HEAD
  (`220bff0`) immediately before the run; `packageDescription()$Built`
  (2026-07-05 15:04 UTC) verified newer than the target commits (stale-DLL check,
  docs/47 §1 pitfall).
- **CPU health preflight** (docs/22 lesson): a 600-generation gaussian calibration run
  gave cpu/wall = 9.27 (OFF) and 9.15 (ON) — the machine grants full multi-threaded
  CPU; wall times are trustworthy. The same preflight confirmed the feature is active
  and bit-identical: hit rate 0.282, and expression, loss, and `n_evals` all identical
  ON vs OFF at a generation-bound (not wall-clock-bound) budget.
- **Single-seed caveat.** `runs=1` means each per-problem wall time is a single
  measurement; the median in criterion (1) is over the 25 problems, not over seeds.
  Short-run entries (< ~25 s) carry the most relative timing noise.

## 4. Results (25 x 1 seed, OFF vs ON)

Result CSVs: `benchmarks/results/feynman_gate_20260706.csv` (OFF),
`feynman_gate_cache_20260706.csv` (ON, with per-run `cache_hits`/`cache_misses`/
`hit_rate` columns).

| problem | wall OFF (s) | wall ON (s) | change | hit rate | expr identical | recovered |
|---|---|---|---|---|---|---|
| gaussian | 10.5 | 9.2 | -12.0% | 0.282 | yes | both |
| rel_mass | 55.0 | 47.8 | -13.1% | 0.332 | yes | both |
| coulomb | 13.6 | 12.7 | -6.8% | 0.235 | yes | both |
| spring_pe | 1.5 | 1.3 | -8.9% | 0.216 | yes | both |
| lorentz_x | 54.9 | 49.9 | -9.2% | 0.316 | yes | both |
| rel_mom | 62.2 | 44.3 | -28.7% | 0.330 | yes | both |
| harmonic_ke | 48.5 | 35.9 | -26.0% | 0.281 | yes | neither |
| interference | 75.8 | 53.3 | -29.6% | 0.289 | yes | neither |
| bohr_radius | 42.9 | 31.1 | -27.6% | 0.259 | yes | both |
| larmor_rad | 73.6 | 54.5 | -26.0% | 0.274 | yes | both |
| planck | 92.7 | 73.3 | -20.9% | 0.272 | yes | neither |
| center_mass | 45.7 | 46.5 | +1.8% | 0.291 | yes | both |
| torque | 7.4 | 7.6 | +2.8% | 0.230 | yes | both |
| larmor_freq | 11.0 | 11.1 | +1.4% | 0.239 | yes | both |
| doppler_rel | 44.3 | 43.5 | -1.9% | 0.329 | yes | both |
| einstein_smol | 0.7 | 0.7 | -1.4% | 0.189 | yes | both |
| driven_osc | 23.2 | 24.8 | +6.8% | 0.237 | yes | both |
| heat_conduct | 6.5 | 6.7 | +2.9% | 0.225 | yes | both |
| boltzmann_dist | 80.2 | 76.6 | -4.5% | 0.277 | yes | both |
| clausius_moss | 91.0 | 82.2 | -9.7% | 0.333 | yes | both |
| clausius_moss2 | 90.8 | 76.1 | -16.3% | 0.341 | yes | both |
| bohr_magneton | 13.1 | 12.0 | -8.6% | 0.241 | yes | both |
| bose_einstein | 72.2 | 57.8 | -20.0% | 0.303 | yes | neither |
| lens_eq | 120.9 | 98.0 | -18.9% | 0.297 | yes | neither |
| newtons_grav | 61.4 | 53.9 | -12.3% | 0.284 | yes | neither |

Aggregates:

- Total wall: 1200 s (OFF) -> 1011 s (ON), **-15.75%**. Timeouts: 0 in both arms.
- Recovery: 19/25 in both arms (gate PASS both; identical per-problem outcomes, as
  bit-identity requires).
- Hit rate: min 0.189, median 0.281, max 0.341.
- Expressions: 25/25 identical (no run timed out, so all 25 rows are in scope for
  criterion 3; NMSE and loss also identical to all printed digits).

### Criteria evaluation

| # | criterion | measured | met? |
|---|---|---|---|
| 1 | median wall change <= -5% | **-9.71%** | yes |
| 2 | no problem > +2% | driven_osc **+6.8%**, heat_conduct **+2.9%**, torque **+2.8%** | **no** |
| 3 | non-timeout expressions identical | 25/25 | yes |

**Verdict: keep the default OFF** (pre-registered rule: all three or nothing).

## 5. Interpretation

- The speedup is real and concentrated where wall time matters: every problem with
  OFF-wall >= 43 s got faster (up to -29.6%), which is where a default-ON would have
  paid off. The three criterion-2 violations are all short runs (6.5-23.2 s) whose
  absolute regressions are 0.2-1.6 s — the size of single-run timing noise at
  `runs=1`, and all three have healthy hit rates (0.23-0.24), so a genuine systematic
  slowdown is not the obvious explanation. But the bar was pre-registered exactly to
  prevent post-hoc reasoning of this kind from flipping a default, so it stands.
  A future re-screen with multi-seed medians per problem (the Benchmarking
  Requirements norm) could revisit the default with the same three criteria.
- Bit-identity is confirmed end-to-end at the gate scale: identical expressions,
  losses, and recovery on all 25 problems, plus identical `n_evals` in the
  generation-bound preflight. The hit-charging rule works as designed.
- The measured hit rate (0.19-0.34, median 0.28) is the recorded reason the opt-in
  remains useful: roughly a quarter of all forward passes at PySR-parity settings are
  exact re-evaluations of a recently seen tree, and skipping them buys ~10-30% wall
  on the compute-bound problems with zero behaviour change.

## 6. Reproduction

```sh
# from the project root, package installed from HEAD (check Built vs HEAD first)
Rscript benchmarks/02_feynman_gate.R stage=1 runs=1         # arm A (OFF)
Rscript benchmarks/02_feynman_gate.R stage=1 runs=1 cache   # arm B (ON), strictly after A
```

The `cache` flag writes to `feynman_gate_cache_<date>.csv`; both CSVs carry
`cache_hits`/`cache_misses`/`hit_rate` per run (0/0/NA when OFF). Arms must never run
concurrently (wall-clock contamination); check cpu/wall health before trusting wall
times (docs/22).

## 7. Session note

Arm A's launcher session was interrupted by an external usage limit while waiting;
the detached benchmark process itself ran to completion unaffected (verified: full
25-problem log, exit 0, wall 1200.2 s within the 3 h bound, results CSV complete).
No restart was needed; Arm B was launched after confirming Arm A's completion.
