# 39. Tier-1 optional PySR features: model_selection, max_evals / early_stop_condition, weights

**Date:** 2026-06-27
**Scope:** Implement the three "Tier 1" optional (default-OFF) PySR features identified as
worth adding for a usable shipping library, while leaving the default search behaviour
bit-identical to before (CLAUDE.md PySR Default Parity). These are opt-in capabilities; none
is parity-required, and all stay off / inert unless the caller sets them.

The audit that produced the Tier ranking is `docs/29` §C (PySR feature gaps, all default-OFF)
and `docs/32` (none is "true + default-ON + unimplemented"). This file records what was built
and why, so the choices are not re-derived next time.

## What was implemented

### 1. `model_selection` (best / accuracy / score)

PySR's `model_selection` chooses which Pareto-front member is reported as the recommendation.
`select_best` (`hall_of_fame.hpp/.cpp`) now takes a `ModelSelection` enum:

- **accuracy** — the lowest-loss member (`front.back()`; the front is strictly decreasing in
  loss).
- **score** — the member maximising the log-loss drop per unit complexity over the whole
  front (the accuracy/complexity "knee").
- **best** — the same score, but considered only among members within 1.5x of the most
  accurate (lowest) loss. PySR's default.

**Parity correction.** The previous `select_best` was documented as `model_selection="best"`
but actually computed `score` (no 1.5x accuracy filter). It now applies the filter, so the
default `recommended` / `best_index` matches PySR's documented `"best"`. This is a parity fix
(CLAUDE.md: a default that differs from PySR is a bug to fix), not a tuning change. R argument:
`model_selection = c("best","accuracy","score")`.

### 2. `max_evals` and `early_stop_condition`

- **`early_stop_condition`** (numeric form): an additional early-stop loss threshold. The
  effective threshold is `max(target_loss, early_stop_condition)`, so a value larger than
  `target_loss` stops the run sooner at a looser fit. The Julia-function (loss+complexity)
  form is intentionally not supported (`docs/29` §C). Default `0` (off). Note: rsymbolic2's
  existing `target_loss` already provided the float early-stop mechanism; `early_stop_condition`
  is the PySR-named, default-off user knob layered on top of it.

- **`max_evals`**: cap the total candidate evaluations (forward-pass loss evals + the residual
  evaluations consumed by constant-optimisation fits), summed across islands. Default `0` (no
  limit).

  **Design — deterministic, contention-free.** Rather than a shared atomic counter on the hot
  path (which would re-introduce exactly the cross-island contention the share-nothing island
  design eliminated, `docs/23`/`docs/25`), each island keeps a plain `eval_count` and
  self-limits to a fair share `ceil(max_evals / n_populations)`, checked at generation
  boundaries (never mid-fit, so a fit is never aborted and counting stays clean). A global sum
  is checked at each epoch boundary as the hard cap. Consequences:
  - When `max_evals == 0` (default): zero extra work, zero atomics — the default path is
    bit-identical.
  - When set: **deterministic and thread-count independent** (unlike `timeout_seconds`), because
    each island's fair share is fixed and the hall-of-fame merge is order-independent. A capped
    single-island run is an exact prefix of the full run, so `full.loss <= capped.loss` holds
    (monotone HOF) — used as a test invariant.
  - Coarse: may overshoot by up to one generation's evals per island.

### 3. `weights` (weighted least squares)

PySR's `weights` give per-point weights. Implemented as the standard weighted-LS reduction:
scale residual `r_i` (and its Jacobian row) by `sqrt(w_i)`, so the optimiser's SSE becomes
`sum_i w_i (pred_i - y_i)^2`. `Dataset` precomputes `sqrt_weights` once; the residual closure
(`least_squares_problem.hpp`), the Jacobian closure, and `sse_current`
(`evolutionary_search.cpp`) apply it. Empty weights => the multiply is skipped entirely
(default path bit-identical). `compute_y_norm` (the NMSE selection denominator) uses the
weighted variance, so the `loss / y_norm` ratio is invariant to rescaling all weights.

R argument `weights = NULL` (unweighted) accepts a non-negative vector of length `length(y)`.

**Not implemented (deliberately):** arbitrary `elementwise_loss` / `loss_function`. A general
pointwise loss breaks the least-squares assumption the LM optimiser is built on and would
require a separate general optimiser — a larger architectural change deferred per `docs/29` §C.
Weighted L2 (the common case) is covered by `weights`.

## Verification

- New unit tests: `select_best` three modes (`test_hall_of_fame.cpp`); weighted fit ignores a
  zero-weighted outlier and scales linearly with the weights (`test_constant_fitting.cpp`);
  `max_evals` determinism + `full.loss <= capped.loss` bound, `early_stop_condition` gating, and
  weighted search recovery (`test_evolutionary_search.cpp`); R-level `test-new-options.R`
  (model_selection modes + accuracy invariant + invalid-mode error, weight validation,
  zero-weight recovery, max_evals determinism/bound, early-stop gating).
- **Both platforms** (CLAUDE.md mandatory): standalone ctest 19/19 and the full R testthat suite
  pass on **Windows (Rtools45/MinGW)** and **Ubuntu 24.04 (WSL, g++ 13.3)**.
- Default behaviour unchanged: all pre-existing recovery/parity/island tests pass; default paths
  are bit-identical (no weight multiply, no eval counting, `target` == `target_loss`,
  `model_selection` only changes the reported recommendation to PySR's actual `"best"`).
