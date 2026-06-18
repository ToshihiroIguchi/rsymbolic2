# 23. Multivariate Feynman recovery diagnosis (Stage 1 gate 12/25)

Status: **diagnosis only**. No fix is implemented here. Proposed fix directions
are listed at the end and are deferred to a separate, agreed task.

## Context

The Stage 1 Feynman dev gate recovers 12/25 (threshold 18/25, FAIL). The earlier
timeout-overshoot problem (docs/21–22) is resolved and is *not* the cause here:
every run now stays within its 300 s budget. The remaining problem is purely
recovery rate. Eight problems score 0/5, and they share a structure: division /
multi-variable rational forms (center_mass, newtons_grav, lorentz_x, planck,
interference, driven_osc, boltzmann_dist, bose_einstein).

The gate logs show these runs stalling at NMSE ≈ 1e-3…1e-1 (a "good enough"
approximation) rather than crashing. This document isolates *why* the search
does not reach the exact structure, using small controlled experiments on two
constant-free targets where the cause can be cleanly attributed.

Diagnostic harness: `benchmarks/diag_multivar.R` (reuses `feynman_dataset()`,
`compute_nmse()`). It runs the gate's hyperparameters but also captures the full
Pareto front and, with `verbosity=1`, the per-epoch best-loss / tree-size
trajectory. All runs below are single-seed, single-island (`n_populations=1`)
unless noted; single-island is deliberate — it removes both the parallel-
efficiency confound (see §4) and migration, so the search dynamics are
reproducible regardless of machine health.

## Why these two targets isolate the cause

- **center_mass** `(m1*r1 + m2*r2)/(m1+m2)` and **newtons_grav**
  `G*m1*m2/(dx^2+dy^2)` are **constant-free** (newtons_grav's `G` is a variable).
  So their failure cannot be LM constant optimization — it is purely structural
  search reachability. This rules out the "constant optimization stalls"
  hypothesis (H4) for these problems by construction.
- The result object's `pareto_front` is keyed on **raw loss** (parsimony-free),
  and `HallOfFame::update(child)` runs for **every child produced**
  (`evolutionary_search.cpp:227`). Therefore `result$loss` already equals the
  best raw loss ever produced anywhere in the search. For a constant-free target,
  the exact structure would evaluate to loss ≈ 0 on first sight and be captured.

## Hypotheses and verdicts

| # | Hypothesis | Verdict | Evidence |
|---|------------|---------|----------|
| H1 | Structure found but a simpler approximation is reported (selection/reporting) | **Rejected** | HoF captures every child's best raw loss across all complexities; reported NMSE = front-min NMSE. A constant-free exact hit would show loss≈0. It never does. |
| H2 | Premature convergence / diversity collapse trap | **Confirmed (center_mass)** | See §1. |
| H3 | Budget / throughput starvation | **Confirmed (newtons_grav), and amplified by §4** | See §2. |
| H4 | LM constant optimization stalls | **N/A for these targets** | Both are constant-free. |

The two 0/5 problems fail for *different* dominant reasons, both tracing back to
the parsimony scaling and to search throughput.

### 1. center_mass — parsimony-induced diversity collapse (H2)

Controlled experiment: identical config, only `parsimony` changed; `np=1`,
`generations=1000`, no timeout.

| parsimony | best raw loss trajectory | median tree size | outcome |
|-----------|--------------------------|------------------|---------|
| 1e-3 (default) | 42.3 → 42.3 (flat for all 1000 gens) | collapses to **7** by epoch 4, then frozen 7/7 | NMSE 5.9e-2, never improves |
| 0 | 50 → 29 → 9.4 → **1.52** (still descending) | stays **45–47** (near max_nodes) | NMSE ≈ 2.1e-3 and falling |

The exact target needs **11 nodes**. With `parsimony=1e-3` the population
collapses to 7-node trees — **too small to represent the answer** — and then
makes zero progress for 960 further generations. Setting `parsimony=0` removes
the collapse: large structure-bearing trees survive and best loss falls by ~28×.

Mechanism. `selection_cost = loss / y_norm + parsimony * complexity`. For
center_mass, derived `y_norm ≈ 720`. Near the approximation plateau, adding the
nodes needed for the rational structure reduces normalized loss by less than the
parsimony cost of those nodes (`1e-3` per node), so selection consistently
prefers the smaller, worse tree. The penalty is **mis-scaled** for this
low-variance target: it out-weighs the loss gradient exactly where the search
needs to grow.

### 2. newtons_grav — throughput/budget-limited (H3)

Same harness, `parsimony=1e-3` default, `np=1`, 120 s cap:

```
[epoch 0 t=40.8s] best=2.574e+03  size med/max=37/50
[epoch 1 t=120s ] best=1.591e+03  size med/max=42/50   -> NMSE 1.27e-1, still descending
```

Here the population does **not** collapse (size stays ~42) and best loss is still
falling when the budget runs out. Derived `y_norm ≈ 12,500` (wide y range), so the
relative parsimony weight is smaller and the collapse does not occur. The gate
(`np=4`, 300 s) reaches ~1.6e-2 — better than this short run, confirming more
compute helps. This problem is **search-throughput limited**, not trapped.

### 3. The parsimony form is the common root

`parsimony * complexity` is a fixed per-node cost, while `loss / y_norm` varies
per problem. The intended scale-stability (docs/13) does not hold: the *relative*
weight of the penalty versus the loss gradient depends on each problem's
loss-vs-complexity landscape. Low-variance, deceptive targets (center_mass) are
over-penalized into collapse; `parsimony=0` fixes collapse but reintroduces the
bloat/runtime explosion that parsimony was added to prevent (epochs ballooned to
minutes as trees hit size 50). **Both extremes fail; a correctly scaled middle
ground is the fix target.**

## 4. Secondary finding: parallel efficiency regression amplifies H3

A health probe (`benchmarks/diag_omp_check.R`, fixed work, no timeout) shows
broken island scaling on this machine:

```
n_populations=1  wall=14.7s  cpu/wall=0.99
n_populations=4  wall=86.9s  cpu=137.5s  cpu/wall=1.58
```

Four islands do 4× the work but take ~5.9× the wall time of one island (ideal:
~1×). cpu/wall=1.58 (not ~4) with threads *blocked* (not spinning) points to
lock/heap contention, not frequency throttling — switching to the Windows
"performance" power mode improved single-thread speed (23→15 s) but not scaling.
The installed DLL (2026-06-16) already includes the thread-cap fix (commit
2e1a026), and background load was modest (~20%), so this is most likely the
inherent "efficiency falls off above 2–3 threads" noted in docs/04 §6, now worse.

Impact on the gate: at the measured pace, the `np=4`, 300 s gate completes only
~140 of the requested 500 generations (~28% of the intended budget). This
directly starves H3-type problems. Restoring parallel efficiency (or re-balancing
the budget) would multiply the effective search per wall-second.

## Proposed fix directions (deferred — require agreement before implementation)

1. **Re-scale parsimony so it cannot collapse the population below the size
   needed to express the target.** Options to evaluate: a smaller default; making
   the penalty relative to the achievable loss gradient rather than a flat
   per-node cost; or a complexity floor / size-aware penalty. Must not regress
   Nguyen runtime (docs/13). Verify via a parsimony sweep on center_mass +
   newtons_grav + the Nguyen no-regression set.
2. **Diversity preservation** independent of parsimony (e.g. periodic
   re-seeding / restart of collapsed islands, or protecting a fraction of large
   trees) to prevent premature convergence on deceptive low-variance targets.
3. **Parallel-efficiency / budget**: investigate the island-scaling contention
   (§4); separately, the gate budget assumes throughput the machine isn't
   delivering. Either is a lever on H3-type problems.

Recommended next step: a small `parsimony` sweep (e.g. `{0, 1e-5, 1e-4, 3e-4,
1e-3}`) on center_mass and newtons_grav with a bounded per-run timeout, plus the
Nguyen no-regression check, to locate the scaling that lifts recovery without
reintroducing bloat. This is a separate task.

## Reproduction

```
# health probe
Rscript benchmarks/diag_omp_check.R

# center_mass collapse vs parsimony=0 (bounded; parsimony=0 bloats, keep timeout small)
Rscript benchmarks/diag_multivar.R mode=traj keys=center_mass seeds=1 islands=1 gens=1000 timeout=180
Rscript benchmarks/diag_multivar.R mode=traj keys=center_mass seeds=1 islands=1 gens=200  timeout=180 parsimony=0

# newtons_grav throughput-limited
Rscript benchmarks/diag_multivar.R mode=traj keys=newtons_grav seeds=1 islands=1 gens=500 timeout=120
```

Raw outputs: `benchmarks/results/diag_*_runs_*.csv` and `diag_*_front_*.csv`.
