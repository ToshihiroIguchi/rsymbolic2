# 43. Multi-restart recovery ceiling: measure before building

**Date:** 2026-07-01
**Branch:** `feature/high-accuracy-options`
**Status:** Phase 0 complete. Decision: **NO-GO** — multi-restart is dominated by simply
running one search longer at equal compute (0c). No feature was built.

## Purpose

First candidate for the opt-in high-accuracy layer (CLAUDE.md "Opt-in high-accuracy
options"): **run-level multi-restart** — run N independent searches with distinct seeds,
merge their halls of fame, report the best Pareto front. Before writing any feature code,
this doc measures **how much recovery multi-restart could buy, and on which problems**, so
the build decision rests on evidence (CLAUDE.md #5), not intuition.

The whole ceiling is computable from per-seed recovery data already on disk: a K-restart
bundle recovers a problem iff at least one of its K independent runs recovers, so under the
independent-restart model the K-restart recovery probability is

    union_K = 1 - (1 - p)^K,   p = single-run recovery rate.

Two consequences frame everything:
- **p = 0 problems are restart-inert.** If no seed ever recovers at a given budget, no
  bundle of such runs recovers. `docs/38` already showed `interference` is p = 0 at the
  parity budget (0/15 @2800 gens). Restart cannot help these; they need a budget or
  structural lever (a later, separate plan).
- **Only p in (0, 1) problems are restart-responsive** — those rsymbolic2 solves
  *sometimes*. These are the target set.

## Method

- **0a (free, no new compute):** `benchmarks/analyze_restart_ceiling.R` reads the existing
  5-seed dev-gate CSV (`benchmarks/results/feynman_gate_20260627.csv`, 25 problems × 5
  seeds, per-seed `recovered` = NMSE < 1e-4 from `utils.R`), computes per problem
  `p_hat = r/S` and the binomial union-of-K curve, and classifies each problem. It reuses
  the recovery column verbatim; no search is run.
- **0b (targeted new compute):** re-run only the responsive keys at more seeds to sharpen
  `p_hat` (20-seed parity-budget run via `02_feynman_gate.R stage=1 runs=20 target=…`), and
  the **equal-compute test** — union-of-K at budget B vs a single run at budget K·B (via
  `diag_interference.R gens=…`) — to check restart beats simply running longer.
- Config is the frozen Stage-1 parity setup (`02_feynman_gate.R BENCH_PARAMS`: pop=27,
  islands=31, tournament=15, gens=2800, scaling=1040, optprob=0.14, maxsize=30, neg-free
  operator set), OMP pinned to the 10 physical cores. Recovery threshold NMSE < 1e-4.

## 0a results — single-run recovery rate over the 5-seed dev gate

Source: `feynman_gate_20260627.csv` (S = 5 seeds/problem). Union-of-K under
`1 - (1 - p_hat)^K` (independent restarts).

| class | problems | K1 | K2 | K3 | K5 | K10 |
|---|---|---|---|---|---|---|
| **RESPONSIVE p=0.8** | clausius_moss, lens_eq, lorentz_x | 0.80 | 0.96 | 0.99 | 1.00 | 1.00 |
| **RESPONSIVE p=0.4** | center_mass | 0.40 | 0.64 | 0.78 | 0.92 | 0.99 |
| **RESPONSIVE p=0.2** | boltzmann_dist, harmonic_ke, newtons_grav | 0.20 | 0.36 | 0.49 | 0.67 | 0.89 |
| **INERT p=0** | bose_einstein, interference, planck | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| **SOLVED p=1** | 15 problems | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Counts: **7 responsive, 3 inert, 15 solved.**

- **Sanity vs `docs/38`:** `interference` is p = 0 at the parity budget — matches `docs/38`
  (0/15 @2800). The restart model correctly flags it as inert.
- **The lever is real but bounded.** All seven responsive problems finish their 2800-gen
  budget well under the 300 s timeout (observed 40–114 s), i.e. they are *generation-bound,
  not time-bound* — so their recovery outcome is independent of the machine's power/clock
  state (only wall-clock time is). Multi-restart converts the four gate-failing responsive
  problems (center_mass 0.4; boltzmann_dist / harmonic_ke / newtons_grav 0.2) from a coin
  flip into ~0.9+ recovery by K = 5–10 — *if* the p̂ estimates hold up at more seeds and
  restart beats running longer (both tested in 0b).
- **Caveat:** p̂ from 5 seeds is noisy (0.2 = 1/5 could be anywhere ~0.05–0.45). 0b
  re-estimates the responsive keys at 20 seeds before any decision.

## 0b results — sharpened p (20 seeds) and equal-compute test

### Sharpened single-run recovery rate

Source: `feynman_gate_diag_20260702.csv` (S = 20 seeds/problem, parity budget 2800 gens,
OMP = 10 physical cores, high-performance power). Every run finished 2800 gens in 50–101 s
median (generation-bound, not time-bound).

| key | p̂ (5-seed 0a) | **p̂ (20-seed 0b)** | K3 | K5 | K10 | K20 |
|---|---|---|---|---|---|---|
| clausius_moss | 0.80 | **0.95** | 1.00 | 1.00 | 1.00 | 1.00 |
| lens_eq | 0.80 | **0.75** | 0.98 | 1.00 | 1.00 | 1.00 |
| lorentz_x | 0.80 | **0.55** | 0.91 | 0.98 | 1.00 | 1.00 |
| center_mass | 0.40 | **0.40** | 0.78 | 0.92 | 0.99 | 1.00 |
| boltzmann_dist | 0.20 | **0.15** | 0.39 | 0.56 | 0.80 | 0.96 |
| newtons_grav | 0.20 | **0.10** | 0.27 | 0.41 | 0.65 | 0.88 |
| harmonic_ke | 0.20 | **0.05** | 0.14 | 0.23 | 0.40 | 0.64 |

- All seven stay p ∈ (0, 1): every one has a real restart ceiling. The 5-seed estimates
  were noisy as expected — `lorentz_x` dropped 0.80→0.55, `harmonic_ke` 0.20→0.05 — which
  is exactly why 0a's numbers were treated as provisional.
- **The lift is large for the mid/high-p problems** (lorentz_x, center_mass reach ≥0.9 by
  K=5; clausius_moss/lens_eq essentially certain by K=3) and **real but expensive for the
  low-p tail** (harmonic_ke needs K≈20 for ~0.64).
- High seed-variance at a fixed budget (e.g. lorentz_x recovers 55% of seeds, each run
  finishing the same 2800 gens) is the signature of a **trajectory-limited**, not
  budget-limited, problem — the regime where independent restarts should beat running one
  search longer. The equal-compute test below checks this directly.

### Equal-compute test (single @ K·B vs union-of-K @ B)

The decisive test: at matched total compute, does bundling K restarts beat simply running
one search K× longer? K = 5, so union-of-5 @ 2800 gens (5 × 2800 = 14000 total) vs a single
run @ 14000 gens. Single-run measured over 10 seeds via `diag_interference.R key=… gens=14000
timeout=900` (14000 gens ≈ 490 s; 300 s would truncate it, so timeout was raised — recovery
is still generation-bound and power-independent). Three keys spanning the p-range:

| key | p̂@2800 | union-of-5 @2800 (5×2800) | **single @14000 (measured, 10 seeds)** | winner |
|---|---|---|---|---|
| lorentz_x | 0.55 | 0.98 | **1.00 (10/10)** | longer single (+0.02) |
| center_mass | 0.40 | 0.92 | **1.00 (10/10)** | longer single (+0.08) |
| boltzmann_dist | 0.15 | 0.56 | **0.60 (6/10)** | longer single (+0.04) |

**On all three, single @ K·B ≥ union-of-K @ B.** Running one search longer matches or beats
K independent restarts at equal total compute. The mechanism is clear: a single long run
keeps refining one hall-of-fame and population across all 14000 gens, whereas each restart
discards that progress and re-initialises. Multi-restart is therefore **dominated** by the
existing generation-budget knob — it would trade compute for *less* recovery than simply
raising `generations`/`niterations`.

This also refines `docs/35`: for these responsive problems the generation budget is **not** a
dead lever — more generations genuinely lift recovery (lorentz_x 0.55→1.0, center_mass
0.40→1.0, boltzmann_dist 0.15→0.60). The plateau `docs/35` observed does not extend to the
2800→14000 range for the hard-tail problems.

## 0c decision — NO-GO for multi-restart

**Do not build the multi-restart / ensemble feature.** The go criterion was "union-of-K @ B
materially exceeds single @ K·B" — restart beating a longer single run at equal compute. The
measurement is the opposite: single @ K·B ≥ union-of-K @ B on every problem tested. Building
`n_restarts` would add public API surface for a strategy strictly dominated by the existing
`generations`/`niterations` parameter. This is exactly the outcome the "measure before
building" plan (Phase 0) existed to catch: no C++/API change was made, and none should be.

**What actually moves accuracy here:** the generation budget. It is already a parameter; a
user wanting higher accuracy raises `generations` (equivalently PySR `niterations`) and pays
linearly in compute. This is a documentation/preset matter, not a new mechanism — and it must
respect PySR default parity: the *default* budget stays PySR's; a larger budget is the opt-in.

**Still out of reach:** the p = 0 problems (bose_einstein, interference, planck). More budget
polishes wrong basins (`docs/38`) and restart is inert; neither lever discovers the compact
true structure. Those need a *structural* lever (e.g. dimensional analysis / typed operators
to prune the space), which is a separate mechanism — take it to its own plan, do not fold it
in here (CLAUDE.md: no speculative scope creep).

**Recommendation:** close this feature as measured-and-rejected. Next candidate for the
opt-in high-accuracy layer: (a) a documented "accuracy vs compute" note / optional larger
budget preset (trivial, already-available knob), and/or (b) dimensional analysis as a fresh
plan. Decide with the user which to pursue.

## Reproduction

```
# 0a — free re-analysis of the existing 5-seed gate CSV:
Rscript benchmarks/analyze_restart_ceiling.R csv=benchmarks/results/feynman_gate_20260627.csv

# 0b — 20-seed sharpening on the responsive keys (OMP pinned to physical cores):
#   $env:OMP_NUM_THREADS=10
Rscript benchmarks/02_feynman_gate.R stage=1 runs=20 nofailfast \
  target=clausius_moss,lens_eq,lorentz_x,center_mass,boltzmann_dist,harmonic_ke,newtons_grav
Rscript benchmarks/analyze_restart_ceiling.R csv=benchmarks/results/feynman_gate_diag_20260701.csv kmax=20
```

See also: `docs/38` (interference is p=0 / deceptive basin), `docs/35` (throughput/plateau),
`docs/28`/`docs/29` (PySR parity spec), CLAUDE.md "Opt-in high-accuracy options".
