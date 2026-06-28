# 24. Frequency-based adaptive parsimony

Status: **implemented and measured.** The mechanism ships behind a new option
`adaptive_parsimony_scaling` (R) / `SearchOptions::adaptive_parsimony_scaling`
(C++). This document records the mechanism, why it was added, the sweep evidence,
and the default decision.

## Why

docs/23 isolated the Stage 1 Feynman gate failure (12/25) to a single root cause:
the fixed-scalar `parsimony` term in `selection_cost = loss/y_norm +
parsimony*complexity` is mis-scaled per problem. On low-variance targets
(center_mass, y_norm≈720) `parsimony=1e-3` drives the population to **collapse to
7 nodes** — too small to express the 11-node answer — after which the search is
flat for the rest of the run (H2 diversity collapse). Setting `parsimony=0`
removes the collapse but lets trees bloat to the node cap, which starves
throughput (H3). Both extremes fail.

## Mechanism (borrowed from PySR / SymbolicRegression.jl, re-implemented in C++)

PySR has a **two-layer** design, and rsymbolic2 now mirrors it:

1. **Output archive — error-vs-complexity Pareto front.** `HallOfFame` keeps the
   best raw-loss member at each complexity (`hall_of_fame.{hpp,cpp}`), returned as
   `result$pareto_front`. Unchanged by this work; it is keyed on **raw loss**, so
   reporting is never penalized.
2. **Evolutionary selection pressure — frequency-adjusted cost.** In tournament
   selection a candidate's base cost is multiplied by a frequency penalty:

   ```
   base     = loss / y_norm + parsimony * complexity
   adjusted = base * exp(adaptive_parsimony_scaling * normalized_freq[complexity])
   ```

   `normalized_freq` is a per-island running histogram over complexity
   (`RunningSearchStatistics` in `evolutionary_search.cpp`): every evaluated child
   increments `frequencies[complexity]`; `normalized = frequencies / sum`; bins
   start at 1.0; a sliding window (`parsimony_window`, default 100000) scales the
   bins down once their sum exceeds it. This penalises **over-represented**
   complexities rather than **large** ones — so when the population piles up at one
   size, that size is pushed back. It is a self-balancing diversity pressure that
   keeps the population spread across complexities, which is exactly what lets the
   Pareto archive (layer 1) get populated at the complexity the answer needs.

### What is "adaptive"

The **frequency histogram** adapts during the run; the **scaling coefficient is a
fixed hyperparameter** (this is also true in PySR). "Adaptive" refers to the
histogram, not the coefficient.

> **Correction (docs/28, PySR-parity).** PySR's installed default for
> `adaptive_parsimony_scaling` is **1040.0** (verified in `pysr/sr.py`), **not ~20** —
> an earlier note in this file claimed ~20, which was wrong. SymbolicRegression.jl uses
> `member.cost * exp(scaling * normalized_frequencies[size])` where
> `normalized_frequencies = frequencies/sum` sums to 1 over **`maxsize`** bins (uniform
> `1/maxsize ≈ 0.033`). 1040 does not explode because a *uniform* histogram gives every
> member the same factor (it cancels in the ranking); the penalty acts on relative
> frequency differences. To make 1040 transfer exactly, rsymbolic2 must size its
> histogram by `maxsize` (= `max_nodes`, with PySR's 30), not `max_nodes+2`. Per
> CLAUDE.md "PySR Default Parity", the default is **1040 (ON)**, matching PySR — the
> "default OFF" conclusion below is **superseded**. The `scaling=0.5` sweep result was
> specific to the 4×500 single-island collapse probe and is not PySR's value. See
> `docs/28` for the authoritative spec.

### Justified divergence from PySR

PySR ships `parsimony=0.0` and relies on the frequency mechanism alone. Our
measurements show `parsimony=0` bloats and starves throughput here (different
loss/throughput characteristics — see the probe below). rsymbolic2 therefore keeps
a small linear `parsimony` (bloat control) **and** adds the frequency penalty
(anti-collapse). **Superseded by docs/28 (PySR parity):** PySR ships `parsimony=0`
with `adaptive_parsimony_scaling=1040` over a `maxsize`-binned sum-to-1 histogram;
rsymbolic2 now matches that exactly rather than keeping a self-tuned linear parsimony
or a swept `scaling`. The `0.5` below is a 4×500 single-island artefact, not PySR's
value.

## Design choices (implementation)

- **Per island, share-nothing.** One `RunningSearchStatistics` per `Island`, so
  `n_populations=1` stays fully deterministic and parallel islands need no
  synchronisation. Updates consume **no RNG**, so fixed-seed reproducibility holds
  (verified by tests).
- **Backward compatible.** `scaling <= 0` returns exactly the previous
  `base_cost`, byte-for-byte. Default is `0` (off) until/unless flipped with
  evidence.
- **Normalize cadence.** The normalized snapshot is refreshed once per generation
  (cheap, O(max_nodes)); raw counts accumulate continuously. Matches SR.jl's
  per-cycle cadence.
- **Migration uses `base_cost`** (no frequency factor): migration transfers
  quality between islands and must not depend on one island's histogram.
- **Overflow guard.** The exponent is clamped to ≤ 50 so a dominated histogram
  cannot produce `+inf` ties.
- **Window deviation.** `move_window` scales all bins down proportionally; SR.jl
  subtracts from the largest bins. The proportional form is simpler and keeps
  every bin strictly positive. Over the runs measured here the window is never hit
  (sum « 100000), so the two are equivalent in practice.

## Sweep (single-island for clean attribution; bounded timeouts)

`parsimony=1e-3`, `optimize_probability=0.1` fixed (shipped values). All runs
`cpu/wall≈1.0` (healthy). Harness: `benchmarks/diag_multivar.R` (now takes a
`scaling=` knob) and `benchmarks/_b3_sweep.R` (Nguyen no-regression).

### Scaling magnitude is narrow and centred on ~0.5

Selecting the coefficient: near the plateau the loss differences are tiny, so the
frequency factor must overcome only the ~0.004 parsimony gap, not dominate the
loss term. `exp(scaling * freq)` with `freq` up to ~0.3 means the useful range is
order 0.5, **not** PySR's 1040 (which here gives `exp(~300)` → loss ignored).

center_mass (single island, 120–180 s cap), `best raw loss` trajectory:

| scaling | population size | best loss | outcome |
|---------|-----------------|-----------|---------|
| 0 (off) | collapses to **7/7** by epoch ~4 | flat **42.3** | NMSE 5.88e-2, stuck |
| 0.3     | bloat then collapse to 5–6 | flat 53 | NMSE 6.7e-2, worse |
| **0.5** | stays **18–31** (no collapse) | **49→45→28, descending** | NMSE 3.97e-2 |
| 2       | 33–38 (bloat) | flat 53 | NMSE 7.4e-2, loss gradient swamped |
| 20      | 33/50 (bloat) | flat 44 | NMSE 6.2e-2, selection ignores loss |

`scaling=0.5` is the only tested value that both breaks the collapse and lets the
loss keep descending. Below it the population still collapses; above it the
multiplicative penalty swamps the (weak) loss gradient and the search chases rare
sizes instead of accuracy.

### scaling=0.5 helps both diagnosed targets, no Nguyen regression

Single-island, 120 s cap, same build:

| target | scaling=0 | scaling=0.5 |
|--------|-----------|-------------|
| center_mass | collapse, NMSE **5.88e-2** | descends, NMSE **3.97e-2** |
| newtons_grav | size 37–41, NMSE **1.27e-1** | size 31, NMSE **2.38e-2** (~5×) |

Nguyen no-regression (`benchmarks/_b3_sweep.R`, np=4, 200 gen, 120 s cap):

| case | scaling=0 | scaling=0.5 |
|------|-----------|-------------|
| N9 s5  | recovered, 2 s | recovered, 4 s |
| N10 s3 | recovered, 1 s | recovered, 1 s |
| N1 s1  | recovered, 2 s | recovered, 1 s |
| N5 s1  | recovered, 2 s | recovered, 5 s |
| N7 s1  | recovered, 2 s | recovered, 2 s |

All Nguyen cases still recover quickly; no slow-run blowup.

## Gate-config check (np=4, the 8 diagnosed-failing problems)

The single-island sweep above cleanly attributes the *mechanism*, but the gate
runs at `n_populations=4` with stochastic migration. Running the 8 division/
rational problems that score 0/5 in the gate, at the gate config
(`pop=500, islands=4, gens=500`), 150 s cap, seed=1, `parsimony=1e-3`:

| problem | scaling=0 | scaling=0.5 |
|---------|-----------|-------------|
| center_mass    | 5.88e-2 | 6.48e-2 |
| newtons_grav   | 4.79e-2 | 8.90e-2 |
| lorentz_x      | 8.17e-3 | **9.79e-5 ✓recovered** |
| planck         | 7.91e-2 | 9.38e-2 |
| interference   | 5.02e-2 | 4.40e-2 |
| driven_osc     | 4.19e-1 | 2.19e-1 |
| boltzmann_dist | 2.61e-1 | 1.80e-1 |
| bose_einstein  | **9.91e-4** | 1.94e-2 |

Mixed: `scaling=0.5` recovers lorentz_x and improves driven_osc / boltzmann_dist /
interference, but regresses newtons_grav and bose_einstein (the latter is nearly
recovered at scaling=0). The seed/config variance is large — e.g. newtons_grav is
2.38e-2 single-island at 0.5 but 8.90e-2 at np=4 — so a single seed cannot rank
the two reliably. `cpu/wall` was 2.0–2.9 here (healthier than the 1.58 in docs/23).

## Default decision: ship as an option, keep the default OFF (scaling=0)

The clean single-island attribution proves the mechanism does what it was designed
to do (removes the collapse, improves the diagnosed targets, no Nguyen
regression). But at the actual gate config a **single global** `scaling=0.5` is not
a clear net win across problems: it trades some recoveries for others at single
seed, and per-problem the optimal coefficient clearly differs. Flipping the
shipped default on this evidence is not justified (CLAUDE.md: do not weaken / force
a change without evidence; let results change the plan).

Decision:
- **Default stays `adaptive_parsimony_scaling = 0` (off)** — zero regression, all
  tests green, fully backward compatible.
- The mechanism **ships as a tunable option** for users/benchmarks that want it.
- A default flip should be reconsidered only after: (a) a **multi-seed** gate study
  (the single-seed variance here is too large to decide), and (b) resolving the
  §4 parallel-efficiency regression (docs/23), which starves exactly these
  throughput-limited problems and confounds the gate signal. Both are tracked
  separately and are out of scope here.

Note: center_mass and newtons_grav still do not *recover* (NMSE < 1e-4) within
these bounded budgets — they only improve. Reaching recovery also depends on H3
throughput, not selection pressure alone.

## Reproduction

```
# center_mass: collapse (scaling=0) vs descent (scaling=0.5)
Rscript benchmarks/diag_multivar.R mode=traj keys=center_mass seeds=1 islands=1 gens=1000 timeout=120 parsimony=1e-3 scaling=0
Rscript benchmarks/diag_multivar.R mode=traj keys=center_mass seeds=1 islands=1 gens=1000 timeout=150 parsimony=1e-3 scaling=0.5

# newtons_grav
Rscript benchmarks/diag_multivar.R mode=traj keys=newtons_grav seeds=1 islands=1 gens=1000 timeout=120 parsimony=1e-3 scaling=0.5

# Nguyen no-regression
Rscript benchmarks/_b3_sweep.R 0,0.5
```
