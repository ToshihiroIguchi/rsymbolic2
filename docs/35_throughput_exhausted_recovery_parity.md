# 35. Throughput is no longer the bottleneck: recovery parity with PySR reached

**Date:** 2026-06-27

> **Correction (2026-06-27, `docs/38`).** The `interference` analysis below (Measurement 3:
> "exploration-bound, ~3× generations, same backbone family PySR uses, not a structure
> rsymbolic2 is blind to") is **superseded by a many-seed, Pareto-structural re-examination**.
> Measured facts: PySR recovers `interference` only ~13–20 % of runs (the 2/3 here was a
> small-sample fluke); both tools predominantly fall into the *same* deceptive basin
> `(I1+I2)·(1+α·cos δ)`; rsymbolic2 **never** builds the compact exact `2√(I1 I2)·cos δ + I1 + I2`
> structure (0/23 parity-config runs, incl. 8 @8400), and its 8400 "recoveries" are the wrong
> basin polished under threshold, **not** the same family PySR uses. So this is a deceptive-basin
> + seed-noise effect, not a clean exploration-bound 3× requirement, and the "18 = 18 with a
> 1-for-1 swap" headline should be read as "≈18 each, the swap within seed noise." See `docs/38`.

## Purpose

`docs/26` §5 and `docs/30` framed the Feynman recovery gap as **throughput-bound**: at a
fair 4-thread budget the hard problems hit the 300 s timeout before completing the
PySR-equivalent generation budget, so the levers were faster evaluation (SoA / batched
`MultiDual`, both shipped) and search efficiency. `docs/31` (constant-optimisation cadence,
LM 93 % → 10.7 % of compute, ~10× candidate throughput) and `docs/34` (operator-set parity,
neg removed) have since changed the picture. This file re-confirms the **current** bottleneck
on a measured basis and records the consequence: **throughput is no longer the binding
constraint, and rsymbolic2 has reached PySR's documented recovery total *and* membership.**

This supersedes the throughput framing of `docs/26` §5 / `docs/30` for the *current* code.
The SoA / cadence optimisations were correct and load-bearing — they are *why* the timeouts
are gone — but with them in place the recovery gap is no longer a compute problem.

## Measurement 1 — nothing times out anymore

Authoritative 5-seed gate `benchmarks/results/feynman_gate_20260627.csv` (neg-free parity set,
`02_feynman_gate.R stage=1`, 300 s/problem budget): **all 125 rows have `timed_out = FALSE`.**
Every problem — including all 7 failures — **completes the full `generations = 2800` budget
well within the 300 s limit**:

| failing far-miss | median wall | budget used | result |
|------------------|------------:|------------:|--------|
| interference   | ~62 s  | ~21 % | 0/5, completed |
| harmonic_ke    | ~55 s  | ~18 % | 1/5, completed |
| center_mass    | ~60 s  | ~20 % | 2/5, completed |
| newtons_grav   | ~57 s  | ~19 % | 1/5, completed |
| bose_einstein  | ~75 s  | ~25 % | 0/5, completed |
| planck         | ~85 s  | ~28 % | 0/5, completed |
| boltzmann_dist | ~88 s  | ~29 % | 1/5, completed |

`generations = 2800` is the PySR `niterations × ncycles` parity mapping (`docs/28` §C) and is
**fixed by parity** — it cannot be raised without violating CLAUDE.md. The search now exhausts
exactly PySR's generation budget and finishes with 70–80 % of the wall-clock budget unused.

**Consequence for throughput optimisation.** A faster evaluator (the shipped SoA/MultiDual, or
any further work) makes the *same* 2800 generations finish *sooner* — it does not add
generations and cannot change the result. Throughput is a **dead lever** for recovery: the
binding constraint is now per-generation search quality, not candidates per second. The
`docs/30` headroom (vectorising transcendentals via SLEEF, etc.) is confirmed **irrelevant to
recovery** and remains correctly un-pursued (`docs/30` SLEEF decision stands).

## Measurement 2 — recovery parity with PySR, total *and* membership

Per-problem head-to-head of the current rsymbolic2 gate (`feynman_gate_20260627.csv`, 5 seeds,
majority ≥3/5) against faithful PySR (`sr_comparison_feynman_20260621.csv`, SR.jl engine,
`annealing=false`, `precision=32`, 3 seeds, majority ≥2/3):

|  | rsymbolic2 18/25 | PySR 18/25 |
|--|------------------|------------|
| both recover (17) | gaussian, rel_mass, coulomb, spring_pe, rel_mom, bohr_radius, larmor_rad, torque, larmor_freq, doppler_rel, einstein_smol, driven_osc, heat_conduct, clausius_moss, clausius_moss2, bohr_magneton, lens_eq | (same 17) |
| **rsymbolic2 only (1)** | **lorentz_x** (4/5) | lorentz_x 1/3 — PySR fails |
| **PySR only (1)** | interference 0/5 — rsym fails | **interference** (2/3) |
| both fail (6) | harmonic_ke, planck, center_mass, boltzmann_dist, bose_einstein, newtons_grav | (same 6) |

The earlier strict-subset result (`docs/26` §5: rsym-only = 0, rsym ⊂ PySR at 12/25) is **gone**.
The current head-to-head is **symmetric**: 18/25 each, identical except a **1-for-1 swap**
(rsymbolic2 wins `lorentz_x`, PySR wins `interference`). Two consequences:

1. **CLAUDE.md's highest-priority goal — PySR default parity — is reached on recovery**, not
   just on settings. The total *and* the membership match to within one symmetric problem.
2. **6 of the 7 rsymbolic2 failures are problems PySR also fails** (harmonic_ke, planck,
   center_mass, boltzmann_dist, bose_einstein, newtons_grav). Raising those would mean
   *exceeding* PySR — a search-quality ambition, **not a parity obligation** — and (per
   Measurement 1) not reachable through throughput.

## Measurement 3 — the one real gap (`interference`) is exploration-bound, not throughput-bound

`interference` is the sole problem PySR recovers that rsymbolic2 does not. A compute-scaling
diagnostic (non-parity, throwaway — the same role as varying `generations` in `docs/26` §3)
held everything at the parity config except the generation count:

| generations | seed 1 | seed 2 | seed 3 | recovered |
|-------------|-------:|-------:|-------:|:---------:|
| 2800 (parity)     | 1.19e-3 | — | — | 0 |
| **8400 (3×)**     | **3.96e-5 ✅** | 9.52e-4 | 1.88e-4 | **1/3** |

(`OMP_NUM_THREADS=4`; 8400-gen runs took 290–343 s.) Three reads:

- The structure **is reachable** — 3× the generation budget recovers it (1/3, with seed 3 at
  1.88e-4 just above the 1e-4 line), landing in PySR's own ballpark (PySR 2/3). The found
  backbone (a `cos(x3)·… + (x1+x2)` form with tanh/pow factors) is the **same family** PySR's
  recovered expressions use — so this is not a structure rsymbolic2 is blind to.
- The 30× NMSE drop from *more generations* (not more optimiser effort) shows the miss is
  **exploration-bound** (the search needs more draws to find a polishable tree), not
  **polish-bound** (constants left unconverged). PySR reaches it inside the 2800-equivalent
  budget; rsymbolic2 needs ~3× — a **per-generation search-efficiency** difference.
- That efficiency difference is exactly the CLAUDE.md-permitted **category-A** residue
  (`docs/29` §A/§E): self-LM vs BFGS, Float64 vs Float32, and the RNG trajectory. The
  structural-mutation and constant-optimiser *settings* are already matched
  (`docs/29` A#13, `docs/33`); only the implementation methods differ.

## Why there is no parity-preserving lever left for the far-misses

- **Throughput** (candidates/s): dead — nothing times out; `generations` is parity-fixed at
  2800 and already exhausted (Measurement 1).
- **More generations**: a parity violation (`generations` is a must-match setting), usable only
  as a diagnostic.
- **Optimiser effort** (`optimizer_iterations = 8`, `optimizer_nrestarts = 2`): must-match PySR
  settings (CLAUDE.md). Raising them moves *away* from PySR, not toward it — rejected.
- **Structural search** (mutation/crossover mix, tournament, parsimony, migration): all
  parity-matched (`docs/28`, `docs/29` §A/§B/§D). Not a free lever.
- The **only** remaining legal lever is improving the **inner self-LM solver's convergence
  within the matched 8 iterations × 3 starts** (an allowed-divergence implementation
  improvement, A#1). Measurement 3 indicates `interference` is exploration- not polish-bound,
  so even a perfect optimiser would not obviously close it. Expected headroom: small; pursue
  only with measured evidence (CLAUDE.md #5), not speculatively.

## Decision

1. **No further throughput optimisation for recovery.** Measured zero gain; the binding
   constraint is per-generation search quality, and the generation budget is parity-fixed and
   already exhausted. The shipped SoA/MultiDual stay (they removed the timeouts); the `docs/30`
   SLEEF/vectorisation headroom remains correctly un-pursued.
2. **Recovery parity with PySR is reached** (18/25 = 18/25, symmetric membership). Per CLAUDE.md
   this is the highest-priority goal. The 6 shared-hard failures are an *exceed-PySR* ambition,
   not a parity gap, and are not throughput-addressable.
3. **`interference`** (the one real gap) is exploration-bound within the parity budget; under
   parity there is no clean legal lever. If it is ever revisited, the only sanctioned avenue is
   self-LM convergence quality (A#1), gated behind a measurement that the miss is polish- not
   exploration-bound — which the current evidence argues against.

## Reproduction

```
# Authoritative gate (5 seeds) — note timed_out=FALSE for every row
Rscript benchmarks/02_feynman_gate.R stage=1            # -> feynman_gate_<date>.csv

# Faithful PySR reference (SR.jl engine, 3 seeds)
julia -t 4 benchmarks/05_feynman_pysr_comparison.jl seeds=3

# interference compute-scaling diagnostic (non-parity; generations override via
# symbolic_regression(generations=, timeout_seconds=) on the gate operator set):
#   2800 gens -> miss (1.19e-3);  8400 gens -> 1/3 (seed 1 = 3.96e-5 recovered)
```

See also: `docs/26` §5 (prior throughput framing, now superseded for current code),
`docs/30` (throughput profile + SoA/SLEEF decisions), `docs/31` (cadence fix that removed the
timeouts), `docs/33` (optimiser multi-start parity), `docs/34` (operator-set parity, neg),
`docs/29` §A/§E (category-A implementation differences and their recovery consequence).
