# 44. Opt-in high-accuracy candidates: Phase 0 screen (excluding dimensional analysis)

**Date:** 2026-07-03
**Branch:** `feature/high-accuracy-options`
**Status:** Phase 0 complete. Verdicts: budget documentation **GO**;
separability decomposition **GO** (separate implementation plan);
`seed_expressions`, `nested_constraints`/`constraints`, `annealing` **NO-GO**.
A by-product, the **structural-truth audit**, revises how recovery itself is
counted (see §2) and is kept as a permanent diagnostic
(`benchmarks/diag_structural_audit.R`).

## 1. Purpose and method

Following the docs/43 method (measure the ceiling before building), this doc
screens every remaining candidate for the CLAUDE.md opt-in high-accuracy layer,
dimensional analysis excluded by user direction. Targets are the problems
unrecovered at PySR-parity defaults (docs/43): the generation-bound Class A
(center_mass, boltzmann_dist, newtons_grav, harmonic_ke, lorentz_x) and the
structure-inert Class B (planck, bose_einstein, interference; p = 0 at the
parity budget).

Screens run (all timeout-bounded, Stage-1 parity config unless stated):

- **0-A budget curve** (`diag_interference.R`, 5600/14000 gens): completes the
  docs/43 equal-compute data; also closes "does budget crack Class B" (no).
- **0-B reference-engine mechanism screen**
  (`06_srjl_mechanism_screen.jl`, SymbolicRegression.jl at PySR defaults):
  arms baseline / nested_constraints / constraints / annealing / domain-informed,
  3 Class-B problems x 10 seeds. This is a **diagnostic**, not a parity
  comparison: SR.jl is used as a cheap testbed for mechanisms we would otherwise
  have to implement first and measure after.
- **0-C seed-expression oracle** (`standalone/benchmarks/diag_seed_oracle.cpp`
  + internal `SearchOptions::seed_trees` hook, default empty = byte-identical):
  inject partial/full true structure into every island's initial population.
- **0-D separability oracle** (`diag_separability.R` + `diag_residual.R`):
  numeric separability/merge detection on the true functions, then parity-config
  searches on the implied residual datasets.

## 2. The structural-truth audit (methodology by-product, affects everything)

While auditing 0-A we found that the recovery criterion (train NMSE < 1e-4)
conflates two populations:

- **true structure** — the expression IS the formula; NMSE ~1e-32 (all these
  formulas are exactly representable and their constants LM-converge);
- **threshold grinders** — wrong structure (pow-soup with fitted fractional
  exponents, or smooth surrogates such as `c^(log^2(x1/x0))` standing in for
  `2*sqrt(x1*x0)`) polished to just under 1e-4 on the training box. First seen
  for interference in docs/38; it is in fact widespread.

`benchmarks/diag_structural_audit.R` separates them mechanically: it re-evaluates
every recovered expression OUTSIDE the training box (extended domains with
per-problem validity predicates) against the true function. True structure
extrapolates at machine precision; grinders degrade by 2-6 orders of magnitude.
The evaluator replicates the engine's operator semantics exactly (safe sqrt
neg->0, SR.jl-style safe_pow, unparenthesised negative-constant printing), and
self-validates per row by reproducing the recorded train NMSE (rows it cannot
reproduce would be flagged EVAL_MISMATCH; after the semantics fixes there are
none). Verdict bands on extrapolation NMSE: TRUE < 1e-6 <= UNCERTAIN <= 1e-3 <
GRINDER (also GRINDER if > 5% of extrapolation points evaluate non-finite).

Headline corrections (p_true = extrapolation-verified recovery rate; p_thr =
the old threshold rate; full tables in `results/structural_audit_summary.csv`):

| key | 2800 gens (20 seeds) | 5600 (10) | 14000 (10) |
|---|---|---|---|
| newtons_grav   | 0.00 (thr 0.10) | 0.30 (thr 0.60) | **0.90** (thr 1.00) |
| center_mass    | 0.30 (thr 0.40) | — | **0.80** (thr 1.00) |
| boltzmann_dist | 0.10 (thr 0.15) | 0.20 (thr 0.30) | **0.40** (thr 0.60) |
| harmonic_ke    | 0.00 (thr 0.05) | 0.10 (thr 0.30) | 0.10 (thr 0.60) |
| **lorentz_x**  | **0.00** (thr 0.55) | — | **0.00** (thr 1.00) |
| bose_einstein  | 0 | — | 0.00 (thr 0.60 — all grinders) |
| planck         | 0 | — | 0.00 (thr 0.00) |
| clausius_moss  | 0.25 + 12/20 UNCERTAIN (thr 0.95) | — | — |
| lens_eq        | 0.75 (thr 0.75) | — | — |

- **lorentz_x has never been truly recovered** at any budget: on its training
  domain (u/c in [0.1, 0.67]) the relativistic correction is small and smooth
  approximants grind under 1e-4. Its docs/43 classification as "responsive
  p = 0.55" was a threshold artifact; structurally it behaves like Class B.
- The clausius_moss UNCERTAIN mass (extrapolation 1e-6..1e-4, train ~1e-6) looks
  like true rational structure with under-converged constants, not grinding;
  left as UNCERTAIN rather than resolved here.
- **Consequence for the parity gate:** the authoritative 18/25 counts (both
  tools) use the threshold criterion, so both sides are inflated by grinders.
  Any structural-truth restatement of the parity comparison requires running the
  same audit on PySR's expressions first — follow-up, out of scope here.

## 3. 0-A — budget curve (Class A) and Class-B closure

New measurements (10 seeds; planck/bose 5 seeds), threshold and structural:

- Structural truth in the table above. The budget lever is real but smaller than
  the threshold numbers claimed: at 5x budget it lifts newtons_grav 0->0.9,
  center_mass 0.3->0.8, boltzmann_dist 0.1->0.4 — and does nothing true for
  harmonic_ke (0.1) or lorentz_x (0.0).
- Class B stays structurally p = 0 at 14000 gens: planck 0/5 even by threshold;
  bose_einstein's 3/5 threshold "recoveries" are all grinders (no `exp` in any
  of them). Confirms docs/43: budget does not crack Class B.

**Verdict: GO, as documentation only.** The lever already exists (the public
`generations` parameter = PySR `niterations`); no new API is added (an
`accuracy_level` preset would duplicate an existing parameter — Simplicity).
The `generations` docs now state the measured effect and point here.

## 4. 0-B — mechanism screen on the reference engine (Class B)

SR.jl 1.11.0 at PySR documented defaults, one mechanism per arm, 10 seeds,
`timeout_in_seconds=600` (`results/srjl_mechanism_screen_20260703.csv`).
Threshold recovery counts (structural spot-checks in notes):

| arm | interference | planck | bose_einstein |
|---|---|---|---|
| baseline    | 2/10 | 0/10 | 4/10 |
| nested      | 1/10 | 0/10 | 0/10 |
| constraints | 0/10 | 0/10 | 2/10 |
| annealing   | 2/10 (1 true, 3.3e-14) | 0/10 | 1/10 |
| domain      | 0/10 | 0/10 | 0/10 |

No arm exceeds baseline on any problem; the domain-informed constraint arm
(plausibly physicist-written nesting/complexity caps) is strictly harmful.
Annealing's single machine-precision interference hit (1/10) is within the known
SR.jl baseline true rate (~13%, docs/38). GO criterion (lift p=0 to >= 0.2
generic / >= 0.3 domain) is met nowhere.

**Verdict: NO-GO** for `nested_constraints`, `constraints`, and `annealing` as
accuracy levers. (They may still be worth implementing someday for PySR API
compatibility — but not on accuracy grounds, and not now.)

## 5. 0-C — seed-expression oracle (Class B)

Internal hook `SearchOptions::seed_trees` (default empty; the existing
`evolutionary_search` ctest confirms the empty default is a no-op) + throwaway
driver `standalone/benchmarks/diag_seed_oracle.cpp`. 10 seeds x 2800 gens,
parity config, gate training data:

| key | baseline | partial seeds | full true expression |
|---|---|---|---|
| interference  | 0/10 true (1 grinder) | sqrt_prod 0/10 true (2 grinders); sqrt_cos 0/10 true (1 grinder) | 10/10 @ ~1e-29 |
| planck        | 0/10 | expden 0/10; recip 0/10 | 10/10 @ ~1e-29 |
| bose_einstein | 0/10 | expden 0/10 true; recip **1/10 true** | 10/10 @ ~1e-29 |

The decisive observation: **the failure is retention-bound, not
discovery-bound.** Full true expressions are held and polished perfectly
(10/10), but injected *partial* true structure — including the exact
`sqrt(x0*x1)` term interference needs — is discarded by the search; the
deceptive basin (`(I1+I2)*(1+a*cos d)` family and its `c^(log^2(x1/x0))`
smooth surrogates) outcompetes it under the parity loss/parsimony regime.
Grinders even reproduce inside the seeded arms. GO criterion (partial arms
reach p >= 0.5 on any Class-B problem) is missed by a wide margin (best 1/10).

**Verdict: NO-GO** for `seed_expressions` at parity search settings. The
retention-bound diagnosis also tempers expectations for any future
constraint-style lever (incl. dimensional analysis): guiding the search toward
the true region is not enough if the basin ranking discards it; removing the
deceptive basin from the search space (or decomposing it away, §6) is what works.
The internal `seed_trees` hook (5 lines, not exposed in R/Python, byte-identical
when empty) is kept so this experiment stays reproducible; the trade-off
(tiny core addition vs docs reproducibility) is accepted deliberately.

## 6. 0-D — separability decomposition oracle (Class B): the winner

Detection (`diag_separability.R`, numeric tests on the true functions over the
benchmark domains; newtons_grav as positive control — all its expected merges
PASS; matrix in `results/separability_oracle.csv`):

- **bose_einstein**: `hbar:omega` and `kB:T` product-merges PASS -> detectable
  2-var residual `f(s,u) = s/(exp(s/u)-1)`.
- **planck**: `c` multiplicative-separability and `k:T` product-merge PASS
  (`hbar:omega` correctly FAILS — the stray omega^2 factor breaks invariance)
  -> detectable 3-var residual `g(hbar,omega,u)`.
- **interference**: nothing passes. Not decomposable by these tests (its
  compact term is `2*sqrt(I1*I2)*cos(delta)` — no clean variable merge).

Residual solvability at the full parity budget (`diag_residual.R`, 5 seeds,
2800 gens, exact gate data transformed by the oracle chain):

| residual | vars | recovered |
|---|---|---|
| bose_einstein detectable `s/(exp(s/u)-1)` | 2 | **5/5** |
| planck detectable `hbar*omega^3/(pi^2(exp(hbar*omega/u)-1))` | 3 | 0/5 |
| oracle 1-var `q(r) = r/(exp(r)-1)` (both problems) | 1 | **5/5** (also 1/1 at only 100 gens) |

So: merge-detection alone converts bose_einstein from structurally p = 0 to
p = 1.0. planck additionally needs subset-homogeneity detection
(f(c*s, c*u) = c*f(s,u) extracts the `u` factor and lands on `q(r)`, which is
trivially solved); with it, planck's oracle chain also ends at 5/5.
interference remains out of reach for every lever tested in this doc.

**Verdict: GO.** Next step per the approved plan: a separate implementation
plan for an opt-in decomposition layer (R/Python wrapper around the core
search — detect merges/separability numerically on (X, y), recurse on the
reduced problem, recombine; C++ core untouched), including subset-homogeneity
to cover planck. Off by default; when off, nothing changes.

## 7. Decisions

1. **Build**: separability decomposition (opt-in, wrapper layer) — its own plan.
2. **Document**: `generations` as the accuracy-vs-compute lever with the
   structural numbers from §3 (done in the R/Python docs alongside this doc).
3. **Close as measured-and-rejected**: seed_expressions, nested_constraints /
   constraints, annealing (accuracy grounds), joining multi-restart (docs/43).
4. **Adopt**: structural-truth audit as a standing secondary metric; follow-up
   item — run it on PySR-side expressions before restating any parity claim,
   and reconsider lorentz_x's classification (structurally Class B).

## 8. Reproduction

```
# 0-A budget curve (per key/gens):
Rscript benchmarks/diag_interference.R key=newtons_grav seeds=10 gens=14000 timeout=900 tag=bc_newtons_grav_g14000

# structural-truth audit over all result files:
Rscript benchmarks/diag_structural_audit.R

# 0-B (per arm):
julia -t 4 benchmarks/06_srjl_mechanism_screen.jl arms=nested

# 0-C (build target diag_seed_oracle in standalone/build-win, then per key/arm):
./standalone/build-win/standalone/diag_seed_oracle.exe interference sqrt_prod 10 2800 benchmarks/results/seed_oracle_interference_sqrt_prod.csv

# 0-D:
Rscript benchmarks/diag_separability.R
Rscript benchmarks/diag_residual.R csv=benchmarks/data/residual_bose2v_detectable_train.csv seeds=5
```

See also: docs/43 (multi-restart NO-GO, budget-vs-restart), docs/38
(interference deceptive basin), docs/35 (throughput), docs/28-29 (parity spec),
CLAUDE.md "Opt-in high-accuracy options".
