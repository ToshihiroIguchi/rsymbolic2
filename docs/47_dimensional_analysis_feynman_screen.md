# 47. Dimensional analysis Feynman accuracy screen (docs/46 §6)

**Date:** 2026-07-04
**Status:** COMPLETE. **Verdict: NO-GO as an accuracy option** (§6) — dimensional
analysis stays shipped as a PySR feature-parity item (docs/46), with no claim that
it raises Feynman recovery. **Collateral finding (GO, documentation):** interference
is budget-responsive after all — p_true 0 → 0.27 at 14000 generations with **no**
units (§4a), correcting docs/44 §3 and extending the `generations` lever's measured
coverage to a fourth problem.

This is the accuracy screen declared (and deliberately deferred) in
`docs/46_dimensional_analysis.md` §6: does turning on the opt-in dimensional-analysis
feature (`X_units`/`y_units`, penalty 1000, constants as wildcards) raise Feynman
recovery, at the frozen Stage-1 PySR-parity budget? It is also the follow-up to the
docs/44 Phase-0 screen, whose one open question was this measurement.

## 1. Setup

- **Units.** Every problem in the 25-equation dev set carries SI unit metadata in
  `benchmarks/feynman_datasets.R` (`units` named vector in `domains` order + `y_unit`),
  derived from the Feynman equations themselves; variables standing in for physical
  constants (eps, hbar, kB, G, ...) carry that constant's SI unit. All 45 distinct unit
  strings parse through `units/unit_parser.hpp`; each true formula is dimensionally
  consistent under its assigned units (hand-verified per problem; the `larmor_freq`
  4th variable is I.34.8's momentum p, `kg*m/s`).
- **Arms.** `benchmarks/02_feynman_gate.R stage=1` (OFF) vs `stage=1 units` (ON); the
  new `units` flag passes the per-problem `X_units`/`y_units` and changes nothing else.
  Both arms: 25 problems × 5 seeds, 2800 generations, 300 s timeout, PySR-parity
  settings pinned in the script. Machine: Windows 11, AC power, idle load checked
  before the run.
- **Judgement.** Threshold recovery (train NMSE < 1e-4, `benchmarks/utils.R`) AND the
  extrapolation-based structural audit (`benchmarks/diag_structural_audit.R`, new
  `csv=` single-file mode; p_true = fraction of seeds whose recovered expression is
  TRUE_STRUCTURE off the training box). The lorentz_x lesson (docs/44 §2): threshold
  alone is not trusted.
- **Pre-registered GO bar** (fixed before running): GO iff some hard-set problem gains
  ≥ +0.2 median p_true AND no recovered problem degrades; otherwise NO-GO.

### Invalid first attempt (recorded as a pitfall)

The first ON arm ran against a stale installed DLL (built 2026-07-02, predating the
dimensional-analysis commit 027476f): `X_units` was accepted by the R wrapper but the
engine ignored it, and all 125 ON runs returned expressions **identical** to the OFF
arm. No error is raised in this failure mode. Detection: expression-level identity
between arms; confirmation: a minimal repro (`y = x0 + x1` with `x0="m"`, `x1="s"`)
returned the dimensionally-violating `x0 + x1` un-penalised. After rebuilding from
HEAD the repro diverges as expected and `test-units.R` passes. Rule going forward:
before any feature screen, compare `packageDescription()$Built` against the target
commit date, and sanity-check that the feature changes a case it must change.

## 2. Phase 1 result: 2800 generations, 25 × 5, ON vs OFF

Result CSVs: `benchmarks/results/feynman_gate_20260704.csv` (OFF),
`feynman_gate_units_20260704.csv` (ON); audits `structural_audit_summary_47_{off,on}.csv`.
The OFF arm reproduces the authoritative 2026-06-27 gate (18/25 threshold PASS): no
regression from the benchmark-side changes. ON differs genuinely (5/125 expressions
identical, the trivial instant-recovery runs).

**Headline: threshold 18/25 (OFF) → 19/25 (ON); structural majority (≥3/5 seeds
TRUE_STRUCTURE) 12/25 (OFF) → 13/25 (ON).**

Problems whose p_true (structural, 5 seeds) moved:

| problem | p_true OFF | p_true ON | note |
|---|---|---|---|
| newtons_grav | 0.0 | **0.4** | largest gain at equal budget |
| center_mass | 0.4 | **0.6** | crosses to structural majority |
| bohr_radius | 0.8 | 1.0 | |
| lens_eq | 0.8 | 1.0 | threshold 4/5 → 5/5 |
| clausius_moss2 | 0.0 | 0.4 | |
| driven_osc | 1.0 | **0.6** | regression: 2 seeds fail entirely |
| boltzmann_dist | 0.2 | **0.0** | loses its single true recovery |
| rel_mom | 0.2 | 0.0 | UNCERTAIN band |
| clausius_moss | 0.4 | 0.0 | threshold up 0.8 → 1.0, all UNCERTAIN |

Unmoved: the Class-B wall stands. planck / bose_einstein / interference /
harmonic_ke stay at p_true = 0. bose_einstein's threshold lift (0/5 → 2/5) is **not**
structural: both expressions are exp-free rational approximants (extrap NMSE 2–5e-4,
UNCERTAIN) — grinders in the making, consistent with docs/44's retention-bound
diagnosis. A welcome side-effect: ON suppresses grinders (lorentz_x threshold 4/5 →
3/5, harmonic_ke 1/5 → 0/5 — all suppressed recoveries were GRINDER verdicts), i.e.
the penalty prunes pow-soup that fakes the threshold.

Secondary observations:

- ON runs are consistently faster in wall-clock (e.g. boltzmann_dist median 89 s →
  27 s): penalised populations stay smaller/simpler. Not a claim, just an observation.
- New audit coverage (these keys were never audited before): rel_mass, doppler_rel,
  clausius_moss2 pass the threshold 5/5 in BOTH arms but are structurally UNCERTAIN on
  every seed (extrap NMSE 1e-6..1e-3) — neither confirmed-true nor grinder. The
  headline "18/25" therefore overstates confirmed structural recovery in both arms
  equally; dual reporting (threshold + structural) is adopted from this doc onward.

**Phase 1 alone is inconclusive against the pre-registered bar**: the gain condition
is met (newtons_grav +0.4, center_mass +0.2) but driven_osc 1.0 → 0.6 violates the
no-degradation condition. At 5 seeds a ±1–2 seed swing is within noise (docs/38), so
the four moved-either-way problems (driven_osc, newtons_grav, center_mass,
boltzmann_dist) are being re-measured at 15 seeds per arm before the verdict.

## 3. Phase 1b: 15-seed differential on the four divergent problems

`runs=15 target=center_mass,driven_osc,boltzmann_dist,newtons_grav`, both arms
(`feynman_gate_diag_20260704.csv` OFF, `feynman_gate_diag_units_20260704.csv` ON;
audits `structural_audit_summary_47_diff_{off,on}.csv`). p_true over 15 seeds:

| problem | OFF | ON | reading |
|---|---|---|---|
| driven_osc | 0.933 | 0.600 | **regression is real** (14/15 vs 9/15 threshold) |
| boltzmann_dist | 0.133 | 0.000 | regression direction confirmed |
| newtons_grav | 0.000 | 0.200 | gain shrinks to the bar's edge (was +0.4 at 5 seeds) |
| center_mass | 0.333 | 0.467 | +0.13, below the bar; Phase-1 gap was small-sample |

**Verdict at the default budget (2800 generations): NO-GO under the pre-registered
bar.** The no-degradation condition fails on driven_osc (whose units are fully
consistent — the penalty reshapes the search, it does not merely filter violators),
and the Phase-1 hard-set gains largely dissolve at 15 seeds.

## 4. Phase 2: units ON × 14000 generations (budget interaction)

Hard set × 5 seeds, `units gens=14000 timeout=1500`
(`phase2_hard_units_g14000_20260704.csv`; audit
`structural_audit_summary_47_p2_on.csv`), against the docs/44 §3 OFF×14000
structural rates:

| problem | p_true OFF×14000 (docs/44) | p_true ON×14000 | 
|---|---|---|
| newtons_grav | 0.9 | 1.0 |
| center_mass | 0.8 | 1.0 |
| boltzmann_dist | 0.4 | 0.4 |
| harmonic_ke | 0.1 | 0.2 |
| planck | 0.0 | 0.0 |
| bose_einstein | 0.0 | 0.0 (3/5 threshold, all exp-free approximants) |
| lorentz_x | 0.0 | 0.0 (5/5 threshold, all grinders) |
| **interference** | 0.0 (0 at 8400; never built in any prior run) | **0.2** |

At 5× budget the ON arm degrades nothing on the hard set and — the one qualitative
event of this screen — **seed 1 built the true interference structure at machine
precision** (extrap NMSE 1.2e-33; the expression simplifies exactly to
`I1 + I2 + 2*sqrt(I1*I2)*cos(delta)`). This is the first true interference recovery
in the project's history (docs/38: 0/23 at parity budget, 0 at 8400 generations;
PySR's own true rate is ~13–20%). Because docs/44 never measured OFF×14000 on
interference, a 15-seed OFF/ON confirmation at 14000 is running to separate a units
effect from a pure budget effect (§4a).

### 4a. Interference confirmation: OFF vs ON × 14000 × 15 seeds

`conf_interference_{off,on}_g14000_20260704.csv`; audits
`structural_audit_{rows,summary}_47_conf_{off,on}.csv`.

| arm | threshold | p_true (TRUE_STRUCTURE) |
|---|---|---|
| OFF × 14000 | 10/15 | **0.267** (4/15) |
| ON × 14000 | 7/15 | **0.200** (3/15) |

**The interference recovery is a budget effect, not a units effect.** The units-off
parity configuration recovers the true structure on 4/15 seeds at 14000 generations
— slightly more than the ON arm, difference within noise. This overturns the
docs/44 §3 claim that budget does not crack interference: that claim extrapolated
from 8400 gens (p_true 0, docs/38); the onset lies between 8400 and 14000, and the
14000-gen rate matches PySR's own true rate (~13–20%, docs/38). docs/44 carries an
addendum pointing here; the `generations` parameter docs (R/Python) now list
interference 0 → 0.27 among the measured 5×-budget effects. planck and
bose_einstein were measured directly at 14000 in docs/44 and stay at p = 0.

## 5. Files touched by this screen

- `benchmarks/feynman_datasets.R` — SI unit metadata (measurement metadata only; data
  generation and the units-off benchmark unchanged).
- `benchmarks/02_feynman_gate.R` — `units` flag; `gens=`/`timeout=` overrides with
  label suffixes so no authoritative CSV is ever overwritten.
- `benchmarks/diag_structural_audit.R` — `csv=`/`gens=`/`out=` single-CSV mode;
  validity predicates for rel_mass/rel_mom/doppler_rel/clausius_moss2 (keys first
  audited here; the frozen docs/44 manifest results are unaffected).
- `r-package/rsymbolic2/R/symbolic_regression.R` (+ regenerated Rd) and
  `python/rsymbolic2/__init__.py` — `generations` docs gain interference 0 → 0.27.
- `docs/44_high_accuracy_phase0_screen.md` — §3 addendum correcting the
  interference budget claim.

No library code (C++ core, R/Python binding behaviour) is touched by this screen.

## 6. Verdict

**Dimensional analysis as an accuracy lever: NO-GO at both budgets.**

- **Default budget (2800):** pre-registered bar failed. driven_osc regresses for
  real (p_true 0.933 → 0.600 over 15 seeds) even though its units are fully
  consistent; the hard-set gains seen at 5 seeds largely dissolve at 15
  (newtons_grav +0.2, center_mass +0.13, boltzmann_dist −0.13). Net effect is
  problem-dependent with both signs: the penalty reshapes the search, it does not
  merely remove invalid candidates.
- **5× budget (14000):** ON degrades nothing on the hard set but adds nothing
  beyond the OFF budget effect (§4/§4a); the apparent Class-B/threshold lifts are
  exp-free approximants, and the interference recovery happens without units.
- The docs/44 §5 retention-bound diagnosis survives another constraint-style
  lever: pruning/penalising the space does not re-rank the deceptive basins that
  block planck, bose_einstein, and (at default budget) interference.

Consequences:

1. `X_units`/`y_units` remain exactly what docs/46 shipped: a PySR
   feature-parity option for users whose data carries physical units — not an
   accuracy feature. The R/Python docs make no recovery claim for it; this doc is
   the measured evidence (CLAUDE.md second-layer condition 2).
2. Observed side-effects worth knowing, not claims: units-ON suppresses threshold
   grinders (lorentz_x 4/5 → 3/5, harmonic_ke 1/5 → 0/5 at 2800; all suppressed
   recoveries were GRINDER verdicts) and runs 2–4× faster in wall-clock
   (penalised populations stay smaller). At fixed generations neither converts
   into recovery.
3. The only proven accuracy lever remains `generations`, now measured to cover
   newtons_grav (0 → 0.9), center_mass (0.3 → 0.8), boltzmann_dist (0.1 → 0.4),
   and interference (0 → 0.27) at 5× budget. Still unreachable by every lever
   screened to date (docs/43–47): planck, bose_einstein (Class B), harmonic_ke
   (~0.1–0.2 flat), lorentz_x (never structurally recovered at any budget).
4. A "hard" generation-time dimensional constraint (reject violating mutations
   instead of penalising) is **not** pursued: the soft version's failure mode at
   2800 is not resource waste but search reshaping, and at 14000 the OFF arm
   already matches ON — there is no measured upside for a harder variant to
   amplify, and docs/44 §4's constraints screen showed hard rejection to be
   harmful in the reference engine.

## 7. Reporting standard adopted

From this screen onward, Feynman results report **both** the threshold count and
the structural (extrapolation-audited) count — e.g. the 2026-07-04 parity gate is
"18/25 threshold, 12/25 structural-majority". The threshold criterion itself is
unchanged (CLAUDE.md forbids weakening it; the structural count is an additional,
stricter lens). Newly audited keys rel_mass, doppler_rel, clausius_moss2 pass the
threshold 5/5 in both arms yet are structurally UNCERTAIN on every seed — neither
confirmed-true nor grinder — which the single-number headline previously hid.
