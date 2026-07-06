# 50. Linear scaling (Keijzer 2003) Feynman accuracy screen

**Date:** 2026-07-06
**Status:** COMPLETE.
**Verdict: NO-GO as a recommended accuracy option.** Linear scaling stays a
**documented, experimental opt-in** (`linear_scaling`, off by default), not a
recommended high-accuracy default. It fails the pre-registered GO bar on criterion
1 only:

- **Criterion 1 (≥ 2 additional threshold-majority recoveries): FAIL.** The ON arm
  gains exactly **one** problem (`harmonic_ke`, threshold-majority 1/5 → 5/5); no
  second problem crosses into majority recovery. Net = |gained| − |lost| = 1 − 0 = **+1**.
- **Criterion 2 (no problem loses majority recovery): PASS.** The lost set is empty;
  every problem recovered by majority OFF is still recovered by majority ON.
- **Criterion 3 (no structural regression): PASS.** All 12 problems whose OFF-arm
  structural verdict was majority-TRUE (p_true ≥ 0.6) remain majority-TRUE ON;
  `harmonic_ke` additionally becomes structurally TRUE (p_true 0.0 → 1.0), a genuine
  structural gain — just an isolated one.

The single gain is real and structurally confirmed, but the bar was fixed at ≥ 2
before the run and is honoured as written. The option ships either way (CLAUDE.md
second layer); the evidence here places it in the *experimental* tier, not the
*recommended* tier.

## 1. Setup

- **Option under test.** `linear_scaling` (the Keijzer-2003 best-affine-fit scorer),
  added opt-in in commit 220bff0, off by default. Semantics summary (§1a).
- **Arms.**
  - **OFF baseline (authoritative, PySR-parity):** the docs/47 units-OFF gate,
    `benchmarks/results/feynman_gate_20260704.csv` (25 problems × 5 seeds, 2800
    generations, 300 s timeout, PySR-parity settings). **Reused, not rerun** — see §1b.
  - **ON arm:** `benchmarks/results/feynman_gate_linear_scaling_20260706.csv`
    (same 25 × 5, 2800 generations, 300 s timeout, PySR-parity settings; only
    `linear_scaling` flipped on). Log: `benchmarks/results/_linscale_on_arm.log`.
    Machine: Windows 11, AC power.
- **Judgement.** Threshold recovery (train NMSE < 1e-4, `benchmarks/utils.R`; majority
  = ≥ 3/5 seeds) **and** the extrapolation-based structural audit
  (`benchmarks/diag_structural_audit.R`, `csv=` single-file mode; per-seed bands
  TRUE_STRUCTURE < 1e-6 ≤ UNCERTAIN ≤ 1e-3 < GRINDER; p_true = fraction of seeds
  verified TRUE off the training box). Dual reporting per docs/47 §7.
- **Pre-registered GO bar** (fixed before the run, not adjusted to the data): GO iff
  **all** of — (1) the ON arm recovers ≥ 2 additional problems by the threshold
  criterion (majority) relative to OFF, **and** (2) no problem loses majority
  recovery, **and** (3) no problem whose OFF-arm majority structural verdict was TRUE
  becomes structurally false ON. GO ⇒ recommended high-accuracy option; NO-GO ⇒
  documented experimental opt-in with the evidence.

### 1a. Option semantics (what `linear_scaling` changes)

The scorer minimises the best-affine-fit loss `min_{a,b} sum_i w_i (a·f_i + b − y_i)^2`
(closed-form weighted least squares over the candidate outputs `f_i`; a degenerate
guard — zero variance in `f` — falls back to the best constant fit). The LM constant
optimiser still runs on **unscaled** residuals in this v1 (a documented mismatch
between the fit objective and the optimiser objective). Optimised members are
re-scored through the scaled scorer so the hall-of-fame ordering never mixes scaled
and unscaled costs. At `finalize`, the winning `a·f + b` affine wrap is materialised
into every reported tree (identity-skip tolerance when `a≈1, b≈0`, `reindex_constants`,
fresh HOF rebuild); this can exceed `maxsize` by up to 4 nodes at reporting time. The
option is incompatible with `X_units`/`y_units` (the wrappers reject the combination).
Because the reported expression is a `%.6g` string round-trip of the materialised
tree, a predict-reconstructed loss matches the stored C++ loss only to ~6 significant
figures (the stored C++ loss is itself exact and self-consistent). With scaling every
constant tree ties at the weighted SST, so the complexity-1 bucket is first-seen-wins
(deterministic, harmless).

### 1b. Why the OFF baseline is reused, not rerun

`linear_scaling` defaults to **false**; with the flag off the scorer, the constant
optimiser, and finalize are bit-identical to the code path the docs/47 units-OFF gate
exercised (this is enforced by construction and by the option's parity test). The
docs/47 OFF CSV is therefore the exact OFF arm for this screen, and re-running it
would only reproduce it. The ON arm is the only thing that had to be measured.

## 2. Threshold result (majority recovery, ≥ 3/5 seeds, NMSE < 1e-4)

Per-problem majority recovery (`k` = seeds recovered / 5). **`delta` is ON − OFF in
seed count.** Rows that cross the majority line (≥ 3) are marked.

| problem | OFF (k/5) | ON (k/5) | delta | note |
|---|---|---|---|---|
| gaussian | 5 | 5 | 0 | |
| rel_mass | 5 | 5 | 0 | |
| coulomb | 5 | 5 | 0 | |
| spring_pe | 5 | 5 | 0 | |
| lorentz_x | 4 | 3 | −1 | both majority |
| rel_mom | 5 | 5 | 0 | |
| **harmonic_ke** | 1 | 5 | **+4** | **GAINED** — crosses to majority |
| interference | 0 | 1 | +1 | still sub-majority |
| bohr_radius | 5 | 5 | 0 | |
| larmor_rad | 5 | 5 | 0 | |
| planck | 0 | 0 | 0 | |
| center_mass | 2 | 0 | −2 | still sub-majority |
| torque | 5 | 5 | 0 | |
| larmor_freq | 5 | 5 | 0 | |
| doppler_rel | 5 | 5 | 0 | |
| einstein_smol | 5 | 5 | 0 | |
| driven_osc | 5 | 5 | 0 | |
| heat_conduct | 5 | 5 | 0 | |
| boltzmann_dist | 1 | 2 | +1 | still sub-majority |
| clausius_moss | 4 | 5 | +1 | both majority |
| clausius_moss2 | 5 | 5 | 0 | |
| bohr_magneton | 5 | 5 | 0 | |
| bose_einstein | 0 | 0 | 0 | |
| lens_eq | 4 | 4 | 0 | both majority |
| newtons_grav | 1 | 1 | 0 | |

- **Gained (sub-majority OFF → majority ON):** `{ harmonic_ke }` → **|gained| = 1**.
- **Lost (majority OFF → sub-majority ON):** `{ }` → **|lost| = 0**.
- **Net = +1.**
- **Threshold gate: OFF 18/25 → ON 19/25** (majority). `harmonic_ke` is the only
  problem that changes side of the gate.

Criterion 1 requires ≥ 2 gained ⇒ **FAIL** (only 1). Criterion 2 requires no loss ⇒
**PASS**.

## 3. Structural result (extrapolation audit)

`benchmarks/results/structural_audit_summary_50_on.csv` (ON) vs
`structural_audit_summary_47_off.csv` (OFF). p_true = seeds verified TRUE_STRUCTURE
off the training box / 5. **Majority-structural** = p_true ≥ 0.6.

| problem | OFF p_true | ON p_true | note |
|---|---|---|---|
| gaussian | 1.0 | 1.0 | |
| rel_mass | 0.0 | 0.2 | UNCERTAIN band both arms |
| coulomb | 1.0 | 1.0 | |
| spring_pe | 1.0 | 1.0 | |
| lorentz_x | 0.0 | 0.0 | threshold recoveries are grinders |
| rel_mom | 0.2 | 0.2 | |
| **harmonic_ke** | 0.0 | **1.0** | **structural gain** (matches the threshold gain) |
| interference | 0.0 | 0.2 | seed-1 true recovery (cf. docs/47 §4) |
| bohr_radius | 0.8 | 1.0 | |
| larmor_rad | 1.0 | 1.0 | |
| planck | 0.0 | 0.0 | Class-B wall |
| center_mass | 0.4 | 0.0 | down, but never OFF-majority |
| torque | 1.0 | 1.0 | |
| larmor_freq | 1.0 | 0.8 | 1 seed → EVAL_MISMATCH (see below); still majority |
| doppler_rel | 0.0 | 0.0 | UNCERTAIN both arms |
| einstein_smol | 1.0 | 1.0 | |
| driven_osc | 1.0 | 1.0 | |
| heat_conduct | 1.0 | 0.8 | 1 seed → EVAL_MISMATCH; still majority |
| boltzmann_dist | 0.2 | 0.0 | never OFF-majority |
| clausius_moss | 0.4 | 0.0 | 3 seeds → EVAL_MISMATCH; never OFF-majority |
| clausius_moss2 | 0.0 | 0.4 | up, not majority |
| bohr_magneton | 1.0 | 1.0 | |
| bose_einstein | 0.0 | 0.0 | Class-B wall |
| lens_eq | 0.8 | 0.6 | 1 seed → EVAL_MISMATCH; still majority |
| newtons_grav | 0.0 | 0.2 | |

**OFF structural-majority set (12):** gaussian, coulomb, spring_pe, bohr_radius,
larmor_rad, torque, larmor_freq, einstein_smol, driven_osc, heat_conduct,
bohr_magneton, lens_eq. **Every one stays ≥ 0.6 ON.** Criterion 3 ⇒ **PASS**. ON adds
`harmonic_ke` (0.0 → 1.0), so structural-majority is **12/25 OFF → 13/25 ON**.

**EVAL_MISMATCH is an artifact of the option, not a structural failure.** On some
seeds (larmor_freq 1, heat_conduct 1, lens_eq 1, clausius_moss 3, clausius_moss2 1)
the audit's self-check — which re-evaluates the reported expression on the training
sample and compares to the recorded C++ NMSE — diverges by more than one order of
magnitude and is scored EVAL_MISMATCH rather than verified TRUE. This is the expected
consequence of §1a: the reported string is the `%.6g` round-trip of the materialised
`a·f + b` tree, so re-evaluation reproduces the stored (exact) C++ loss only to ~6
significant figures. It caps the *verifiable* p_true of some affine-materialised
recoveries; it does not indicate a wrong structure. All three affected
structural-majority problems still clear p_true ≥ 0.6, so criterion 3 is unaffected.

## 4. Overall run health

- **ON-arm wall time:** sum of per-run elapsed = **4555 s ≈ 76 min** (log completed
  2026-07-06 21:57; ~1h18m end-to-end including dataset generation), for 125 runs.
- **No timeouts.** The longest single run was **148 s** (clausius_moss seed 5), well
  under the 300 s per-run cap; nothing was truncated. CPU health nominal (compare
  docs: benchmark wall-clock inflates ~50× only under CPU starvation; this run's
  per-problem times track the OFF arm's, so no contamination).
- **Faster than OFF.** Total ON run time (76 min) is *below* OFF (100 min, max run
  118 s) — same observation as docs/47's units arm: the scaled scorer lets
  populations converge on simpler members. This is an observation, not a claim.
- **Gate counts:** ON threshold **19/25** (log: PASS, threshold 18/25); OFF threshold
  **18/25**. Structural-majority **12/25 OFF → 13/25 ON**.

## 5. Reproduction

```
# ON arm (this screen):
Rscript benchmarks/02_feynman_gate.R stage=1 runs=5 linear_scaling nofailfast

# OFF baseline: reused from docs/47 (feynman_gate_20260704.csv); bit-identical to the
# current default code path (linear_scaling defaults false), so not rerun.

# Structural audit of the ON arm (produces structural_audit_{rows,summary}_50_on.csv):
Rscript benchmarks/diag_structural_audit.R \
    csv=feynman_gate_linear_scaling_20260706.csv gens=2800 out=_50_on
```

(On Windows, invoke Rscript by full path `C:\Program Files\R\R-4.6.0\bin\Rscript.exe`;
the R path is required and bash segfaults.)

## 6. What the verdict means for shipping

`linear_scaling` remains **off by default** (default-parity with PySR is unchanged;
this screen touches no default). It ships as a **documented, experimental opt-in**:
the CLAUDE.md second-layer conditions are met (off by default, documented with
measured evidence here, parity preserved when unused), but the pre-registered
recommend-worthy bar (≥ 2 additional recoveries) is **not** met, so it is *not*
promoted to a recommended high-accuracy default.

The measured character of the option, for users who enable it deliberately:

1. **One genuine, structurally-confirmed gain:** `harmonic_ke` goes from 1/5
   threshold / p_true 0.0 to 5/5 threshold / p_true 1.0 — the affine wrap supplies
   the leading `0.25·m·(...)` scale that the unscaled search kept missing on 4/5
   seeds. Isolated, but real.
2. **No majority regressions** (threshold or structural) at the default budget — a
   cleaner profile than the units lever, which regressed driven_osc (docs/47 §3).
3. **Cost:** the `%.6g` materialisation makes some affine recoveries only ~6-sig-fig
   reproducible from the reported string (the stored C++ loss stays exact), and the
   reported tree may exceed `maxsize` by up to 4 nodes. The Class-B wall (planck,
   bose_einstein) and lorentz_x's grinder problem are untouched, consistent with the
   docs/44 §5 retention-bound diagnosis: rescaling the objective does not re-rank the
   deceptive basins.

## 7. Files

- `docs/50_linear_scaling_screen.md` — this document.
- `benchmarks/results/structural_audit_summary_50_on.csv`,
  `benchmarks/results/structural_audit_rows_50_on.csv` — ON-arm structural audit
  (new; results/ is gitignored).
- Inputs (pre-existing): `benchmarks/results/feynman_gate_linear_scaling_20260706.csv`
  (+ `_linscale_on_arm.log`), `benchmarks/results/feynman_gate_20260704.csv`,
  `benchmarks/results/structural_audit_summary_47_off.csv`.
</content>
</invoke>
