# 38. The `interference` gap, examined: small-sample noise + a deceptive basin, not a clean capability gap

**Date:** 2026-06-27

## Purpose

`docs/35` concluded the sole remaining Feynman recovery gap (`interference`: PySR recovers,
rsymbolic2 does not) is "exploration-bound" and that rsymbolic2 needs ~3× the generation
budget to find "the same backbone family PySR uses." That conclusion rested on a **throwaway,
non-parity, 1–3-seed diagnostic** and was never decomposed. Challenged on whether a *small*
allowed-divergence (self-LM/Float64/RNG) could really cause a 3× generation requirement — and
whether it was actually examined — this file does the examination properly: many-seed NMSE
**distributions** at the parity config for both tools, plus **structural inspection** of the
Pareto fronts. The result **corrects `docs/35`** on three points.

## Method

New committed diagnostic `benchmarks/diag_interference.R`: a verbatim copy of the Stage-1
dev-gate parity hyperparameters (`02_feynman_gate.R BENCH_PARAMS`: pop=27, islands=31,
tournament=15, gens=2800, scaling=1040, optprob=0.14, maxsize=30, neg-free operator set),
overriding only `generations` and the seed count from the CLI, and capturing per run the full
Pareto front (complexity, loss→NMSE, expression) and the front-minimum expression. Sanity:
seed 1 @2800 reproduces the authoritative gate NMSE **1.19e-3 bit-for-bit**, confirming the
installed build matches the gate. PySR side: `05_feynman_pysr_comparison.jl seeds=15
only=interference` (SR.jl 1.11.0, Julia 1.12.6, faithful PySR defaults: `precision=32`,
`annealing=false`). Target: `interference` = `I1 + I2 + 2·√(I1·I2)·cos(δ)` (x0=I1, x1=I2,
x2=δ; ranges [1,5]³; operators add/sub/mul/div/pow + exp/log/sin/cos/sqrt/tanh/abs/square).
Recovery: NMSE < 1e-4 (`utils.R`, unchanged).

## Measurement 1 — many-seed distributions at the parity budget (refutes the 2/3 framing)

| seed | rsymbolic2 @2800 NMSE | PySR @parity NMSE |
|-----:|----------------------:|------------------:|
| 1  | 1.19e-3 | 2.18e-3 |
| 2  | 1.62e-3 | **9.01e-5 ✅** |
| 3  | 2.53e-4 | **3.45e-14 ✅** |
| 4  | 2.41e-3 | 3.03e-4 |
| 5  | 7.79e-4 | 2.28e-3 |
| 6  | 1.04e-3 | **2.32e-11 ✅** |
| 7  | 2.45e-3 | 3.06e-4 |
| 8  | 1.04e-3 | 3.11e-3 |
| 9  | 2.70e-4 | 4.62e-4 |
| 10 | 1.34e-3 | 1.36e-4 |
| 11 | **1.57e-4** | 1.12e-4 |
| 12 | 5.36e-4 | 2.29e-3 |
| 13 | 2.56e-3 | 1.78e-3 |
| 14 | 2.34e-3 | 2.64e-3 |
| 15 | 2.34e-3 | 5.62e-4 |
| **recovered** | **0/15** | **3/15 (20 %)** |
| **median** | 1.19e-3 | 4.62e-4 (*a fail*) |
| **min** | 1.57e-4 | 3.45e-14 |

- **PySR's `interference` recovery is ~20 %, not reliable.** The `docs/26`/`docs/35` head-to-head
  saw PySR 2/3 — a **small-sample fluke**. At 15 seeds PySR's *median* NMSE (4.62e-4) is itself a
  failure, and 12/15 PySR runs miss, several right at the threshold (1.12e-4, 1.36e-4).
- rsymbolic2 is 0/15 but its **best (1.57e-4) is 1.57× the threshold** — a near miss, and its
  distribution overlaps PySR's failing mass (both span ~1e-4…3e-3).

## Measurement 2 — structural inspection (the real, but narrow, difference)

PySR's three "recoveries" split into two mechanisms:

- **Exact compact structure (seeds 3, 6 — 2/15 ≈ 13 %):**
  - seed 3 (3.45e-14, **cx=13**): `(sqrt(x1·(x2·4))·cos(x3)) + x1 + x2` = `2·√(I1 I2)·cos δ + I1 + I2`.
  - seed 6 (2.32e-11, **cx=16**): `((x2·x1)^0.49999·2.00002)·cos(x3) + x2 + |x1|` — same exact form via `^0.5`.
- **Deceptive-basin threshold-crossing (seed 2, 9.01e-5):** nested-`abs` junk that merely dips under 1e-4.

The other 12 PySR misses and **all 15 rsymbolic2 runs** sit in the **same deceptive basin**:
`(I1+I2)·(1 + α·cos δ)` (multiplicative factoring), e.g. PySR seed 7
`(x2+x1)·((cos(x3)·…)+0.9996)` and rsymbolic2 seed 6
`(x1+x0)·((tanh(exp(…))^cos x2)+cos x2/1.009)`. This basin captures most variance by **cx≈8**
(`(1.017+cos x2)·(x0+x1)`, NMSE 6.8e-3) and then plateaus: the true coefficient
`2√(I1 I2)/(I1+I2)` is approximated by ever-deeper junk of the ratio x1/x0
(`tanh(x0^square(x0^log(tanh(x1/…))))`), bloating to cx 21–30 while NMSE only crawls 6.8e-3→1.2e-3.

**A grep of all 15 rsymbolic2 fronts finds no clean `√(I1·I2)` / `(I1·I2)^0.5` term.** rsymbolic2
does not discover the compact exact structure at the parity budget; PySR does so ~13 % of runs.

## Measurement 3 — 8400 generations: rsymbolic2 grinds the wrong basin, it does not find the structure

8 seeds at 3× the budget (`tag=p2_8400`, full front captured):

| seed | NMSE @8400 | recovered | best-expression basin |
|-----:|-----------:|:---------:|-----------------------|
| 1 | 3.96e-5 | ✅ | `(…tanh(…)·cos x2+0.995)·(x1+x0+0.06)` — **deceptive basin** |
| 7 | 1.03e-5 | ✅ | `(x0+x1)·(cos x2+((0.885^…)^square(log x1−log x0))^cos x2)` — **deceptive basin** |
| 3 | 1.88e-4 | ❌ | deceptive basin |
| 6 | 2.08e-4 | ❌ | deceptive basin |
| others (2,4,5,8) | 3.9e-4…1.9e-3 | ❌ | deceptive basin |

**2/8 recovered — but both recoveries are the deceptive `(I1+I2)·(1+α·cos δ)` basin polished hard
enough to dip under 1e-4, not the compact `√(I1 I2)` structure.** A grep of the whole 8400 front
again finds **no clean `√(I1·I2)` term**. So `docs/35`'s reading — "the 8400 backbone is the same
family PySR uses; rsymbolic2 is not blind to the structure" — is **wrong**: rsymbolic2 never builds
the compact form; its extra-generation "recoveries" are brute-force polishing of the *wrong* basin
until it crosses the line. More generations raise that grind-across rate (0/15 @2800 → 2/8 @8400),
which is why a "3×" effect appears at all — but the mechanism is polish-of-deceptive-basin, not
discovery of the true structure.

## Findings (corrections to `docs/35`)

1. **No clean capability gap; the 2/3 was noise.** PySR recovers `interference` ~13–20 % of the
   time (exact structure ~13 %), not reliably. Both tools predominantly fall into the *same*
   deceptive basin. The "18 vs 18, PySR wins interference / rsymbolic2 wins lorentz_x" 1-for-1 swap
   in `docs/35` is **within seed noise** at this problem, not a stable membership difference.

2. **The genuine, narrow difference is structural-discovery luck.** PySR's mutation/selection
   trajectory occasionally (~13 %) assembles the compact `2√(I1 I2)·cos δ + I1 + I2` (cx 13–16,
   then trivially polished to 1e-11…1e-14). rsymbolic2 did so in **0/23** parity-config runs
   (15 @2800 + 8 @8400). This is an **exploration / structural-discovery** difference, **not**:
   - **polish-bound** — when the structure is found it polishes to ~1e-14 instantly; and
     rsymbolic2's *own* recoveries show it can already polish the harder deceptive basin to 1e-5.
     So constant-optimiser effort (the hardcoded `optimizer_nrestarts=2`/`iterations=8`, and the
     self-LM-vs-BFGS axis) is **not** the bottleneck — the planned optimizer-effort ablation
     (Phase 4) is therefore **not warranted** and was not run.
   - **Float32 vs Float64** — Float64 is the *more* precise core; PySR finds the structure *in
     Float32*. Precision cannot explain a structure rsymbolic2 fails to build. (`docs/36` already
     showed Float32 only worsens the loss floor.) Ruled out.
   - The residue is the `docs/29` §A/§E **category-A** trajectory difference (RNG stream + the
     concrete structural-mutation dynamics), which CLAUDE.md *permits* as an implementation
     divergence. It changes *which* basins get built, not the matched settings.

3. **There is no parity-legal lever.** `generations` is parity-fixed at 2800; raising it is the
   only thing shown to move the (wrong-basin) recovery rate and it is a parity violation usable
   only as a diagnostic. Optimizer effort is a must-match PySR setting *and* (per finding 2) not
   the bottleneck. Structural-search settings are already parity-matched (`docs/28`/`docs/29`).
   Improving compact-structure discovery without changing settings would require an
   implementation-level change to the structural-mutation/RNG dynamics whose *expected* benefit is
   one noise-level problem — not worth pursuing speculatively (CLAUDE.md #5).

## Decision

- **`interference` is not evidence of an rsymbolic2 recovery deficit.** It is a deceptive
  multi-variable problem both tools usually miss; PySR's edge is a ~13 % chance of stumbling onto
  the compact `√(I1 I2)` form, which is seed-luck within the allowed category-A trajectory
  divergence. No fix is warranted; chasing it would be an *exceed-PySR* ambition on a single
  noise-level problem, not a parity obligation.
- **`docs/35` is corrected** by an added note: the far-miss is not "same-family, exploration-bound,
  3×-generations"; it is a deceptive-basin trap where rsymbolic2's only path across the line is
  polishing the wrong basin, and PySR's recovery is itself unreliable (~13–20 %).
- The earlier symmetric "18 = 18 with a 1-for-1 swap" headline (`docs/35`) should be read as
  "≈18 each, the swap inside seed noise," not as a stable per-problem difference.

## Reproduction

```
# rsymbolic2, parity config, full Pareto front captured (set OMP_NUM_THREADS to physical cores):
Rscript benchmarks/diag_interference.R seeds=15 gens=2800 timeout=200 tag=p1_2800
Rscript benchmarks/diag_interference.R seeds=8  gens=8400 timeout=420 tag=p2_8400
# PySR (SR.jl engine), faithful defaults:
julia -t 4 benchmarks/05_feynman_pysr_comparison.jl seeds=15 only=interference
```

Artefacts: `benchmarks/results/diag_interference_p1_2800_{runs,front}.csv`,
`diag_interference_p2_8400_{runs,front}.csv`, `sr_comparison_feynman_20260627.csv`.

See also: `docs/35` (superseded framing of this problem), `docs/36` (BFGS/Float32 measured),
`docs/29` §A/§E (category-A implementation differences).
