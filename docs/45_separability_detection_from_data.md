# 45. Separability decomposition, Phase 0-E: detection from data alone (NO-GO)

**Date:** 2026-07-03
**Branch:** `feature/high-accuracy-options`
**Status:** Phase 0 complete. Verdict: **NO-GO** — variable-merge separability cannot be
detected reliably from sampled `(X, y)` with a dependency-free surrogate. The docs/44 §6
"GO" was an *oracle ceiling*, not a feasibility result; this screen measures the missing
piece and closes the feature. No product code was built. The data-only detector
(`benchmarks/diag_detect_from_data.R`) is kept as a standing diagnostic.

## 1. Why this screen exists

docs/44 §6 marked separability decomposition **GO** and deferred it to "a separate
implementation plan." Re-examining that verdict before building revealed that the 0-D
oracle (`benchmarks/diag_separability.R`) proved something narrower than feasibility:

- Its merge/separability tests evaluate the **true closed-form formula** `prob$fn`
  (`benchmarks/feynman_datasets.R`) **directly**, at **freshly constructed off-sample
  points** that hold a merged quantity invariant (`merge_test`'s `xt[i] <- x[i]*cc;
  xt[j] <- x[j]/cc`). Detection tolerance is `rel_err < 1e-6`.
- The residual targets it exports are **hand-derived analytic factor-outs** using known
  structure (`export_residual("bose_q_oracle", cbind(s/u), ds$y / u)`; planck
  `ds$y * ds$X[,3]^2`).
- The repo has **no surrogate / interpolator / GP / spline / NN** anywhere; the only
  "query f at a point" capability is that oracle `fn`.

A shipped opt-in feature has none of this — only sampled `(X, y)`. So the 0-D result reads
"**IF** merges can be detected and the exact residual factor is known, **THEN** the residual
is solvable (bose 5/5)." The decision-determining question was never measured: **can the
merges be detected from `(X, y)` alone, reliably enough to fire on the true merges without
false-firing elsewhere?** A false-positive merge on any currently-solved problem would
corrupt a correct result, so detection reliability is a **Correctness** gate, not a
nicety. This screen answers that question directly, per the docs/43–44 "measure the real
blocker before building" discipline.

## 2. Method — `benchmarks/diag_detect_from_data.R`

A detector that consumes **only** `feynman_dataset(key, n, data_seed)$X` / `$y` (plus
variable labels). It never touches `prob$fn`, `prob$formula`, or the true expression
(grep-verified; the secondary method's domain box comes from `apply(X, 2, range)` of the
sample, not `prob$domains`). It replaces the oracle's exact `fn` with a **dependency-free
k-NN surrogate** (base R only) fit from the data, and asks whether a variable merge holds.

**Primary method — split-fit comparison** (self-normalizing; uses only in-sample points,
so it sidesteps the off-sample-oracle problem entirely):

- Surrogate: k-NN regression (z-scored inputs, Euclidean, mean of k=10 neighbours),
  train/test split, accuracy = test NMSE.
- `nmse_full` = surrogate on all original columns. For each pair (i<j) and mode in
  {prod, ratio, sum, diff}, replace columns i,j by the single merged column
  (`xi*xj` / `xi/xj` / `xi+xj` / `xi-xj`) and refit → `nmse_merged`.
- A merge is **DETECTED** iff the surrogate actually fits (`nmse_full < 0.05`, else the
  pair is **UNTESTABLE** — a meaningless comparison) **and** merging costs no accuracy
  (`nmse_merged <= nmse_full * 1.25`).

**Secondary method — surrogate-query invariance**: fit one k-NN surrogate, port
`merge_test`'s constructed points but query the surrogate, and decide by contrasting the
invariance transform against a control transform that *breaks* the merged quantity (since
the surrogate cannot reach the oracle's 1e-6). Kept secondary.

**Ground truth** is the existing oracle, regenerated over all 25 keys
(`benchmarks/diag_separability.R keys=<all>` → `results/separability_oracle.csv`). The
data-only detector's job is to reproduce that PASS/FAIL matrix from `(X, y)` alone.

Runs: n=800, data seeds 42/43/44, plus a noise=1e-3 robustness run. CSVs in
`benchmarks/results/detect_from_data*.csv` (gitignored).

## 3. Results

Primary split-fit sweep (n=800, 3 seeds; vs oracle ground truth):

| metric | value |
|---|---|
| testable rows (surrogate fit `nmse_full < 0.05`) | 516 / 1512 |
| true positives (detected ∧ oracle PASS) | 42 |
| **false positives (detected ∧ oracle FAIL)** | **60 = 12.7% of oracle-FAIL testable rows** |
| false negatives (missed ∧ oracle PASS) | 0 |

The 0% false-negative rate looks encouraging, but the detector **finds** true merges while
being **unable to discriminate** them from false ones:

- **bose_einstein (the headline true positive).** Both true product merges are detected
  3/3 seeds — but the **false `merge_sum` on the same two pairs is also detected 3/3**, with
  overlapping discrimination statistics (verified in `detect_from_data.csv`): for
  `hbar:omega` the true-prod ratio is 0.44–0.55 and the false-sum ratio 0.47–0.64; at seed
  43 the false sum (0.474) even edges out the true product (0.477). **No threshold on
  `nmse_merged/nmse_full` separates true from false.** A decomposition layer reading this
  signal would merge bose's variables by *sum* as readily as by *product*.
- **newtons_grav (positive control).** UNTESTABLE at n=800 (5-var k-NN cannot reach
  `nmse_full < 0.05`; observed 0.078–0.129). Forced to n=6000/k=15 it becomes testable and
  detects the 3 true product merges — but also 10 false ones (≈27% FP for that one problem).
- **False positives are pervasive.** Nearly every testable multi-var problem has ≥1 false
  merge; `rel_mass` fires on **all four** modes for `v:c` though only ratio is true.
- **Noise.** 1e-3 relative noise on y changes nothing row-for-row — negligible against the
  surrogate's own ~2–15% NMSE floor. That floor is itself the point: the whole procedure
  operates orders of magnitude away from the 1e-6 discrimination the oracle relied on.
- **Secondary method**: TP=31, FP=48 (10.1%), FN=11 (26.2%) — similar FP rate, much worse
  recall. Confirms it is not the answer either.

## 4. Verdict — NO-GO, and why it is structural

**Separability decomposition is closed as not shippable from data.** The blocker is not a
tuning problem. Over the benchmark domains many variables share narrow, similar ranges, so
the prod/ratio/sum/diff transforms of a pair are strongly correlated; a coarse
local-averaging surrogate cannot distinguish "genuinely invariant under this merge" from
"y just doesn't move much along this direction in this box." Tightening `slack` only trades
the 0% false-negative rate for false negatives without removing the overlap. Making it work
would require either a **substantially stronger surrogate (a trained NN → a new heavy,
non-portable dependency** on the Rtools/MinGW/UCRT target) or a fundamentally different,
domain-aware statistic — an unbuilt research effort.

Weighed against CLAUDE.md priorities this is not close: the payoff is **two** problems
(bose_einstein, planck; interference is not decomposable — docs/44 §6), PySR parity (the
top configuration goal) is **already met** (18/18), and the cost is a dependency/complexity
burden that hits **Portability** (#3), **Simplicity** (#4), and the Dependency Policy
("default answer is no"), while risking **Correctness** (#1) via false merges on
currently-solved problems. The opt-in high-accuracy layer's clause 2 (measured evidence
before the feature is trusted) is not satisfied — the evidence points the other way.

**What still stands as the delivered high-accuracy option:** the generation budget
(`generations` / PySR `niterations`), documented in docs/44 §3 with its measured
structural effect. Nothing here changes that.

## 5. Decisions

1. **Close** separability decomposition as measured-and-rejected for a shippable feature,
   joining multi-restart (docs/43), seed_expressions / constraints / annealing (docs/44).
   No C++/API/wrapper code was written.
2. **Keep** `benchmarks/diag_detect_from_data.R` as a standing diagnostic (like
   `diag_separability.R` / `diag_structural_audit.R`) so this measurement is reproducible.
3. **Reaffirm** the only opt-in high-accuracy lever with positive evidence to date is the
   documented generation budget (docs/44 §3).
4. The remaining unscreened candidate is **dimensional analysis** (excluded from docs/44 by
   user direction) — a separate future plan, not folded in here.

## 6. Reproduction

```
# oracle ground truth over all keys (uses prob$fn — this is the reference, not the detector):
Rscript benchmarks/diag_separability.R keys=<comma-joined all n_vars>=2 keys>

# data-only detector (X,y only), default sweep + noise run:
Rscript benchmarks/diag_detect_from_data.R n=800 seeds=42,43,44
Rscript benchmarks/diag_detect_from_data.R n=800 seeds=42,43,44 noise=1e-3
```

See also: docs/44 (Phase 0 screen; §6 oracle GO this supersedes for the shippable case),
docs/43 (multi-restart NO-GO), docs/38 (deceptive basin), CLAUDE.md "Opt-in high-accuracy
options".
