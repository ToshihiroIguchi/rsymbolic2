# 55. Search-time strong simplification (`strong_simplify`) Feynman accuracy screen

**Date:** 2026-07-22
**Status:** DONE — verdict **NO-GO** (ships as a documented experimental opt-in,
default OFF; see §6).

## 1. Setup

- **Option under test.** `strong_simplify` (opt-in, off by default, all bindings:
  R/Python/WASM). It is an "opt-in high-accuracy option" per CLAUDE.md's second
  layer: a deliberate divergence from PySR's SR.jl simplifier, applied *during* the
  search rather than only once at result finalisation. The simplifier itself
  (`display_simplify()`, two-layer Cohen normalisation + a bounded e-graph) is
  unchanged — see `docs/54` for its design, rule set and finalisation-time use. This
  screen concerns only the new, additional, in-loop application of that same
  simplifier.

  When `strong_simplify` is on, `optimize_and_simplify_population` additionally
  offers each population member's weak-simplified tree to `display_simplify()` under
  a small deterministic budget (see below) once per optimisation pass, and the
  rewrite is **adopted** — replacing the member in place — only when **both**
  adoption gates pass:

  1. **Strictly smaller.** The candidate rewrite must have fewer nodes than the tree
     it would replace. A same-size or larger rewrite is discarded; the original
     member is left untouched.
  2. **Operator-set membership.** Every operator the rewrite introduces must already
     be enabled in the search's `unary_ops`/`binary_ops`. `display_simplify` can
     introduce `neg`, `square`, or `abs` as normalisation byproducts (e.g. folding
     `x - y` toward a canonical form, or collapsing `x * x`) even when the user did
     not request those operators. Per CLAUDE.md, the enabled operator set is the
     shared problem input given identically to rsymbolic2 and PySR, not something a
     display-layer rewrite is allowed to silently expand; a rewrite that would add an
     operator outside the requester's set is rejected and the original member is kept.

  With `strong_simplify = false` (the default) the hook is an untaken branch:
  `optimize_and_simplify_population` never calls the extra `display_simplify()` pass,
  and the search is byte-identical to the PySR-parity path — enforced by construction
  and exercised by the standalone bit-identity gate (§1b).

  **Deterministic budget.** The in-loop `display_simplify()` call uses a fixed
  candidate-limit triple `{max_iterations, max_enodes, max_millis}` =
  `{4, 1000, 1.0e9}` (`kSearchStrongSimplifyLimits` in `evolutionary_search.cpp`).
  `max_millis` is neutralised with a huge sentinel so only `max_iterations` /
  `max_enodes` can stop the e-graph saturation loop — the call never has a live
  wall-clock budget of its own, keeping the search's timing behaviour governed
  entirely by the outer `timeout_seconds`, if any.

  This triple was chosen from a micro-benchmark over 6000 random ~30-node trees
  spanning the full operator set, timing `display_simplify` per call at three
  candidate-limit settings:

  | Candidate limits | p50 | p90 | max | shrink rate |
  |---|---|---|---|---|
  | {10, 10000, 1e9} | 0.0068 ms | 5.25 ms | 74.6 ms | 51.2% (3074/6000) |
  | {6, 2000, 1e9} | 0.0062 ms | 1.10 ms | 11.5 ms | 51.2% (3073/6000) |
  | {4, 1000, 1e9} | 0.0062 ms | 0.41 ms | 1.59 ms | 51.1% (3069/6000) |

  `{4, 1000}` was chosen as the smallest candidate-limit setting with a sub-millisecond
  p90 and a shrink rate statistically indistinguishable from the two larger settings
  (51.1% vs 51.2%, a 5-tree difference out of 6000) — i.e. tightening the budget this
  far costs essentially no simplification power while cutting the worst-case call cost
  by ~47x (74.6 ms -> 1.59 ms) and the p90 by ~13x. This keeps the extra in-loop call
  cheap enough to run on every optimisation pass without materially changing the
  search's wall-clock profile.

### 1b. Why the OFF baseline is reused, not rerun

`strong_simplify` defaults to **false**; with the flag off, `optimize_and_simplify_population`
never invokes the extra in-loop `display_simplify()` call, so the evolutionary
trajectory is bit-identical to the pre-existing PySR-parity code path. This is
enforced by construction (the call site is behind an `if (options.strong_simplify)`
branch) and verified by the standalone C++ gate in
`standalone/tests/test_strong_simplify.cpp`, which asserts OFF-path bit-identity
against the pre-existing search trajectory. The existing authoritative OFF-arm
Feynman CSV (the docs/47 units-OFF gate, reused by docs/50 for the same reason) is
therefore the exact OFF arm for this screen too, and re-running it would only
reproduce it. Only the ON arm needs to be measured.

## Pre-registered GO/NO-GO bar

Fixed before the screen is run; copied verbatim so it cannot be adjusted to the data
after the fact:

> GO iff ALL of: (1) the ON arm gains >= 2 additional majority (>= 3/5 seeds)
> threshold recoveries (NMSE < 1e-4) over OFF; (2) no problem loses majority
> threshold recovery; (3) no structural regression (no problem whose OFF-arm
> majority structural verdict was TRUE_STRUCTURE becomes structurally false ON);
> (4) no strong_simplify-induced 300 s timeout (a timeout is a health failure, not
> scored). GO => recommended high-accuracy option. NO-GO => documented experimental
> opt-in (ships in all bindings, default OFF, not recommended for routine use — same
> treatment as linear_scaling/docs/50).

## 2. Threshold result

ON arm: `feynman_gate_strong_simplify_20260722.csv` (25 keys x 5 seeds, stage-1
config: pop=27, islands=31, gens=2800, scaling=1040, 300 s/run). OFF arm reused
from `feynman_gate_20260704.csv` (see §1b). Majority = >= 3/5 seeds recovered at
NMSE < 1e-4.

Majority recovery is unchanged on 23 of 25 keys. Only these move (both directions
shown; keys where neither arm reaches majority are listed for completeness because
they carry the effect signal even below threshold):

| key | OFF seeds | ON seeds | OFF maj | ON maj | change |
|---|---|---|---|---|---|
| center_mass    | 2/5 | 4/5 | –   | MAJ | **majority GAIN** |
| clausius_moss  | 4/5 | 5/5 | MAJ | MAJ | (both majority) |
| lorentz_x      | 4/5 | 3/5 | MAJ | MAJ | (both majority; known false-recovery, §3) |
| interference   | 0/5 | 2/5 | –   | –   | up, still sub-majority |
| harmonic_ke    | 1/5 | 0/5 | –   | –   | down, both sub-majority |
| boltzmann_dist | 1/5 | 0/5 | –   | –   | down, both sub-majority |

Aggregate: OFF has **18** majority-recovered keys, ON has **19**. Net threshold
change = **+1 majority GAIN (center_mass), 0 majority LOSS.**

## 3. Structural result

Structural audit (`diag_structural_audit.R`, extrapolation-verified `p_true`) on
both arms: `structural_audit_summary_55_on.csv` and `..._55_off.csv`. A key is a
structural-majority recovery when `p_true >= 0.6`. Keys whose structural verdict
moves across that line:

| key | OFF p_true | ON p_true | note |
|---|---|---|---|
| center_mass   | 0.4 | 0.6 | **structural GAIN** |
| clausius_moss2| 0.0 | 0.6 | **structural GAIN** (threshold 5/5 in both arms, but OFF's recoveries were structurally false; ON's are structurally true) |

No key regresses structurally (no OFF `p_true >= 0.6` key drops below 0.6 ON).
`lorentz_x` stays `p_true = 0.0` in both arms — its 3–4/5 threshold "recoveries"
are the known extrapolation-false basin from docs/44, not a real structure, so the
one-seed threshold drop (4/5 -> 3/5) is not a structural loss. So criterion (3) of
the GO bar (no structural regression) is **met**, and structurally the ON arm is
mildly *better* (2 structural gains, 0 structural regressions).

## 4. Run health

125 runs, all with `timed_out = FALSE`; **no strong_simplify-induced 300 s
timeout** (criterion (4) met). Total ON-arm wall time 137.0 min; per-run mean
65.8 s, max **150 s** (about half the 300 s cap). The extra in-loop simplification
is genuinely active, not inert: aggregate adoption telemetry over the whole arm is
**2,250,759 adopted / 6,359,526 attempts = 35.4 %** — i.e. the two adoption gates
(strictly-smaller AND operator-set membership) accept about a third of offered
rewrites, per-key adoption ranging ~25–40 %. The operator-set gate is doing real
work rather than rejecting everything (which would have made the option a no-op).

## 5. Reproduction

```powershell
# Preflight: confirm the installed package is built from the current HEAD (a stale
# DLL silently no-ops new options rather than erroring).
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" -e "print(packageDescription('rsymbolic2')$Built)"

# CPU-health probe (benchmark wall time is contaminated ~50x under CPU starvation;
# confirm cpu/wall is healthy before trusting timings) — see benchmarks/utils.R.

# 1-seed sanity pass (de-risks the full 5-seed gate):
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" benchmarks/02_feynman_gate.R stage=1 runs=1 strong_simplify

# Full 5-seed ON arm:
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" benchmarks/02_feynman_gate.R stage=1 runs=5 strong_simplify nofailfast

# Structural audit of the ON arm (produces structural_audit_{rows,summary}_55_on.csv):
& "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" benchmarks/diag_structural_audit.R csv=<csv> gens=2800 out=_55_on
```

(On Windows, invoke Rscript by full path; the R path is required and bash segfaults
R on this machine.)

## 5a. Verdict: NO-GO

Applying the pre-registered bar to the results above:

| criterion | requirement | measured | pass |
|---|---|---|---|
| (1) additional majority threshold recoveries | >= 2 | **1** (center_mass) | **FAIL** |
| (2) no majority threshold recovery lost | 0 lost | 0 lost | pass |
| (3) no structural regression | none | none (2 structural gains) | pass |
| (4) no strong_simplify-induced 300 s timeout | none | none (max 150 s) | pass |

Criterion (1) requires >= 2 additional majority threshold recoveries; the ON arm
delivered **1** (center_mass). The bar is not met -> **NO-GO**.

The honest secondary signal: `strong_simplify` is not harmful and is weakly
*positive* — it added one threshold-majority recovery and two structural-majority
recoveries (center_mass, clausius_moss2) with zero regressions of either kind, and
it improved several sub-majority problems (interference 0/5 -> 2/5). But the effect
is below the pre-registered threshold bar, and — per the docs/43–50 discipline — the
bar is fixed before the run and is **not** relaxed to fit a favourable-looking but
sub-threshold result. Promising-but-under-bar is exactly the case the "documented
experimental opt-in" outcome exists for.

## 6. What the verdict means for shipping

Per CLAUDE.md's second layer, `strong_simplify` ships in all bindings (R/Python/WASM)
with **default OFF** — the default search remains byte-identical to the PySR-parity
path, this screen touched no default. Because the verdict is **NO-GO**, it ships as
a *documented, experimental* opt-in, **not** a recommended high-accuracy option: the
same treatment `linear_scaling` received in docs/50. The measured evidence for
anyone considering turning it on is this document: a small, mixed accuracy effect
(+1 threshold / +2 structural majority recoveries, 0 regressions) bought at ~1.3x the
OFF-arm wall time on the hard problems, with no ground-truth-recovery justification
strong enough to recommend it by default.

## 7. Files

- `docs/55_search_time_strong_simplification_screen.md` — this document.
- `r-package/rsymbolic2/src/rsymbolic/search/evolutionary_search.hpp` —
  `SearchOptions::strong_simplify` (semantics comment) and the
  `n_strong_simplify_attempts`/`n_strong_simplify_adopted` `SearchResult` counters.
- `r-package/rsymbolic2/src/evolutionary_search.cpp` — `kSearchStrongSimplifyLimits`
  and the `optimize_and_simplify_population` adoption-gate implementation.
- `standalone/tests/test_strong_simplify.cpp` — OFF-path bit-identity gate and
  adoption-gate unit coverage.
- `benchmarks/02_feynman_gate.R` — `strong_simplify` args flag
  (`STRONG_SIMPLIFY_ON`), `_strong_simplify` CSV label suffix, and the per-run
  `strong_simplify_attempts`/`strong_simplify_adopted` CSV columns.
- Screen outputs: `benchmarks/results/feynman_gate_strong_simplify_20260722.csv`
  (ON arm, 125 runs) and `benchmarks/results/structural_audit_{rows,summary}_55_on.csv`.
- OFF arm reused from `benchmarks/results/feynman_gate_20260704.csv` (docs/47, docs/50;
  see §1b), with a matched structural audit written to
  `benchmarks/results/structural_audit_{rows,summary}_55_off.csv` for the §3 comparison.
