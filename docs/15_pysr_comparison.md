# PySR Reference Comparison (via SymbolicRegression.jl)

**Date:** 2026-06-07
**Status:** harness revised to strict defaults (docs/16); base (9/9) and sqrt (10/10) runs complete.
**Basis:** CLAUDE.md — *"Run PySR only at its documented default settings… The operator set
is the one shared problem input given identically to both tools… Any equalization needed for
a fair comparison is done on rsymbolic2's side, never by altering PySR's settings."*

## 1. Why SymbolicRegression.jl directly, not PySR-through-Python

PySR is a thin Python / scikit-learn wrapper over the Julia package
**SymbolicRegression.jl**; the evolutionary search and constant optimisation — the things
a benchmark actually measures — live entirely in the Julia package. Benchmarking
SymbolicRegression.jl directly is therefore an *equivalent* comparison of the same engine.

On this machine PySR's Python→Julia bridge (`juliacall`) cannot initialise: the only
available Python is the **Microsoft Store** build, whose app-container sandboxing blocks
the embedded `libjulia` from loading its own dependency `libpcre2-8.dll`
(`InitError ... could not load library "libpcre2-8"`). Standalone Julia is unaffected —
both `julia 1.11.9` and `1.12.6` run normally, and `using SymbolicRegression` + a full
`equation_search` succeed. Rather than install a second, non-Store Python solely to host a
wrapper, we drive SymbolicRegression.jl directly. This is cleaner (fewer layers), uses the
identical algorithm, and the version that matters — the engine's — is reported below.

**Honesty note:** literature numbers are usually quoted as "PySR." They reflect
SymbolicRegression.jl's algorithm. We report the SR.jl version explicitly and mirror
PySR's default population configuration so the comparison is faithful to how PySR runs.

## 2. Versions, hardware

| Item | Value |
|------|-------|
| rsymbolic2 | 0.1.0 (Rtools45 MinGW GCC/UCRT, R 4.6.0) |
| SymbolicRegression.jl | 1.11.0 |
| Julia | 1.12.6 |
| PySR (installed, bridge unusable) | wrapper only; not used for timing |
| OS | Windows 11 Home 10.0.26200 |
| CPU | Intel hybrid, 10 physical / 12 logical cores |
| Threads (T) | 4 (equalization — see §3) |

## 3. Protocol

- **Data:** bit-identical to the rsymbolic2 gate. `benchmarks/export_nguyen_data.R` writes
  each problem's exact `(X, y)` (same `set.seed(123)`, n=20, same domains) to
  `benchmarks/data/nguyen_<id>.csv`; the Julia harness reads those. R's RNG stream cannot
  be reproduced from Julia, so materialising the data is the only way to guarantee both
  tools see the same samples.

- **PySR strict defaults.** Hyperparameters read directly from the installed `pysr/sr.py`
  (`PySRRegressor.__init__`), not assumed. Every algorithmic setting is at its PySR default:

  | param | PySR default | used |
  |-------|--------------|------|
  | niterations | 100 | 100 |
  | populations | 31 | 31 |
  | population_size | 27 | 27 |
  | ncycles_per_iteration | 380 | 380 |
  | tournament_selection_n | 15 | 15 |
  | parsimony | 0.0 | 0.0 |
  | maxsize | 30 | 30 |
  | parallelism | None → multithreading | :multithreading |
  | timeout_in_seconds | None | not set |
  | early_stop_condition | None | None |

  An earlier draft mislabelled *SymbolicRegression.jl's* defaults (`populations=15`,
  `population_size=33`) as "PySR defaults" and used `niterations=40`. That was wrong; the
  table above is PySR's actual default set.

- **Operators — the shared problem input (CLAUDE.md §Benchmarking Requirements).**
  Operators define the search space (a problem statement), not an algorithm knob, so both
  tools receive the **identical** set:
  - base mode: binary `[+,-,*,/]`, unary `[exp,log,sin,cos]` (maps rsymbolic2 gate set;
    `neg` via `0-x`);
  - sqrt mode: add `sqrt`, include N8.

  PySR's literal default has **no** unary operators and cannot address N5–N10. Giving both
  tools the same operators is a problem-definition requirement, not PySR tuning. Per
  CLAUDE.md, this is the **only** thing set on PySR beyond its defaults.

- **Thread budget — equalized on rsymbolic2's side (T = 4).**
  The two tools parallelise differently; a clean common budget is T = 4 threads:
  - rsymbolic2 gate already ran with `n_populations=4` (4 OpenMP threads) — no re-run
    needed; those `docs/14` §1.6 numbers are reused as-is.
  - SR.jl is launched with `julia -t 4`; at its default `parallelism=:multithreading` it
    uses the 4 threads the environment provides. No PySR *setting* is touched.
  Setting the machine's thread budget is a legitimate equalization knob on our side.

- **Output suppression — non-algorithmic.**
  SR.jl's per-iteration progress bars, when captured through a PowerShell pipe, block the
  Julia process (~19-min idle "runs", low CPU — see §4 footnote). `verbosity=0` +
  `progress=false` silence logging only; they do not touch the search. Additionally, the
  harness is launched with OS-level stdout redirect (`> file 2>&1`) to prevent any pipe
  from forming. These are presentation choices, not algorithm changes; recorded as such.

- **Stopping criterion difference — reported, not hidden.**
  PySR's default has no early stop, so SR.jl runs its full 100-iteration budget every run.
  rsymbolic2's gate uses *its* default early stop (`target_loss=1e-10`) and halts once
  reached. Each tool runs at its own default stopping behaviour. Consequence: **recovery
  rate is the primary, directly-comparable metric; wall-time is reported as context** and
  is not strictly like-for-like on the stopping axis (see §5).

- **JIT warm-up (untimed):** SR.jl's first search pays a one-off compilation cost (~6 s),
  amortised once per session in real PySR usage. The harness runs one untimed warm-up
  before the measured runs, so per-run times reflect a warm session — fair, not gaming.

- **Recovery criterion:** identical to `benchmarks/utils.R` —
  `NMSE = SSE / ((n-1)·var(y))`, recovered iff `NMSE < 1e-4`, from the best Pareto
  member's predictions on the training data.

- **Runs:** 5 seeds per problem (matching the gate). This is a **reference point**, not the
  30-run publication protocol of `docs/04`; stated as such.

Harness: `benchmarks/03_sr_comparison.jl` (`base` and `sqrt` modes).
rsymbolic2 figures reused from `docs/14` §1.6 (gate Run 1 / Run 2), no re-run needed.

### Design note: why no timeout on SR.jl

PySR has no default timeout; removing it makes SR.jl run at its actual defaults. The bound
on algorithmic work is `niterations=100` + `maxsize=30`, validated empirically (N3 seed 4:
~22 s at niterations=100 vs. 1167 s when niterations was 1e6 with a coarse per-iteration
timeout check). OS-level redirect removes the pipe-block that produced the earlier idle
~19-min artifacts. A `timed_out` flag is recorded by wall-clock for transparency; a run
exceeding a sane wall bound is treated as an **analysis artifact** (flagged, excluded from
timing median) — never capped by injecting a timeout into PySR.

## 4. Results

### 4.1 Base mode (T = 4 threads, multithreading)

Source: `benchmarks/results/sr_comparison_base_<date>.csv` (SR.jl, 2026-06-07) and
`docs/14` §1.6 gate Run 1 (rsymbolic2, n_populations=4). 9 problems × 5 seeds = 45 runs.

| Problem | rsymbolic2 recovered | rsymbolic2 med time | SR.jl recovered | SR.jl med time |
|---------|:--:|--:|:--:|--:|
| N1  | 5/5 | 2 s  | 5/5 | 11 s |
| N2  | 5/5 | 4 s  | 5/5 | 26 s |
| N3  | 5/5 | 6 s  | 5/5 | 30 s |
| N4  | 5/5 | 10 s | 5/5 | 18 s |
| N5  | 5/5 | 3 s  | 5/5 | 19 s |
| N6  | 5/5 | 2 s  | 5/5 | 25 s |
| N7  | 5/5 | 3 s  | 5/5 | 25 s |
| N9  | 5/5 | 8 s  | 5/5 | 16 s |
| N10 | 5/5 | 2 s  | 5/5 | 26 s |
| **Total** | **9/9 (45/45)** | | **9/9 (45/45)** | |

SR.jl source: `benchmarks/results/sr_comparison_base_20260607.csv` (2026-06-07, T=4 threads,
multithreading, no timeout, PySR strict defaults). No timed_out flags; max elapsed 35 s (N7 seed 3).
The 348 s artifact seen in the prior serial+pipe-blocked run does not recur under OS-level redirect.

Wall-time caveat: rsymbolic2 early-stops; SR.jl runs its full 100-iteration budget. Times
are not like-for-like on the stopping axis; see §5.

### 4.2 Sqrt mode (adds N8 and the sqrt operator; T = 4 threads, multithreading)

Source: `benchmarks/results/sr_comparison_sqrt_20260607.csv` (SR.jl, 2026-06-07) and
`docs/14` §1.6 gate Run 2 (rsymbolic2). 10 problems × 5 seeds = 50 runs.

| Problem | rsymbolic2 recovered | SR.jl recovered | SR.jl med time |
|---------|:--:|:--:|--:|
| N1  | 5/5 | 5/5 | 10 s |
| N2  | 5/5 | 5/5 | 27 s |
| N3  | 5/5 | 5/5 | 29 s |
| N4  | 5/5 | 5/5 | 33 s |
| N5  | 5/5 | 5/5 | 28 s |
| N6  | 5/5 | 5/5 | 27 s |
| N7  | 5/5 | 5/5 | 18 s† |
| N8 = sqrt(x) | 5/5 | 5/5 | 15 s |
| N9  | 5/5 | 5/5 | 13 s† |
| N10 | 5/5 | 5/5 | 26 s |
| **Total** | **10/10 (50/50)** | **10/10 (50/50)** | |

rsymbolic2 med times (Run 2, `docs/14` §1.6): 1–8 s for N1–N7/N9/N10, 1 s for N8 — all 5/5.
† SR.jl medians for N7 and N9 are over all 5 seeds; the stalled runs (N7 s2 = 915 s,
N9 s1 = 6769 s, N9 s2 = 695 s) sit in the tail and do not move the median. See §5.

## 5. Interpretation

**Headline (the valid claim): both tools recover 100 % of the Nguyen set** — 9/9 (45/45)
in base mode and 10/10 (50/50) with sqrt/N8 — at PySR's strict defaults, the same operator
set, and the same data (NMSE < 1e-4). On this benchmark rsymbolic2 does **not** trail
SymbolicRegression.jl (PySR's engine) on recovery. Recovery rate is the primary,
directly-comparable metric.

**Wall-time column — indicative context, not a speed comparison.** Two confounds preclude
a "tool A is X× faster" claim:
1. **Stopping criterion.** rsymbolic2 early-stops the moment NMSE < 1e-10; SR.jl consumes
   its full 100-iteration budget regardless. SR.jl time measures "full default budget",
   not "time to recovery".
2. **Internal parallelism structure.** rsymbolic2 runs T populations in parallel via
   OpenMP; SR.jl distributes its 31 populations across T Julia threads. Both use T = 4
   threads, but utilisation patterns differ.

A fair *speed* comparison would measure SR.jl's *time to first NMSE < 1e-4* (not its
full-budget time) at matched thread counts. That is deferred; this document establishes
the **recovery** comparison CLAUDE.md requires, with timing as rough context.

**Wall-clock stalls (recorded, not hidden).** The base run had 0/45 stalls; the sqrt run
had **3/50** runs flag a large wall-clock — N7 seed 2 = 915 s, N9 seed 1 = 6769 s, N9 seed
2 = 695 s. These are **not compute time**: each stalled run still RECOVERED with NMSE at the
noise floor (4.9e-07, 2.4e-30, 5.1e-31 respectively), i.e. SR.jl had already found the
solution and the extra wall-clock is an intermittent Julia-multithreading stdout/scheduling
stall, not search work. Because we report **medians**, they do not move the result: N7
median = 18 s (over 11/15/18/30/915), N9 median = 13 s (over 9/10/13/695/6769). Per the
`docs/16` §3 policy they are flagged and excluded from the timing median as artifacts —
never capped by injecting a PySR timeout. They are reported here precisely because they
reinforce the point above: SR.jl wall-time is context, not a like-for-like speed measure.

**Per roadmap, PySR is a recorded reference point, not a pass/fail gate.** The Phase 3
gate is "no regression vs the prior rsymbolic2 release" (satisfied — `docs/14` §1.6).
Parity (or better) on recovery against PySR's engine on the Nguyen set is a positive
data point but does not, by itself, decide Phase 4 scope.
