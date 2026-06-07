# Implementation Plan: PySR Comparison at Strict Defaults

**Date:** 2026-06-07
**Basis:** Updated CLAUDE.md principle — *"Run PySR only at its documented default
settings… The operator set is the one shared problem input given identically to both
tools… Any equalization needed for a fair comparison is done on rsymbolic2's side, never
by altering PySR's settings."*
**Supersedes:** `docs/14` §2 (PySR-baseline outline) and the methodology + results in the
already-committed `docs/15` (which violate the new principle — see §1). The harness
`benchmarks/03_sr_comparison.jl` is revised, not rewritten.

## 0. Why this revision

The committed comparison ran SymbolicRegression.jl (PySR's engine) but **deviated from
PySR's defaults** in two ways and framed the operator set as a "deviation". Under the new
principle that is not acceptable: PySR must run at its true defaults, the operator set is a
shared problem input (option A), and all equalization happens on rsymbolic2's side.

## 1. What is non-compliant in the committed harness/docs15

Read from the installed `pysr/sr.py` (`PySRRegressor.__init__`), PySR's relevant defaults:

| param | PySR default | committed harness | compliant? |
|-------|--------------|-------------------|:--:|
| parallelism | `None` → multithreading | forced `:serial` | ❌ |
| procs | `None` (all available) | (serial) | ❌ |
| timeout_in_seconds | none | `120` backstop | ❌ |
| deterministic | `False` | (serial, seeded) | ❌ |
| niterations | 100 | 100 | ✅ |
| populations | 31 | 31 | ✅ |
| population_size | 27 | 27 | ✅ |
| ncycles_per_iteration | 380 | 380 | ✅ |
| tournament_selection_n | 15 | 15 | ✅ |
| parsimony | 0.0 | 0.0 | ✅ |
| maxsize | 30 | 30 | ✅ |
| early_stop_condition | None | None | ✅ |

So two algorithm-affecting deviations must go: **forced serial** and the **timeout
backstop**. The hyperparameter block is already correct.

## 2. Target configuration

### 2.1 PySR (SymbolicRegression.jl) — strict defaults

- `parallelism = :multithreading` (PySR's resolved default for `parallelism=None`). Drop
  `parallelism = :serial`.
- **Remove `timeout_in_seconds`.** PySR has no default timeout; the 100-iteration budget
  plus `maxsize=30` already bound the work (validated: N3 s4 → ~22 s once `niterations`
  was 100 and output was no longer pipe-blocked).
- Keep all verified default hyperparameters (the ✅ rows above).
- `deterministic` stays at its default (`False`). Under multithreading the run is not
  bit-reproducible; that is PySR's default behaviour. We report medians over 5 seeds.

### 2.2 Operators — the one shared problem input (option A)

Operators define the search space (a problem statement), not an algorithm tuning knob, so
both tools receive the **identical** set:
- base mode: binary `[+,-,*,/]`, unary `[exp,log,sin,cos]` (maps rsymbolic2's gate set;
  `neg` via `0 - x`).
- sqrt mode: add `sqrt`, include N8.

This is given to PySR explicitly (PySR's literal default has no unary operators and could
not address N5–N10). Documented as a shared problem input, consistent with the CLAUDE.md
wording. It is the **only** thing we set on PySR beyond its defaults.

### 2.3 Output handling (non-algorithmic — must not change results)

SR.jl's per-iteration progress bar/tables previously blocked the process on a full capture
pipe. `progress=false` (and quiet `verbosity`) only suppress **logging**; they do not touch
the search or its results. To keep even this off PySR's algorithm path, the harness will
additionally redirect Julia stdout at the OS level (`> file`) rather than through a
PowerShell pipe, so blocking cannot recur. These are presentation-only and are recorded as
such; they are not a hyperparameter change.

### 2.4 Thread/compute budget — equalized on rsymbolic2's side

The two tools parallelise differently (PySR: populations across Julia threads; rsymbolic2:
OpenMP across islands, thread-count = island-count granular). A clean common budget is
**T = 4 threads**, because:
- it is rsymbolic2's existing gate configuration (`n_populations = 4`, `docs/14` §1.6), so
  **rsymbolic2 needs no re-run** — those gate numbers are the comparison baseline;
- it sits in rsymbolic2's measured-reasonable range (its efficiency falls off above ~2–3
  threads, `docs/04` §6 / `docs/09` §3), so 12 threads would misrepresent it;
- PySR is **not** altered: we launch Julia with `-t 4`, and PySR at its default
  `procs=None`/multithreading simply uses the 4 threads the environment provides. Setting
  the machine's thread budget is the equalization knob we legitimately control; no PySR
  *setting* is touched.

T is a recorded equalization parameter (T = 4). If a future comparison wants T = all cores,
that requires re-running rsymbolic2 at a matched island count and is out of scope here.

### 2.5 Residual stopping-criterion difference (reported, not hidden)

PySR's default has **no early stop**, so SR.jl runs its full 100-iteration budget every
run; rsymbolic2's gate uses *its* default early stop (`target_loss=1e-10`) and halts once
it reaches it. Each tool therefore runs at its own default stopping behaviour. Consequence:
**recovery rate is the primary, directly-comparable metric; wall-time is reported as
context** and still is not strictly like-for-like on the stopping axis. (Optional, not in
this plan's scope: a separate "full-budget vs full-budget" timing run with rsymbolic2's
early stop disabled — an equalization on our side — could make wall-time comparable; deferred.)

## 3. Operational safeguard for the no-timeout decision

Removing the timeout means a pathological run is not auto-capped. Mitigations that are
**not** PySR settings:
- `niterations=100` + `maxsize=30` bound the algorithmic work (validated ~17–48 s/run).
- OS-level stdout redirection removes the pipe-block that caused the earlier idle "19-min"
  artifacts.
- The harness still records `timed_out`-style flags by wall-clock for transparency, and a
  run exceeding a sane wall bound is handled in **analysis** (flagged, excluded from the
  timing median as an artifact) — never by injecting a PySR timeout.
- Operationally, the run is launched in the background and monitored; if the whole run
  hangs it is investigated, not silently capped.

## 4. Implementation steps

1. **Revise `benchmarks/03_sr_comparison.jl`:**
   - `run_one`: `parallelism = :multithreading`; remove `verbosity`/keep `progress=false`
     as output-only; drop the timeout from the search call path.
   - `make_options`: remove `timeout_in_seconds`; keep the verified default hyperparameters
     and operators (§2.2). Update the warm-up Options likewise.
   - Update the header comment to state: PySR strict defaults; operators = shared problem
     input; threads equalized via the launch (`-t 4`); output suppression is non-algorithmic.
2. **Launch** with Julia multithreaded and OS-level redirect, e.g.
   `julia +release -t 4 benchmarks/03_sr_comparison.jl base > benchmarks/results/_sr_base.log 2>&1`
   (background; monitor). Then `sqrt` mode.
3. **Rewrite `docs/15` §3–§5** to the compliant methodology and the new results table:
   - State PySR strict-defaults, operators-as-shared-input, T=4 equalization, multithreaded.
   - rsymbolic2 column reuses `docs/14` §1.6 gate (n_populations=4) numbers.
   - Keep recovery as the headline metric; wall-time as caveated context (§2.5).
   - Remove the now-obsolete "deviations: serial + backstop" framing.
4. **Sanity re-validate** one previously-slow case (N3) under the new multithreaded, no-
   timeout config before the full run, to confirm bounded runtime (as done before each run).
5. **Commit**: harness revision + docs/15 rewrite, referencing this plan and the CLAUDE.md
   principle change.

## 5. Metrics and reporting

- **Primary:** exact-recovery rate (NMSE < 1e-4, the `utils.R` criterion), per problem and
  overall, both tools, 5 seeds. This is thread- and stopping-criterion-robust.
- **Context:** median wall-time per problem at T=4, with the §2.5 caveat stated plainly.
- Full reporting checklist from `docs/04` §5 (versions, hardware, time limit, n_runs, full
  table, no cherry-picking). PySR remains a recorded reference point, not a pass/fail gate.

## 6. Scope boundary

In scope: revise the harness to PySR strict defaults, re-run base (and sqrt), rewrite
`docs/15`. Out of scope: T = all-cores comparison (needs rsymbolic2 re-run at matched
islands), the optional full-budget timing equalization (§2.5), Feynman, and any change to
the shipped library. No PySR setting is tuned; only the shared operator input and the
machine thread budget are set, per CLAUDE.md.
