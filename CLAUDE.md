# rsymbolic2

A native symbolic regression library with a C++ core, an R interface, and a
planned Python interface. The **shipped library and its runtime must not depend on
Julia.** Benchmarking is exempt: comparison against reference tools (e.g. PySR, whose
engine is SymbolicRegression.jl) may use Julia in the development/benchmark environment
only — never in anything the library ships or requires at runtime.

This file holds the project's durable principles. Implementation detail that may
change lives in `docs/` (architecture, dependencies, migration, benchmarks,
roadmap). When this file and `docs/` disagree, this file wins; update `docs/`.

## PySR Default Parity (highest-priority configuration rule)

**rsymbolic2's default configuration and search behaviour MUST be identical to
PySR's documented defaults. Only the *implementation method* may differ.** This is
the highest-priority rule for choosing default values and designing the search; it
overrides any self-tuned value, sweep result, or "we measured something better here"
argument. When a default or behaviour differs from PySR, that is a bug to fix, not a
trade-off to weigh — do **not** substitute a value chosen by our own benchmark.

- **Authoritative source.** PySR's installed `pysr/sr.py` `PySRRegressor.__init__`
  defaults and the SymbolicRegression.jl mechanisms they drive. The full default
  table and the exact mechanisms (frequency-adaptive parsimony normalisation, cost
  formula, mutation-weight set, tournament, migration) are recorded in
  `docs/28_pysr_default_parity_spec.md`. Read defaults from the installed source, not
  from memory or documentation prose — e.g. the installed default
  `adaptive_parsimony_scaling` is **1040.0**, not 20.
- **What "implementation method may differ" covers (allowed divergences):** the C++
  core with no Julia runtime; the constant optimiser algorithm (rsymbolic2 uses
  self-LM where PySR uses BFGS — `docs/06`); numeric precision (Float64 core vs PySR
  `precision=32`); the parallelism mechanism and RNG stream; behaviour-neutral
  evaluation accounting and caching (counting evaluations, and memoising
  deterministic full-data loss evaluations behind a strict structural-equality
  guard) — permitted only while the returned values and the search trajectory
  remain bit-identical with the mechanism on or off. These change *how* a
  result is computed, never *which* settings define the search.
- **What must match exactly (settings & behaviour):** population structure
  (`populations`, `population_size`), `ncycles_per_iteration`/`niterations`,
  `tournament_selection_n`/`tournament_selection_p`, `parsimony`,
  `adaptive_parsimony_scaling` and its normalisation, `maxsize`/`maxdepth`, every
  mutation weight, `crossover_probability`, `fraction_replaced`/`fraction_replaced_hof`,
  `optimize_probability`, `optimizer_nrestarts`/`optimizer_iterations`,
  `perturbation_factor`, `should_optimize_constants`, `annealing`, `warmup_maxsize_by`,
  and the default-off features (batching, turbo, etc. stay off).
- **Operators remain the shared problem input** (per Benchmarking Requirements), not a
  default to copy: PySR ships no default operator set.
- This rule does **not** waive Correctness: matching a value but computing it wrongly
  is still a bug. Parity defines *what* the search is; the Project Priorities govern
  *how* it is built.

### Scope: which components the parity rule binds (the web GUI is exempt)

The PySR default-parity rule binds the **shipped library**: the R package, the Python
package, and the shared C++ core they both call. Their defaults and search behaviour
must be PySR-identical, per everything above.

The **web GUI** (`web/app/`) is a browser-based *demonstration* front end, not the
shipped library, and is **not required to adopt PySR's defaults.** It runs the same
C++ engine (so a search launched with identical settings behaves identically), but it
may choose UI/presentation defaults that suit an interactive demo over strict PySR
parity. The concrete case: the web GUI defaults `model_selection` to **`score`** (the
parsimony elbow), not PySR's **`best`**, because PySR's `best` rule can recommend an
over-complex, noise-fitting expression when several equations are all effectively
perfect fits (loss at floating-point-noise level) — a poor first impression for an
answer-first demo. This divergence is confined to which Pareto member the GUI
*highlights*; the search itself is unchanged, `best (PySR default)` stays available in
the dropdown, and the R/Python/C++ layers keep `best` as their PySR-identical default.
When a web-GUI default diverges from PySR, say so in `web/` docs and keep the change on
the presentation side only — never let it reach the shared engine.

### Opt-in high-accuracy options (the second layer)

The parity rule above binds **defaults**. On top of the PySR-identical default layer,
rsymbolic2 may add **opt-in features that deliberately diverge from PySR to improve
accuracy** (e.g. a stronger constant optimiser, larger `maxsize`, more optimiser
restarts, an alternative least-squares backend, Keijzer-2003 linear scaling of
candidate outputs). This is a sanctioned design direction,
not speculative scope expansion. Such a feature is permitted only when **all** of:

1. **Off by default.** The shipped default value reproduces PySR's behaviour exactly.
   Turning the option on is the only thing that changes the search.
2. **Documented with evidence.** Its purpose, the divergence from PySR, and its measured
   accuracy effect are recorded in `docs/` before it is trusted (Benchmarking Requirements
   still apply — medians over runs, not a single best).
3. **Parity preserved when unused.** With every such option left at its default, the
   default-parity comparison against PySR must be unchanged.

When this layer and the default-parity rule appear to conflict, the resolution is always:
defaults match PySR, the divergence lives behind an explicit opt-in.

## Language Requirements

- Discussion with the user: **Japanese**.
- Source code, code comments, commit messages, and documentation: **English**.

## Project Priorities

When two goals conflict, the higher-ranked one wins. State the trade-off you made.

1. **Correctness** — a wrong answer fast is worthless.
2. **Maintainability** — code is read and changed far more than it is written.
3. **Portability** — see Platform Constraints; this outranks raw speed.
4. **Simplicity** — the simplest design that meets the requirement.
5. **Performance** — pursue only with measured evidence, never speculatively.

Performance is intentionally last. Do not trade correctness, maintainability, or
portability for speed unless a benchmark shows the speed matters and the user agrees.

## Platform Constraints

- **Windows 11 and Ubuntu LTS are both mandatory.** Neither is a second-class target.
- A change is not "done" until it builds and its tests pass on **both** platforms.
- **Verification cadence: Windows is the primary dev loop; run Ubuntu less often.** During
  iterative development verify on Windows every change; do **not** re-run the Ubuntu (WSL)
  build/tests on each iteration. Run Ubuntu verification at milestones instead — before
  declaring a unit of work "done", before committing/pushing to a shared branch, and after
  changes that plausibly affect portability (toolchain, build files, platform-specific code,
  threading/OpenMP, new dependencies). The "passes on both platforms" bar for *done* is
  unchanged; only the per-change frequency of the Ubuntu run is reduced.
- **R on Windows is built with Rtools (MinGW/GCC + UCRT), not MSVC.** A library being
  "fine on Windows via MSVC/vcpkg" does NOT mean it is usable from an R package; judge
  dependencies against the Rtools/MinGW/UCRT toolchain the R package actually uses.
- A dependency that raises Windows build or maintenance cost is a **major
  architectural penalty**, weighed against its benefit before adoption — not after.
- Prefer dependencies that are header-only, widely packaged, or easily bundled
  over those requiring per-platform toolchain work. Prefer the R-official mechanism
  (e.g. `SHLIB_OPENMP_CXXFLAGS`, `LinkingTo:` header packages) over hand-rolled
  linking. See `docs/06_windows_dependency_risk.md` for the evidence behind the
  current optimizer/parallelism choices.
- Parallel code must remain correct and complete when OpenMP is absent (CRAN
  binaries, notably macOS, may have it disabled): always provide a serial fallback.

## Dependency Policy

- Every major dependency requires a written technical justification (what it
  provides, why no simpler option suffices, its Windows cost). Record it in `docs/`.
- The default answer to "should we add a dependency?" is **no**. The burden of
  proof is on adding it.
- Popularity is not justification. Recency is not justification.
- Before adopting, name the fallback if the dependency becomes unavailable.

## Development Philosophy

- For **architecture and implementation** (how the engine is built — data
  structures, optimiser internals, parallelism, memory layout), PySR /
  SymbolicRegression.jl is a reference, not a specification: borrow what is justified,
  redesign what is not. For **default settings and search behaviour**, the opposite
  holds — PySR *is* the specification (see "PySR Default Parity" above): match it
  exactly. The split is deliberate: same behaviour, our own implementation.
- Do not assume newer technology is better. Prefer approaches with evidence.
- Write code that reads like the surrounding code. Match existing idioms.
- Prefer a simple correct design over a clever fast one until measurements
  demand otherwise.
- Do not expand scope speculatively. Build what the current phase needs, not what
  a future phase might. Confirm with the user before widening scope. Note: adding an
  **opt-in high-accuracy option** (per "Opt-in high-accuracy options" above) is a
  sanctioned direction, not speculative scope — provided it stays off by default and
  is backed by measured evidence.
- Correctness is verified, not assumed: unit tests for every component, numerical
  checks for automatic differentiation (compare against finite differences), and
  comparison against known-answer problems before trusting the search.
- **Reserve the expensive main thread for planning, orchestration, hard or
  repeatedly-failing work, and load-bearing edits; delegate what can be cleanly
  cut out.** Hand off wide read-only exploration (cross-codebase surveys,
  "where/how is X done") and independently runnable verification or benchmark
  jobs to subagents, taking their conclusions back into the main thread and
  confirming the load-bearing files directly when a change depends on them. This
  conserves the main thread's scarce context and cost on large tasks. Cost is
  saved only when the delegated task is self-contained AND handed to a cheaper
  model (specify `model` explicitly, e.g. Haiku/Sonnet) — spawning is otherwise
  the expensive path, since each subagent re-derives context from cold. So do
  **not** delegate small local edits (inline is cheaper than a cold start), and
  do not fan out speculatively: delegate when the scope is genuinely broad, when
  a self-contained task can move to a cheaper model, or when the user asks.

## Benchmarking Requirements

- Benchmarks are the evidence behind decisions, not decoration to confirm them.
  Let results change the plan.
- **Primary benchmark: Feynman** (ground-truth recovery).
- **Secondary benchmarks: Nguyen, Keijzer, SRBench.**
- **Comparison against PySR is required.** Run PySR **only at its documented default
  settings** — never a hand-tuned "default-equivalent" approximation. The operator set is
  the one shared problem input given identically to both tools (it defines the search
  space, it is not an algorithm tuning knob). Any equalization needed for a fair
  comparison (e.g. the thread/compute budget) is done on **rsymbolic2's** side, never by
  altering PySR's settings. Report version, hardware, time limit, and number of runs.
- Report medians over multiple runs with spread, never a single best run.
- Recovery thresholds and protocol follow `docs/` (benchmark strategy); do not
  weaken thresholds to make results look better.

## Decision-Making Principles

- Prefer evidence over intuition; when evidence is weak, **say so explicitly** and
  proceed with the simplest reversible option.
- Prefer reversible decisions made quickly over irreversible ones made slowly.
- When uncertain about a trade-off that affects the user (a new dependency, an API
  shape, a platform compromise), ask in Japanese before committing.
- Confirm before hard-to-reverse or outward-facing actions (adding dependencies,
  pushing to the remote, changing the public API).

## Status and Uncertainty

The performance targets and library choices in `docs/` are working hypotheses, not
proven results. They are based on the published literature and reference projects,
which may not transfer to this codebase. Treat them as provisional until this
project's own benchmarks confirm them, and revise the docs when they do not.
