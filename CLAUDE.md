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
  `precision=32`); the parallelism mechanism and RNG stream. These change *how* a
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
  a future phase might. Confirm with the user before widening scope.
- Correctness is verified, not assumed: unit tests for every component, numerical
  checks for automatic differentiation (compare against finite differences), and
  comparison against known-answer problems before trusting the search.

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
