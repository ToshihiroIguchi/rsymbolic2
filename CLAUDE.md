# rsymbolic2

A native symbolic regression library with a C++ core, an R interface, and a
planned Python interface. It must run without any Julia dependency.

This file holds the project's durable principles. Implementation detail that may
change lives in `docs/` (architecture, dependencies, migration, benchmarks,
roadmap). When this file and `docs/` disagree, this file wins; update `docs/`.

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

- Do not assume the PySR / SymbolicRegression.jl architecture is optimal. It is a
  reference, not a specification. Borrow what is justified; redesign what is not.
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
- **Comparison against PySR is required** at an equivalent wall-clock budget on the
  same hardware. Report version, hardware, time limit, and number of runs.
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
