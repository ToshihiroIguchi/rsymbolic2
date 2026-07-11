<!--
SPDX-License-Identifier: Apache-2.0
Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
-->

# 51 · Static-site web GUI (WebAssembly)

Status: implemented (Phase 0–1). A browser front end that runs the existing C++ core,
compiled to WebAssembly, as a **static site** with no backend — while the R and Python
interfaces remain unchanged and first-class.

## What was added

A new `web/` subtree that is a **fourth consumer** of the shared core, exactly parallel to
`standalone/`, the R package, and `python/`:

- `web/wasm/rsymbolic2_wasm.cpp` — an Emscripten **embind** bridge, a field-for-field
  sibling of `python/src/rsymbolic2_py.cpp`. It marshals a JS options object (X as a flat
  row-major `Float64Array` + `nrow`/`ncol`, `y`, hyperparameters) into `SearchOptions` /
  `SearchSpace`, calls the same `run_evolution()` entry point, and returns a JS object with
  the same shape as the Python dict (`expression`, `loss`, `complexity`, `recommended`,
  `best_index`, `sst`, `n_evals`, `eval_counts`, `pareto_front{complexity,loss,score,
  expression,latex}`). It includes only `rsymbolic/...` headers — no R/Python headers.
- `web/wasm/CMakeLists.txt` — compiles the **same 9 core `.cpp`** files as
  `standalone/`/`python/` (single source of truth, nothing copied) with `emcmake`, plus a
  Node.js variant for the correctness gate.
- `web/app/` — a dependency-light, build-step-free static site (vanilla ES modules) with
  the standard SR-GUI layout: config panel, interactive Pareto plot + results table, and a
  per-equation detail panel (KaTeX-rendered LaTeX + prediction/residual plots).

## Why this is safe under the project rules

- **PySR Default Parity is untouched.** The WASM bridge resolves the identical defaults the
  Python wrapper does and drives the identical search algorithm, so the web build changes
  *where* the search runs, never *which* search runs. It is deterministic and reproducible
  for a fixed seed within the WASM build (thread-count determinism, docs/37,
  `test_island_model`). It does **not** return a bit-identical *expression* to a native
  (R/Python) build: the evolutionary trajectory is sensitive to last-bit floating-point
  differences between Emscripten's libm and the native (MinGW) libm, so the two builds can
  converge to different but equally valid expressions — verified by the Phase-0 gate, where
  both recover the quadratic example to ~1e-10 loss with different expression forms. This is
  cross-toolchain FP sensitivity inherent to GP search, not a behavioural divergence: same
  engine, same defaults, same recovery quality. Opt-in high-accuracy options
  (`linear_scaling`, `eval_cache`, dimensional analysis) are surfaced but off by default,
  mirroring the library's second layer.
- **The shared C++ core is not modified.** No change to `r-package/rsymbolic2/src/`, so the
  standalone ctest suite, R `testthat` and Python `pytest` remain valid. (A future Phase-3
  live-progress callback would touch the core and must be behaviour-neutral and guarded.)

## Feasibility (why WASM is low-risk here)

The core is ~6.3k LOC of dependency-free, STL-only C++17 with a single entry point. All
OpenMP is behind `#ifdef _OPENMP`; RNG is `std::mt19937_64` seeded from `options.seed` (no
`random_device`); there is no file/OS/thread dependency (only `stderr` for `verbosity>0`,
which Emscripten routes to the console). A single-threaded build needs no OpenMP,
pthreads, or SharedArrayBuffer, hence **no COOP/COEP cross-origin-isolation headers** — it
runs on plain static hosting (GitHub Pages).

## Dependency justification (per the Dependency Policy)

No new dependency is added to the **shipped R/Python library or its C++ runtime**: the WASM
target reuses only the existing STL core. The new dependencies are **web/build-only**:

- **Emscripten SDK (emsdk)** — the C++→WASM toolchain for the web target only, analogous to
  Rtools/scikit-build for the R/Python targets. It is a build tool, not a runtime
  dependency; nothing the library ships requires it. Fallback if it becomes unavailable:
  the web target is optional and can be dropped without affecting R/Python.
- **KaTeX** and **Chart.js** (both MIT) — vendored same-origin under `web/app/vendor/` for
  equation rendering and plotting; front-end assets of the web app only. No Julia, no
  third-party **C++** dependency, is introduced.

## Predict in the browser

`predict` is not in the core or bindings — R and Python each re-parse the returned infix
string. `web/app/js/predict.js` does the same with a small recursive-descent parser/
evaluator (no `eval`) matching `tree.hpp::to_string`. Same safe-pow caveat as the Python
wrapper: `^` maps to JS `**`, so a negative base to a fractional power yields NaN.

## UI layout: deliberately a single-page lightweight entry point

The layout was reviewed against TuringBot and other GUI SR tools (Eureqa, HeuristicLab),
which use a **tabbed full workspace** (Input / Search / Analysis / Prediction / Log). We
deliberately **do not** adopt that. Those are long-running, multi-core, persistent desktop
tools; a tabbed full UI presumes a workload — hours-long continuous search, large data,
background streaming — that this build cannot back: it is **single-threaded, 128 MB fixed
memory, single-shot**, and is positioned as a *lightweight entry point* ("for large or long
searches, use the R or Python package"). A full UI on the weakest backend would also violate
the project's Simplicity priority and the anti-speculative-scope rule. If a TuringBot-class
UI is ever wanted, it belongs on the **native / Python core** (where the compute can support
it), not on WASM.

The web GUI therefore stays a **single page**, refined only to remove three structural pain
points (no new heavy features, no engine/default change):

- **Persistent Run/Stop bar** above the config — the one useful idea borrowed from
  TuringBot's top bar. IDs unchanged, so JS wiring is untouched.
- **Viewport-bounded 2-D layout.** The page is a fixed-height app shell (`height: 100dvh`,
  `overflow: hidden`): header + run bar on top, then a body that fills the rest without the
  *page* scrolling. The body is **config rail** (internal scroll) | **results region**, and
  the results region is itself a 2-column grid — **Pareto explorer** (chart + internally
  scrolling table) | **selected-equation detail** (LaTeX + metrics + fit + export, internal
  scroll). This keeps the Pareto front and the selected detail **co-visible** so the *pick →
  inspect* loop needs no scrolling, and the primary payoff (best equation + fit) is never
  below the fold. An earlier revision stacked the detail *below* the table in one column;
  that removed horizontal eye-travel but pushed the payoff off-screen, so the viewport-bounded
  2-D layout supersedes it. Narrow (`max-width: 1100px`) or short (`max-height: 640px`)
  viewports release the shell to natural page scroll and re-pin the run bar (`position:
  sticky`), since a bounded shell would cramp them. This is layout-only CSS; `main.js` and
  all IDs are unchanged.
- **Empty state**: results/detail are hidden behind a placeholder (spanning the results
  region) until the first run.
- **Operator checkboxes grouped** by meaning (Trigonometric / Exp·Log / Power·other) for
  readability. Note: the core picks unary operators by a random index into the list
  (`mutation.cpp: space.unary_ops[op(rng)]`), so list *order* is part of the fixed-seed
  search trajectory; `checkedOps("un")` re-emits in the canonical `UNARY` order so the
  visual grouping never changes the search.
- **`model_selection` is a results-side live control** (a `recommend: best/accuracy/score`
  dropdown in the Pareto-front header, not a config knob buried in Advanced; the "best" option
  is labelled "best (PySR default)" so the parity default is visible in the UI, no
  value/logic change). It re-picks the recommended (★) member of the *existing* front
  instantly, with no re-run, because the returned front already carries the
  `loss/complexity/score` the rule needs. `main.js` `selectBestIndex()` is a faithful port of
  the core `select_best()` (`hall_of_fame.cpp`) — the C++ stays authoritative and computes
  `best_index` on each run; the JS port only re-applies the same rule (verified: default
  "best" agrees with the run's `best_index`). This resolves the common confusion that PySR's
  default "best" (highest score *within 1.5× of the lowest loss*) can, in a near-recovery
  regime, highlight a bloated near-zero-loss equation rather than the clean high-score one —
  switching to "score" now needs one click, not a re-run.
- **One morphing Run/Stop button** in place of two separate header buttons: idle shows
  "▶ Run", a run in progress swaps it in place for a red "■ Stop" (same slot in the sticky
  header, no layout shift), and `Ctrl`/`Cmd`+`Enter` starts a run from anywhere on the page.
  All three ways a run can end — user Stop, a result, or an error — funnel through one shared
  `finishRun()` that restores the idle button state, so the button can never get stuck showing
  "Stop" after the search has actually finished.
- **8 deterministic example datasets** (`web/app/js/examples.js`) behind a single compact
  `<select>` dropdown, replacing the earlier 2-example wrapping pill-button row so the Data
  card no longer grows with the list: quadratic `y = 2.5x² − 1.3`, damped oscillation
  `e^(−x/2)·sin(3x)`, rational `(x²+1)/(x+3)`, product `x0·x1 + x0`, trig `sin(x0) + x1²`,
  noisy linear `3x0 − 2x1 + ε`, a 200-row large quadratic `x0² − x1² + x0·x1`, and a
  gravity-like `x0·x1/x2²` (1–3 features). All are generated by a shared fixed-seed LCG +
  Box-Muller helper in `examples.js`, so an example always yields the same table. An example
  may declare an `ops` field listing non-default operators its target formula needs (e.g.
  `div` for the rational and gravity-like examples); loading such an example auto-checks those
  operator boxes. This is a UI convenience only — **operators are the shared problem input**
  (per the Benchmarking Requirements), not a PySR-parity default, so pre-checking them touches
  no search default.
- **Full-data preview modal.** The inline 6-row preview is unchanged; a new "View all rows"
  button opens the app's first native `<dialog>` (sticky header row, closed via ×, Esc, or a
  backdrop click), rendering up to 2000 rows with a note if the table is larger. In-place cell
  editing in this grid was **considered and rejected**: the existing paste-area already covers
  hand-entered/edited data, and a spreadsheet-style editor's validation and data-sync
  complexity is not worth it in an analysis GUI — see "Explicitly out of scope" below.
- **Light/dark theme.** A `data-theme` attribute on `<html>` is resolved before first paint by
  an inline `<head>` script (a `localStorage` override, else `prefers-color-scheme`), toggled
  by a header button and persisted. All colors moved to CSS custom properties with a slate
  dark palette; Chart.js axis and series colors are read from `--chart-*` CSS vars at draw time
  (`themeColors()`/`themedScale()` in `plots.js`) rather than mutating `Chart.defaults`, so a
  theme switch redraws correctly without leaking global chart state; KaTeX inherits
  `currentColor` and needs no separate styling.
- **Result-area polish**: one shared number formatter, `web/app/js/format.js`, replaces two
  slightly different `fmt()` implementations that had drifted apart in `main.js` and
  `plots.js`; long expressions in the all-equations table truncate with an ellipsis and a
  `title` tooltip carrying the full text; the Fit chart keeps its existing plot-type branch
  (curve overlay for 1 feature, predicted-vs-actual for ≥2) but gains a dynamic card title
  (`"Fit: <target> vs <feature>"` / `"Predicted vs actual"`) and real dataset column names on
  its axes instead of generic labels; a static HTML legend documents the Pareto point colors
  (selected / recommended ★ / front).

Explicitly **out of scope** (would make it a mismatched full UI): tabs, live log /
records-over-time, a prediction/extrapolation workspace, an editable data grid (including
in-place cell editing in the full-data preview modal — rejected above), and a train/test-split
UI.

## Deferred (later)

Live per-epoch Pareto/best-loss updates (needs a behaviour-neutral core progress callback);
warm-start "continue"; feature-impact; full candidate-pool query (the engine currently
returns only the Pareto front).
