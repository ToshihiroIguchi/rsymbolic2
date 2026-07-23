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
  expression,latex,expression_simplified,latex_simplified,complexity_simplified}`). It
  includes only `rsymbolic/...` headers — no R/Python headers.
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
  (`linear_scaling`, `eval_cache`, `strong_simplify`, dimensional analysis, and macro
  operators — `macro_names`/`macro_bodies`, docs/57) are surfaced but off by default,
  mirroring the library's second layer.
- **The shared C++ core is not modified in a behaviour-changing way.** The core does now
  carry an optional per-epoch progress-observer hook (`SearchOptions::progress_callback`,
  docs/53) used by the WASM binding to drive the live Pareto chart, but it is null by
  default, pure observation (no RNG use, no search-state mutation), and proven
  bit-identical with the hook unset vs. attached (`test_progress_callback`) — so the
  standalone ctest suite, R `testthat` and Python `pytest` remain valid, and PySR Default
  Parity is unaffected. See docs/53 for the full contract and the exact seam.

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
- **The complexity column reports both node counts (`10 → 7`)** when display simplification
  shrinks the tree. Two columns of the results table describe two different trees: the hall
  of fame archives the best member *per raw complexity* (`hall_of_fame.hpp`, complexity =
  `tree.size()`), while the equation shown is its display-simplified form (docs/52, docs/54).
  Distinct raw trees can therefore normalise to the *same* printed expression, which reads as
  a contradiction — three rows showing the identical equation at complexity 7, 8 and 10. The
  bridge now returns `pareto_front.complexity_simplified` (the node count of the simplified
  tree; never larger than `complexity`, since `display_simplify()` adopts a rewrite only when
  it shrinks) and `main.js` `fmtComplexity()` renders `raw → simplified` whenever the two
  differ, plain `raw` otherwise. Two related display effects share the same cause and are
  *not* bugs: the losses of such rows differ (the front is strictly decreasing in loss) but
  round to the same string under `fmt()`'s 4 significant digits, and constants print at
  `%.6g` (`tree.hpp`), so `2.4999999` and `2.5` look identical. The raw searched expression
  stays available as the cell's `title` tooltip, and it — not the simplified string — remains
  the evaluatable round-trip source (docs/48 D2).
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
- **Full-data preview modal.** Clicking the data-summary line ("40 rows × 2 columns", a
  `<button>` styled as that text — the row count and "show me those rows" are one thought, so
  a separate Preview button was folded into it) opens the app's first native `<dialog>`
  (sticky header row, closed via ×, Esc, or a backdrop click), rendering up to 2000 rows with
  a note if the table is larger. An earlier revision additionally kept an inline 6-row
  mini-table behind a disclosure; it was removed as redundant — the modal is the preview.
  In-place cell editing in this grid was **considered and rejected**: pasting a corrected
  table covers hand-entered/edited data, and a spreadsheet-style editor's validation and
  data-sync complexity is not worth it in an analysis GUI — see "Explicitly out of scope".
- **Single search budget (no presets).** An earlier revision offered Quick/Balanced/Full
  preset pills; they were removed. The demo datasets finish fast enough that a reduced
  budget is unnecessary, so the GUI now always defaults to the PySR-parity budget
  (generations 2800, 31 populations × 27) — the same search the R/Python packages run by
  default. Anyone who wants a shorter run can still lower Generations under Settings.
- **Exponential loss ticks.** The Pareto chart's loss axis formats ticks via
  `fmtTick()` (`format.js`): exponential notation once labels leave the readable
  mid-range, so tiny SSE values never render as long decimal strings.
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

- **Static charts (no draw animation).** All three Chart.js constructions share
  `BASE_OPTIONS` in `plots.js`, which sets `animation: false`. Every redraw path destroys and
  rebuilds the chart (theme toggle, equation click, and once per throttled progress snapshot
  while a search runs), so Chart.js's default 1000 ms grow-from-the-axis would replay in
  full each time — a live Pareto front that should be watched converging instead re-inflates
  from zero several times a second, and clicking through equations flashes. Switching the
  live path to an in-place `chart.update("none")` was **considered and rejected**: the
  rebuild is what lets a draw pick up a new theme palette and swap the y-axis between linear
  and logarithmic, and with animation off the visual result is identical.
- **Data intake collapses after load; data and model are one card.** The Data card is the
  problem definition, so Target/Features moved into it (`.model-group`, hidden until there is
  data) and the separate Model card is gone. The intake block (drop zone + example picker) is
  the mirror image — one click's worth of use, then ~150 px of rail held for the whole
  session — so it hides once a table is loaded, behind a `change` button that reveals it
  again (`body.has-data:not(.editing-data) .data-intake`). The **paste textarea and its
  button were removed**: a document-level `paste` listener reaches the same `parseTable()`
  from anywhere on the page with no permanent UI (events originating in an `input`/
  `textarea`/`select`/contenteditable are ignored, so typing into a settings field still
  pastes normally), and the drop zone advertises it — "Drop a CSV here, click to browse, or
  paste (Ctrl+V)". Alternatives rejected: segmented File | Example | Paste tabs and a "Load
  data ▾" popover both add a mode or a layer that costs more than the three inline paths they
  replace.
- **Operators are visible; the parity constants are not.** Operators decide the reachable
  search space and are the shared **problem input** (per the Benchmarking Requirements), not
  a default to tune — an unchecked box makes whole expressions unreachable no matter how long
  the run — so the checkboxes now sit directly in the Search card instead of behind a
  disclosure indistinguishable from the ones holding `tournament_selection_p`. The former
  "Settings" and "Advanced (PySR parity)" disclosures were one panel split by importance
  rather than by kind, so they merged into a single `Settings` panel (budget fields first,
  then a `PySR parity constants` group). The rail keeps only its summary line, which carries
  live values — `2800 generations · seed 1` — because a collapsed panel that says nothing
  forces a click just to learn the budget, and appends `· modified` when **any** field
  differs from its shipped default: the GUI should never let a user drift silently off the
  parity defaults with no way back. `main.js` `DEFAULTS` is the single source for the reset,
  the modified marker, the per-field hint in the dialog, and `readConfig()`'s blank-field
  fallback, so the four cannot drift apart. Operator preset chips were rejected for the same
  reason the Quick/Balanced/Full budget pills were removed (above).
- **Search settings modal.** The 17 settings fields moved out of that disclosure into the
  app's second native `<dialog>` (`#settings-dialog`), opened from the summary line or its
  `edit` button — the same summary-line-plus-button pattern as the Data card. Measured on the
  pre-change build at 1366×768 with an example loaded: the expanded panel is **544 px tall in
  a 691 px rail**, starting at y≈834 — *entirely below the fold*, with `Reset to PySR
  defaults` at the very bottom; the rail's content ran 1465 px against its 691 px pane
  (+774 px of overflow, more than a full pane) with the macro disclosure open too. After the
  move the rail fits its pane exactly (0 px overflow in the default state). The dialog also
  buys what the 384 px rail could not afford: a **163 px field became 320 px in a 2-column
  grid**, and each field now prints a one-line description plus its shipped default
  (`Strength of the frequency-adaptive penalty on crowded complexities. Default 1040.`),
  injected from `DEFAULTS` so the text cannot disagree with the value the search falls back
  to. A field that no longer matches its default is marked at the field itself, not merely
  counted in the summary line — PySR default parity is the project's highest-priority
  configuration rule. Editing is **transactional**: values are snapshotted on open, `Apply`
  keeps them, and every dismissal path (`Cancel`, `×`, Esc, backdrop) restores them, where the
  inline panel could only be undone by resetting all 17 fields at once. The input ids are
  unchanged, so `readConfig()` still reads them straight from the DOM and the search is
  bit-identical (verified: the Quadratic example at seed 1 returns the same 14-row front and
  the same 1,219,366 evaluations before and after). What deliberately stayed out of the
  dialog: the **operator checkboxes** (problem input, see above), the results-side `recommend`
  / `log loss` controls (they re-render the visible chart instantly — a modal would break that
  loop), the macro editor (off by default, 261 px, and its real weakness is that body errors
  surface at Run time, which a modal does not fix), and the two high-accuracy opt-in
  checkboxes (a dialog costs more than the content).
- **Chart cards fill their row.** `.charts-row` stretches both cards to the taller one (the
  Pareto card carries an extra control row and the legend), so the Fit card's fixed 240 px
  canvas left a ~95 px blank strip under it — the tallest unused region in the answer-first
  view. The cards are now flex columns whose `.chart-box` grows into the row height
  (`flex: 1 1 auto` + a `min-height` floor), which closes the gap by construction for any
  future header difference instead of by matching two magic numbers; Chart.js re-fits itself
  (`responsive` + `maintainAspectRatio: false` inside a `position: relative` box), so no JS
  changed. Fixed alongside: the narrow-viewport `.charts-row { grid-template-columns: 1fr }`
  collapse had been dead — it sat in the `max-width: 1100px` block *above* the base
  `.charts-row` rule, and since a media query adds no specificity the later identical
  selector won even while the query matched, leaving two ~250 px charts side by side on a
  phone-width screen. That one declaration now lives in its own media query placed after the
  base rule.

Explicitly **out of scope** (would make it a mismatched full UI): tabs, live log /
records-over-time, a prediction/extrapolation workspace, an editable data grid (including
in-place cell editing in the full-data preview modal — rejected above), and a train/test-split
UI.

## Live per-epoch Pareto updates

Implemented — see **docs/53** for the full contract: the core's behaviour-neutral
`progress_callback` hook, the WASM/worker/main-thread relay, the redraw throttling, and
the four GUI states (idle / running-dimmed / running-live / finished).

## Deferred (later)

warm-start "continue"; feature-impact; full candidate-pool query (the engine currently
returns only the Pareto front).
