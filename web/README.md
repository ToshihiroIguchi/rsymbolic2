<!--
SPDX-License-Identifier: Apache-2.0
Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
-->

# rsymbolic2 web GUI (static site)

A browser front end for rsymbolic2 that runs the **same C++ symbolic-regression engine**
as the R and Python packages, compiled to WebAssembly. It is a **static site** — no
backend server — so it can be hosted on GitHub Pages or opened from any static file host.
The engine runs single-threaded in the browser and is **deterministic and reproducible**
for a fixed seed (identical result every run). It uses the identical search algorithm and
PySR-parity defaults as R/Python, so it recovers the same problems with the same quality.
The *specific* expression returned can differ from a native (R/Python) build: the
evolutionary trajectory is sensitive to last-bit floating-point differences between
Emscripten's libm and the native (MinGW) libm, so the two builds may converge to different
but equally valid expressions — this is normal for FP-sensitive GP search, not a weaker
search. For large or long searches, prefer the R or Python package — see
[How much data fits](#how-much-data-fits) for the measured numbers behind that sentence.

This subtree does **not** touch the shared C++ core (`r-package/rsymbolic2/src/`): the
WASM binding (`wasm/rsymbolic2_wasm.cpp`) is a third thin bridge over the same
`run_evolution()` entry point, exactly parallel to the R (cpp11) and Python (pybind11)
bridges.

## Layout

```
web/
  wasm/                     C++ -> WebAssembly binding + build
    rsymbolic2_wasm.cpp     embind bridge (sibling of python/src/rsymbolic2_py.cpp)
    CMakeLists.txt          emcmake target (compiles the SAME 9 core .cpp, OpenMP off)
    build.ps1 / build.sh    emcc build wrappers
    test/parity_test.cjs    Phase-0 correctness gate (Node)
  app/                      the static site (this is what gets deployed)
    index.html
    css/style.css
    js/*.js                 UI (main, worker, predict, data, plots, latex, tree,
                            export, report)
    vendor/                 KaTeX + Chart.js (vendored, MIT) + built rsymbolic2.{js,wasm}
    examples/ (inline)      example datasets live in js/examples.js
```

## What the browser stores

Two `localStorage` keys, both small, both optional — the site works identically with storage
disabled or full:

| key | contents |
|---|---|
| `theme` | light/dark, when you have used the toggle |
| `rsymbolic2.search-settings.v1` | the operator selection, the macros, the settings dialog's fields and the two high-accuracy opt-ins |

What you set up **before** pressing Run is remembered; what the run produced, and how it is
displayed, is not. Nothing else is stored — in particular:

- **not your data.** It never leaves the browser, a real table does not fit in `localStorage`
  anyway, and auto-restoring one into the fixed 128 MB heap would make a reload a broken page.
- **not the results.** They describe data that is gone. Use the PDF report, the Pareto CSV
  download, the copy buttons, or the R/Python snippet, which reproduces the *run*.
- **not the target/feature/sample choices**, which are statements about one table.
- **not the results-view controls**, above all the model-selection rule — a fresh visit must
  recommend the shipped default, not a click from weeks ago.

When the restored settings differ from the shipped ones, the Search card says so on arrival and
offers **Use defaults**, which forgets them and resets operators, macros, opt-ins and fields
together. A hand-edited or stale entry is discarded rather than half-applied. `docs/63` records
the decision and the validation rules.

## Defaults that differ from PySR

The **search** here is at PySR parity, exactly like the R and Python packages: same
defaults, same mechanisms, same trajectory for a given seed. The GUI is exempt from the
default-parity rule only for *presentation* choices (`CLAUDE.md`, "the web GUI is exempt"),
and it uses that exemption in exactly one place:

| setting | web GUI | R / Python / PySR | why |
|---|---|---|---|
| `model_selection` | `score` | `best` | Which Pareto member is highlighted (★). When several equations are all effectively perfect fits — loss down at floating-point noise — PySR's `best` rule can recommend the most complex of them, which is a poor answer for an answer-first demo. `score` picks the parsimony elbow. |

This changes nothing about the search: it selects a member of the *finished* front, it is a
live control in the Pareto card (switching it re-picks instantly, with no re-run), and
`best (PySR default)` is right there in the dropdown. The copied Python/R snippets emit
`model_selection` whenever it differs from those packages' default, so a pasted snippet
recommends the same equation the screen does.

## Result views

Three charts, all live (switching a control re-draws instantly, nothing re-runs):

- **Pareto front** — complexity vs. loss over the archived equations; click a point (or a
  table row) to select an equation, and the whole result column follows it. Also drawn
  while the search runs, from progress snapshots.
- **fit** — the selected equation against the data: the fitted curve over the observed
  scatter for a single feature, predicted-vs-actual with a dashed reference line otherwise.
- **residual** — actual − predicted against predicted, from the `view` dropdown on the same
  card. This is the diagnostic the other two cannot give: once the points are dense, both a
  curve overlay and a high R² hide systematic error, while a bend or a fan in the residuals
  shows immediately that the equation is missing a term.

The charts share the answer-first layout rather than opening in a modal: they are primary
evidence for the headline equation, and the Pareto → equation → fit loop needs them visible
next to the table. The two modals (`<dialog>`) stay reserved for secondary things — the
full-data preview and the numeric search settings.

Both per-equation views draw a bounded stride subset of the rows (`DISPLAY_POINT_CAP`,
5,000) and say so under the chart; the reported metrics come from the engine, not from the
plotted subset.

At the foot of the result column sits the **equation tree** (`js/tree.js`, docs/48 D6): the
same equation as a syntax tree, operators as inner nodes, data columns and fitted constants
as leaves, told apart by fill. It follows the selected equation like the charts do, and
downloads as a standalone `.svg`. It is last rather than under the hero on measured grounds
(docs/48 D6): it is the one result card that adds no new information, and its height follows
the selected equation (122–442px across one run's front, ~890px for a deep tree), so above
the charts it pushed the Pareto → equation → fit loop off the first screen and moved both
selection surfaces 319px out from under the pointer on every selection change. It is drawn
from the *printed* (display-simplified)
expression, so its node count can be smaller than the `complexity` column, which counts the
raw tree the search archived. Unlike the Chart.js plots it takes its colours from the theme's
CSS variables, so the light/dark toggle recolours it without a redraw. No plotting library is
involved — Chart.js cannot draw trees, and nothing new was vendored.

## Taking the result away

Five exports, four of them scoped to one thing and one to the whole run:

| control | where | what it gives you |
|---|---|---|
| Copy **LaTeX** / **SymPy** / **Python code** / **R code** | Best formula card | the displayed equation as LaTeX or as Python, or a snippet reproducing the *run* in either package |
| **CSV** | All equations card | one row per Pareto member: raw, display-simplified, and SymPy |
| **SVG** | Equation tree card | the tree as a standalone vector file |
| **PDF** | the header | the whole run as one printable document |

The **PDF** button (docs/64) builds a report and opens the browser's own print dialog —
choose "Save as PDF" there. Nothing is vendored to make it: a PDF writer would be a
300–500 KB dependency and would have to re-implement the equation rendering and the tree in
its own drawing API, so the browser does the writing and the site supplies a print
stylesheet. The report is three sections, each on a fresh page: the answer (equation,
metrics, Pareto and fit charts), the evidence (every equation in the front, untruncated, plus
the tree), and a reproducibility appendix (operators and macros, every setting beside its
PySR default with any divergence marked, the R and Python snippets, the run's timing and
evaluation counts, and the notes a reader needs — above all that this build is not
bit-identical to R/Python, so re-running the snippet can return a different but equally
valid expression).

**SymPy** exists because the expression string beside it is not valid Python. Its power
operator is `^`, which Python reads as **xor** — so pasting the displayed equation into
`eval()`, NumPy or `lambdify()` computes the wrong function without complaining
(`sympify()` alone is the exception). The SymPy button copies `**` instead (docs/70,
docs/71). Note that the display simplifier can introduce a squaring on its own, from `x*x`,
even when that operator is not in the library — so a displayed `^` can appear in a run that
never enabled `pow` or `square`. The copied form is the *mathematical* expression: the
engine's
`sqrt`, `log` and `^` are domain-guarded and return NaN where SymPy returns a complex value.

It is a report *about the run*, not a screenshot of the page: the charts are re-rendered
off-screen at roughly 380 dpi, the equation table drops the on-screen scroll cap and
ellipsis, and the report prints black-on-white even in dark mode. Ctrl/Cmd+P takes the same
path. Printing from the browser's own menu also works, but that path cannot wait for the
chart images, so the button is the reliable one.

## How much data fits

The browser build is single-threaded and its WASM heap is fixed at 128 MB, so row count
decides both how long a run takes and whether it can run at all. Measured on the
development machine (full default budget, 2,800 generations x 31 populations ~ 2.83M
evaluations, each one O(rows); full detail and the memory model in `docs/59`):

| rows x fitted columns | full default-budget run | what the GUI does |
|------:|------------------------|-------------------|
| up to ~10,000 cells | seconds to ~5 min | nothing — this is the comfortable range |
| ~60,000 cells | tens of minutes | warns, and offers row sampling / batching in the notice itself |
| above the heap ceiling (~80,000–110,000 rows, shape-dependent) | would abort | samples the table down automatically and says so |

The warning threshold counts **cells**, not rows (docs/70 §3.6): every evaluation walks
every fitted value, so 5,000 rows x 20 columns is an order of magnitude more work than the
5,000 x 2 the figures above are measured on.

Two levers, both visible in the UI:

- **Batching** (Search settings -> Large data, off by default exactly as in PySR) evaluates
  each iteration on a random `batch_size` rows while the hall of fame and the reported
  result stay on the full data. Measured 18x faster at 10,000 rows and 22x at 100,000. It
  speeds up a run; it does **not** raise the row ceiling.
- **Row sampling** (Data card) fits a deterministic sample instead of the whole table. This
  is the only thing that moves the memory ceiling, so it is applied automatically — and
  reported in the summary, the preview and the copied R/Python snippet — when a table
  cannot fit at all.

The ceiling is computed from the columns actually being **fitted** (the ticked features plus
the target), not from the file's width, so untick the columns you do not need and a wide file
gets correspondingly more rows. It also moves with `n_populations` (each island holds O(rows)
optimiser scratch), and is re-checked whenever either of those changes.

Files above 64 MB are refused before they are read, since parsing happens on the main
thread. For anything larger than the comfortable range, the R or Python package is the
right tool: they are multi-threaded and not bounded by a 128 MB heap.

## Custom (macro) operators

The sidebar's **Custom operators (macros)** disclosure (under the operator checkboxes) lets a
user define one-argument templates over the primitives — `gauss = exp(-square(x))` — which the
engine expands as it builds expressions (docs/57). The feature is **off by default**: with no
macro rows the two option arrays sent to the bridge are empty and the search is bit-identical
to the PySR-parity run.

A grouped preset list (Peaks, S-curves, Growth/decay, Powers/roots) fills a row in, since the
syntax is unguessable and the presets are where the feature explains itself. Each carries the
node count it adds — the figure of merit, because a macro's whole benefit is reaching in one
mutation what the primitives reach in several — and a note on which of its numbers end up
*fitted*: a literal in a body becomes a tunable constant, so `x^3` is a fitted power rather
than a cube (`docs/57` §6). `parity_test.cjs` asserts every shipped preset is accepted by the
engine, reading the bodies out of `main.js` so the list is never duplicated.

Bodies are validated by the engine's own parser when Run is pressed, so the browser rejects
exactly what R and Python reject, with the same message — the page carries no second copy of
the grammar. Only name-level problems (blank, duplicate, shadowing a built-in) are reported
before the run starts. Results always print the expanded primitive form, so macros need no
support in `js/predict.js` or in the copied R/Python snippets' round trip; the snippets do
carry a `macro_ops` argument so they reproduce the run.

## Building the WebAssembly module

Requires the [Emscripten SDK](https://emscripten.org/docs/getting_started/downloads.html).
Install and activate it once, then put it on PATH for the build shell:

```bash
# one-time
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk && ./emsdk install latest && ./emsdk activate latest
```

Build (from the repo root), after sourcing the emsdk environment:

- **Windows (PowerShell):** `. C:\path\to\emsdk\emsdk_env.ps1; web\wasm\build.ps1`
- **Ubuntu/macOS:** `source /path/to/emsdk/emsdk_env.sh && web/wasm/build.sh`

This emits `web/app/vendor/rsymbolic2.js` + `rsymbolic2.wasm`.

## Running the correctness gate

```bash
# build the Node variant too (the build above builds both targets)
node web/wasm/test/parity_test.cjs
```

It checks recovery of the quadratic example, run-to-run determinism, the display-simplified
result fields, the progress callback, and the opt-in options' default-off parity —
`strong_simplify` and macro operators (docs/57) must leave the run bit-identical when unused,
which is what proves the code added for them is inert. If the Python `rsymbolic2` package is
importable it also cross-checks that both builds recover the example to comparably tiny loss
(outcome equivalence, not string equality: the two toolchains' libm differ in the last bit).

The macro block additionally pins the guarantees the GUI depends on: an invalid body is
rejected by the engine's own parser with the same message R and Python print, and a macro
never appears by name in a returned expression (results carry the expanded primitive form, so
`js/predict.js` needs no macro knowledge).

## Serving the site locally

The site must be served over HTTP (ES module workers + WASM do not load from `file://`):

```bash
python web/serve.py             # serves web/app at http://localhost:8080
python web/serve.py 8099        # ...on another port
```

`web/serve.py` is `python -m http.server` plus a `Cache-Control: no-store` header. Use it
rather than plain `http.server`: the latter sends no cache headers at all, so browsers apply
*heuristic* freshness and can keep serving a stale `index.html` — or, far worse, a stale
`vendor/rsymbolic2.wasm` — for hours after a rebuild. The page then silently runs the old
engine behind the new UI.

Cache entries already stored by a plain `http.server` stay fresh regardless of the new header,
so you have to discard them once. Note that a **hard reload is not enough**: it re-fetches the
document, `js/*.js` and `js/worker.js`, but the engine is imported *by the worker*
(`worker.js`: `import ... from "../vendor/rsymbolic2.js"`), and a dedicated worker's own
subresource fetches are not covered by the document's reload flag — verified from this
server's access log, where `vendor/rsymbolic2.{js,wasm}` were never re-requested after a
forced reload. Clear the browser cache instead (DevTools → Application → Clear storage, or
Network tab → "Disable cache" while DevTools stays open), then reload.

## Deployment

The site deploys to **GitHub Pages via GitHub Actions**
(`.github/workflows/deploy-pages.yml`): on every push to `master` that touches the
web subtree or the shared C++ core (and on manual `workflow_dispatch`), CI rebuilds
the WASM module from source with the pinned Emscripten toolchain, runs the Node
parity gate, and publishes `web/app/` to Pages. Building in CI (rather than
deploying the committed `vendor/rsymbolic2.{js,wasm}`) guarantees the published
binary matches the sources at that commit; with the same pinned toolchain the WASM
output is bit-identical to the verified local build.

One-time repository setting: **Settings → Pages → Build and deployment → Source =
"GitHub Actions"** (the workflow also tries to enable this automatically). No
cross-origin-isolation (COOP/COEP) headers are needed because the build is
single-threaded, so plain Pages hosting suffices; the same `web/app/` directory can
also be copied to any other static host.
