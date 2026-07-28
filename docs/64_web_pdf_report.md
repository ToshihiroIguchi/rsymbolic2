<!--
SPDX-License-Identifier: Apache-2.0
Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
-->

# Web GUI: a printable PDF report

## 1. Scope and the one rule this change must not break

The web GUI can already export three things: the equation (LaTeX / expression string), a
snippet that reproduces the run (R, Python), and the Pareto front (CSV). What it cannot do is
hand someone **the whole answer as one document** — the equation, the evidence for it, and the
settings that produced it, on paper.

This document specifies that report and how it is produced.

**Nothing here touches the search.** No C++ file, no configuration default, no RNG stream, no
`readConfig()` field. The change is confined to `web/app/` presentation code, so the WASM
parity gate (`web/wasm/test/parity_test.cjs`) is unaffected by construction. This is the
web-GUI presentation layer that CLAUDE.md exempts from PySR default parity — and the report
does not introduce a *new* divergence: it prints the ones that already exist (§5.4).

## 2. How the PDF is produced: the browser, not a library

**Decision: a print stylesheet plus `window.print()`.** The user chooses "Save as PDF" in the
browser's own print dialog.

The alternative was vendoring a PDF writer (jsPDF, pdfmake) into `web/app/vendor/`. It was
rejected on the Dependency Policy: the default answer is no, and the burden of proof is on
adding it.

| | print CSS + `window.print()` | vendored PDF library |
|---|---|---|
| new dependency | none | ~300–500 KB, plus fonts |
| equations | KaTeX renders as it does on screen | must be re-rendered or rasterised |
| tree | the existing SVG, vector | must be re-drawn in the library's own API |
| text in the PDF | selectable, searchable | selectable only if fonts are embedded |
| layout | one CSS block | the whole report re-implemented in a drawing API |
| offline / static site | unchanged | unchanged |

The one thing the browser route does not give us is control over page headers, footers and
page numbers — those come from the browser's own print settings. That is an acceptable loss
for a zero-dependency solution, and it is the same loss every "Print this page" button takes.

A third option, rasterising the page with `html2canvas`, was rejected outright: it produces a
PDF whose text cannot be selected or searched, which is most of the value of a report.

### 2.1 Why the report is a separate DOM subtree, not the app restyled

The obvious implementation — `@media print` rules over the existing results column — does not
work, for reasons that are all in the current CSS:

- `.table-wrap { max-height: 320px; overflow: auto }` — the equation table is a scroll box;
  printed, it shows only the first screenful.
- `#pareto-table td:last-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap }`
  — every equation past ~22rem prints truncated with an ellipsis.
- `.tree-box { overflow-x: auto }` — a wide tree prints clipped.
- The charts are `<canvas>` elements sized to their on-screen container and drawn at the
  screen's device pixel ratio: printed, they are blurry.
- Half the report does not exist in the DOM at all — the settings table, the R/Python
  snippets, the run metadata and the notes are all things the page never shows in one place.

So the report is **built on demand into a hidden container** (`#print-report`), from
application state, and the print stylesheet hides the app and shows that container. The app's
own layout is then irrelevant to the printed output, which is the point: the two can be
changed independently.

## 3. What the report contains

Three sections, in the order the reader needs them, each starting on a fresh page. The
structure follows the GUI's own answer-first principle: the answer, then the evidence, then
the fine print. Measured on the bundled quadratic example that is four A4 pages — the
appendix runs to two, and the equation table takes more when the front is long.

### Page 1 — the answer

| block | source |
|---|---|
| title, generation timestamp, tool version | `APP_VERSION`, `new Date()` |
| data line: `target ~ f(features) — N rows × M columns`, source name | `state.run.source`, snapshotted names |
| sampling banner, when the run fitted a subset | `state.sampling` |
| the equation, rendered large (KaTeX) | `latex_simplified[i]`, feature names substituted |
| the equation as a string | `expression_simplified[i]` |
| loss · complexity · score · R² | the same four the hero card shows |
| Pareto front chart | re-rendered at print resolution (§4) |
| fit **or** residual chart — whichever view is selected | ditto |

Only the selected fit view is printed, not both: the card shows one at a time, and printing
both would dilute the one page that is supposed to be the answer.

Two pieces of wording are decided at build time rather than fixed, because the printed page
has no interaction to explain itself with:

- **The heading is "Best formula" only when the printed equation IS the recommended one.**
  The app's hero card carries that title while showing whatever row was last clicked, which
  on screen is unambiguous — clicking is what changed it. On paper, an equation headed "best"
  that the table below does not mark ★ is a contradiction the reader cannot resolve. When the
  two differ the heading reads "Selected formula" and a line gives both row numbers.
- **The Pareto caption names the marks the figure actually draws.** On screen the chart is
  read with a legend printed above it; the figure has none. And a point that is both selected
  and recommended is drawn only in the selected colour (`plots.js` `pointColors`), so a
  caption promising two marks would point at one that is not there.

### Page 2 — the evidence

- **All equations**: the full Pareto front, every row, no scroll cap, no ellipsis. Same
  columns as the on-screen table (`#`, complexity, loss, score, R², equation), with the
  recommended row marked ★.
- **Equation tree** of the selected equation: the live SVG, cloned and scaled to the page
  width.

### Page 3 — the reproducibility appendix

- **Operator library**: the binary and unary sets actually sent to the engine, and every macro
  with its body.
- **Search settings**: every field, with its PySR-parity default beside it, and any field that
  was moved off that default marked. This is the same discipline the settings dialog applies
  on screen — a divergence from parity must never be silent.
- **Reproduce this run**: the Python and R snippets, byte-identical to what the copy buttons
  produce (same `pythonCall` / `rCall`).
- **This run**: started at, elapsed seconds, whether it finished its budget or stopped early,
  and the evaluation accounting (total, forward, LM residual, cache hits).
- **Notes** (§5.4) and the licence line.

### 3.1 What is deliberately left out

- **A data preview and per-column statistics.** They restate the CSV the reader already has,
  and a large table would push the report's own content off the page.
- **Both fit views.** See above.
- **A selection UI for which sections to print.** The browser's print dialog already has a
  page range. A second, app-specific one is a control that earns nothing.

## 4. Charts at print resolution

The on-screen canvases cannot be used: they are drawn at the screen's device pixel ratio
(1 on most displays) and sized to the results column. The report re-renders each chart
off-screen and embeds it as a PNG data URI.

Two separate knobs, and confusing them produces bad output:

- **Logical size** (CSS px) fixes the *proportion* of text to plot. A chart rendered at
  760 logical px and printed 88 mm wide has 12 px axis labels that land at about 4 pt on
  paper — unreadable. The print charts are therefore rendered at a **small logical size**
  (440 × 300) close to their physical size, and `Chart.defaults.font.size` is temporarily
  raised to 15 for the off-screen render so labels land near 9 pt.
- **`options.devicePixelRatio`** fixes *sharpness* only. Chart.js honours it
  (`_resize`: `options.devicePixelRatio || platform.getDevicePixelRatio()`), scaling the
  backing store while leaving every size in logical px. Set to 3, giving roughly 480 dpi at
  the printed width.

The off-screen canvas lives in a real but off-viewport container (`position: absolute;
left: -10000px`), not `display: none`: Chart.js's responsive sizing reads the container's
computed size, which a `display: none` parent reports as zero.

That container carries `.print-palette`, which re-declares the light `--chart-*` variables.
`themeColors()` gains an element parameter and reads from it, so a dark-theme user gets
charts drawn for white paper without the app's theme being touched (a temporary
`data-theme` swap on `<html>` would have worked too, but it mutates global state to produce
a local effect).

To share exactly one description of each chart between the screen and the page, `plots.js` is
refactored so the Chart.js **config** is built by a pure function
(`paretoConfig`/`predictionConfig`/`residualConfig`) that both `drawPareto(...)` and
`paretoImage(...)` call. The live draw functions keep their signatures and behaviour.

## 5. Mechanics

### 5.1 Building and printing

```
click "PDF"  ->  buildPrintReport()      synchronous: DOM + chart data URIs
             ->  await every <img>.decode()
             ->  window.print()
afterprint   ->  empty #print-report, drop the class
```

The `decode()` wait is why the button path is async: a data URI is not painted the instant
its `src` is set, and printing before the decode gives blank chart boxes.

Three entry points, one builder:

- **the PDF button** — the supported path, above.
- **Ctrl/Cmd+P** — intercepted when a result exists and routed to the same path, so the most
  common way to start a print gets the guaranteed-complete version. (The app already binds
  Ctrl+Enter for Run, so this is not a new kind of interception.)
- **the browser's own menu** — caught by `beforeprint`, which builds the report
  synchronously. This path cannot await the image decode; it is a best-effort net, and the
  charts may be missing from it in the rare case where it is the first print of a session.

Print-time visibility is gated on a body class the builder adds, so a print started with **no
result at all** falls back to printing the app rather than a blank page.

### 5.2 Colours on paper

Nothing inside `#print-report` may reference the app's theme variables: in dark mode
`--text` is near-white, which prints as invisible text on white paper. The report's own
colours are literal, and the two var-driven pieces it inherits — the equation tree's node
fills and the chart palette — are covered by `.print-palette` re-declaring those variables
on the report container and the chart host.

`print-color-adjust: exact` is set so table shading and node fills survive the browser's
default "ignore backgrounds" print behaviour.

### 5.3 Run facts the app does not currently keep

Four things the report needs are, today, either read and discarded or never captured. Each is
added to `state`:

| fact | today |
|---|---|
| the data's source name (file name / example / pasted) | `file.name` is read only for an error message |
| the tool version | not present anywhere in `web/app/` |
| when the run started, and how long it took | `elapsed` is formatted into the status chip and dropped |
| whether the run finished its budget or stopped early | computed in `onResult`, kept only as chip text |

`APP_VERSION` is a literal in `main.js` that must track
`r-package/rsymbolic2/DESCRIPTION` and `python/pyproject.toml` (both 0.1.0). A build-time
stamp was considered and rejected as disproportionate: it would have to work in the GitHub
Actions build *and* in `web/serve.py`, for one string.

The rest is snapshotted at result time into `state.run`, following the discipline the code
already uses for `featureNames`/`targetName`/`sampling`: a displayed result describes the run
that produced it, and controls the user can still move afterwards must not rewrite it.

### 5.4 The notes the report must carry

A number a reader cannot reproduce is worse than no number. The appendix therefore states:

1. **The browser build is not bit-identical to R/Python.** Emscripten's libm differs from the
   native one in the last bits, and a GP search is sensitive to that, so the two can converge
   to different but equally valid expressions (`docs/51`, already in `web/README.md`). Without
   this, someone re-running the printed snippet and getting another equation reads it as a bug.
2. **The printed equation is the display-simplified form** (`docs/52`); the raw searched
   expression is what `complexity` counts and what the snippets round-trip.
3. **Which rows were fitted**, when sampling was in effect.
4. **`model_selection`**, when it is not PySR's `best` — the GUI's one sanctioned divergence,
   which decides which member of the front the report calls the answer.
5. **The fit chart's point cap**, when the scatter is a subset of the fitted rows.

## 6. Files

| file | change |
|---|---|
| `web/app/js/report.js` | **new** — builds the report DOM from a plain context object |
| `web/app/js/plots.js` | extract config builders; `themeColors(root)`; add the three `*Image()` exports |
| `web/app/js/main.js` | `APP_VERSION`, `state.dataSource`/`state.run`, print wiring, `reportContext()` |
| `web/app/index.html` | the PDF button; the empty `#print-report` container |
| `web/app/css/style.css` | `.print-palette`, the off-screen chart host, the `@media print` block |
| `web/README.md` | document the report |

## 7. Verification

Done in the browser (Windows 11, Chromium via Playwright), against `web/serve.py`:

| check | result |
|---|---|
| `web/wasm/test/parity_test.cjs` | PASSED — the engine is untouched |
| report built from a real run | all six sections present, 1 KaTeX equation, 14-row front, 22-setting table, tree SVG, 3 code blocks |
| **an actual PDF**, via `page.pdf({format: 'A4'})` | 269 KB, **4 pages**, section breaks landing where §3 says |
| chart resolution | images 1320 px wide (440 logical × `devicePixelRatio` 3) — about 380 dpi at the printed width, i.e. Chart.js does honour the option |
| dark theme | report renders `#111` on white, tree fills and chart palette light — `.print-palette` holds |
| deep tree (16 nodes, 800 px wide) | scaled to the 186 mm text width, not clipped |
| non-recommended row selected | heading becomes "Selected formula", both row numbers stated, Pareto caption names two marks |
| sampled run (400 of 6,000 rows, uploaded CSV) | file name, sampling banner, sampling note and the snippet's sampling comment all present |
| a setting moved off parity | `generations` and `model_selection` both marked ✱ |
| no result | button disabled and Ctrl+P **not** claimed — the browser's own print runs |
| button / Ctrl+P / `beforeprint` / `afterprint` | all four fire the intended path; `afterprint` empties the container |
| console | no errors, no warnings |

Ubuntu: nothing compiles here — the change is HTML/CSS/JS in a static site, and the milestone
check is the same Node parity gate, which CI also runs before deploying.

One thing automated testing cannot cover: `window.print()` opens the browser's real print
dialog and blocks, which is the feature working. The automated runs stub `print` and assert
*when* it was called and what the DOM held at that moment.
