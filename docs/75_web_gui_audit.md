# 75 — Web GUI audit

Audit of the fourth binding — the browser GUI (`web/app/`, `web/wasm/rsymbolic2_wasm.cpp`) —
following the C++ core audit in docs/73 and the R/Python binding audit in docs/74.

Method: read the five thousand lines of `web/app/`, then drive the running page (dev server +
headless browser) through the paths the reading made suspect. Every finding below was
reproduced in a browser before being fixed, and re-checked there after.

The headline is the same one docs/74 found, one binding over: **the WASM bridge validated its
options carefully and its data not at all** — because it was never given the guards the other
two bridges received. The three bridges are documented as deliberately symmetric
(`rsymbolic2_wasm.cpp` header comment); this closes the asymmetry. The search itself is
untouched: `parity_test.cjs` reports the same recovered expression and the same Pareto losses
before and after.

## Fixed

### 1. The WASM bridge had none of the validation R and Python received (most severe)

docs/74 added positive-count guards to both shipped bindings and to `run_evolution()`. The
core guard cannot catch the case it was written for — by the time a count reaches
`SearchOptions` it is a `std::size_t`, and a wrapped `-1` is indistinguishable from a
deliberate huge request — which is exactly why the *bindings* check while the value is still a
signed `int`. The WASM bridge, written before that commit, checked none of them.

In the browser this is not a hypothetical: every one of these fields is a text box a user can
type into, and `min` on a number input constrains the spinner, not what can be typed.

| typed into the settings dialog | what happened |
| --- | --- |
| `Population size = -1` | `error: vector` — the bare `std::length_error::what()` |
| `Tournament size = -1` | `error: vector` |
| `Generations = -1` | a run of ~1.8e19 generations; in a tab, indistinguishable from a hang |
| `Max depth = -1` | ran, and searched a space no tree can satisfy |

`max_nodes` was the one already covered, by the core guard docs/73 added — it is an `int` all
the way down, so the sign survives to be checked.

Fixed in two layers, matching the shape docs/74 settled on:

- **`web/wasm/rsymbolic2_wasm.cpp`** — the same six guards the R and Python bridges carry
  (`population_size`, `generations`, `tournament_size`, `max_nodes`, `max_depth`,
  `n_populations`), checked while the values are still signed ints, plus the `X`/`y`
  finiteness check. This is the boundary guard, and it holds for any caller of `Module.run()`,
  not only this GUI.
- **`main.js::settingError()`** — the pre-flight check, so the user reads
  *"Population size must be at least 1 (it is -1)."* naming the field as the dialog labels it,
  and no run is launched at all. Driven off each input's own `min` attribute rather than a
  second table of bounds, so a field added later is covered by the `min` its markup already
  needs.

The `X`/`y` finiteness check cannot fire from this GUI — a column holding one non-finite cell
is not offered as a feature or a target (see 3 below) — but it is what makes the three bridges
say the same thing, and `Module.run()` is a public entry point.

New coverage in `web/wasm/test/parity_test.cjs` (the CI gate): each of the six counts at `0`
and `-1`, asserting the message *names the argument* — `"vector"` was an error too, and that
was the defect.

### 2. The vendored WASM was older than the core it is built from

`web/app/vendor/rsymbolic2.{js,wasm}` are tracked in git and were last committed two core
commits ago, so the local dev loop was running an engine without docs/73's fixes. Live users
were unaffected — `deploy-pages.yml` rebuilds from source on every push touching
`r-package/rsymbolic2/src/**` — but anyone serving `web/app/` locally got the stale engine
silently, which is the failure mode `web/serve.py` exists to prevent for the *other* files.

Two of the drifted fixes are reachable from GUI controls: `linear_scaling` is a sidebar
checkbox, and the macro parser's unary-minus precedence decides what the shipped `stretchexp`
preset (`exp(-x^2.0)`) means — under the old parser it was `exp((-x)^2)`, i.e. growth where the
preset's own tooltip promises decay.

Rebuilt and committed. This drift is structural rather than a one-off: the artifact is tracked,
CI rebuilds it, and nothing fails when the two disagree.

### 3. Columns and rows disappeared at intake without a word

A column is usable only if **every** cell parses as a finite number (`data.js
numericColumns`). One blank, `NA`, `n/a`, `1,5` or stray unit therefore removes the whole
column from Target *and* Features — while the summary line went on counting it
("5 rows × 3 columns") and the status said "data loaded — press Run". The user was left to
notice the absence.

The parser drops rows whose field count does not match the header, equally silently.

Both are the right call; neither was worth staying quiet about. `#intake-notice` now reports
them once per load, and names the **first offending cell** rather than only the column —
*"pressure" (row 2 is "NA")* sends the user to a cell, `pressure is not numeric` sends them
through a column. Beyond three dropped columns it counts the rest instead of listing them.

Kept separate from `#data-notice`, which answers a different question: that one says what will
be *fitted* and offers to change it, this one says what the intake could not use at all.

### 4. "Select at least one feature" when there was no feature to select

Pressing Run with a one-column file, or with a table whose other columns all hold one
non-numeric cell, asked the user to do something the page cannot do — and it is precisely the
state finding 3 leaves you in. Run now distinguishes "you unticked them all" from "there are
none", and the second case says what a feature actually requires.

### 5. The example picker kept a stale selection

`#example-select` stayed on the entry it loaded. That broke it in both directions: a `<select>`
fires no `change` event when the already-selected option is picked again, so an example could
never be re-loaded once other data had replaced it — and until then the rail sat there naming a
dataset that was no longer loaded. Now resets to its placeholder after loading, the same way
`#macro-preset` already did. It is a loader; the data summary line is the statement of what is
loaded.

### 6. Two labels that described the wrong thing

- The equation tree's node count said the complexity column "can be larger". Since docs/71 it
  can also be *smaller*: the engine prints one-child nodes in notation, so a `Square` it counts
  as two nodes prints `(x0 ^ 2)` and re-parses as three — a default run shows "complexity 7 → 6"
  beside "7 nodes". The count is right (the tree draws what is printed, so it counts what is
  printed); the caption was one-directional. Stated both ways now, with the reason.
- `Max nodes depth` in the settings dialog is `max_depth`. The name reads as a field that does
  not exist, and finding 1 put it in a user-facing error message. Now `Max depth`; the node
  limit is `Max nodes`, two groups above.

## Checked and correct — do not re-audit

- **`selectBestIndex()` (`main.js`) against `select_best()` (`hall_of_fame.cpp`).** Faithful
  port, including the `bestIdx = 0` fallback and the `-Infinity` seed that makes `scores[0] = 0`
  and `NaN` lose every comparison. Verified by switching `model_selection` through all three
  modes on a live front.
- **`predict.js` against the engine.** Grammar, `erf`, and the safe-pow caveat are all pinned by
  `parity_test.cjs`; the end-to-end assertion recomputes the engine's own reported loss from the
  returned expression.
- **Escaping.** Every user-supplied string (column names, expressions, macro bodies) reaches the
  DOM through `textContent`, `esc()` (text *and* attribute contexts) or `escapeLatexText()`.
  `report.js` is `textContent` throughout.
- **The transactional settings dialog.** Snapshot on open, restore on all four dismissal paths,
  commit on close; `readSettingsFields`/`writeSettingsFields` cover the checkboxes too. Verified
  by editing and cancelling.
- **Settings persistence.** Whitelist-then-restore round-trips operators, macros, opt-ins and
  fields; a bad blob costs the shipped defaults. The arrival notice fires only on a real
  divergence. Verified across a reload.
- **The row policy.** Forced sampling when the table exceeds the ceiling, the disabled "All
  rows" radio, the lock releasing when the ceiling rises again, and `reapplyRowPolicy()`
  invalidating a displayed result only when the fitted ROWS moved. Exercised by shrinking the
  ceiling through `Populations`.
- **Run lifecycle.** Stop with and without a previous result, error, load-data-mid-run; the live
  front is put back or cleared and never left reading as the final answer.
- **The printable report.** Builds through both entry points, both figures render, the settings
  table marks the one sanctioned divergence (`model_selection = score`), and the R/Python
  snippets are syntactically correct with and without macros and opt-ins.

## Verification

- `node web/wasm/test/parity_test.cjs` — PASSED, including the 14 new validation assertions.
  Recovered expression and Pareto losses byte-identical to before the bridge change, so the
  default path is untouched.
- Browser: default run, all three `model_selection` modes, fit/residual views, both loss axes,
  theme toggle, preview dialog, settings dialog (edit/cancel/reset), macro add/preset/remove/
  validation, sampling, stop/restart, print report, settings persistence across reload. No
  console errors.
- The shared C++ core, the R package and the Python package are untouched by this work; the only
  compiled artifact is the emscripten build, which CI rebuilds and re-gates on Ubuntu for every
  push.
