<!--
SPDX-License-Identifier: Apache-2.0
Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
-->

# 53 · Progress callback + live Pareto-chart updates

Status: implemented. A behaviour-neutral, opt-in observation hook in the C++ core,
wired through the WASM binding to drive a live-updating Pareto chart in the web GUI
while a search is running. This closes the "Live per-epoch Pareto/best-loss updates"
item deferred in docs/51.

## The callback contract

`SearchOptions::progress_callback` (`rsymbolic/search/evolutionary_search.hpp`):

```cpp
struct ProgressSnapshot {
    std::size_t epoch = 0;          // completed outer iterations
    std::vector<int> complexity;    // current global pareto front
    std::vector<double> loss;
    std::vector<Tree> tree;         // phase 2; same order/length as the two above
};
// on SearchOptions:
std::function<void(const ProgressSnapshot&)> progress_callback;  // null by default
```

- **Pure observation.** The callback is handed a read-only snapshot of the current
  global Pareto front. Nothing in the search reads the callback's return value, and
  invoking it consumes no RNG draws and mutates no search state. Attaching an observer
  therefore cannot change *which* candidates the search finds.
- **Null default = bit-identical.** With `progress_callback` unset (the default,
  constructed by `SearchOptions`'s default member initializer), the call site is a
  single untaken `if (options.progress_callback)` branch — no snapshot is built, no
  extra `HallOfFame` merge runs. This satisfies PySR Default Parity (CLAUDE.md): the
  shipped default search is unaffected by this feature's existence.
  `standalone/tests/test_progress_callback.cpp` asserts this directly — the same seed
  and options produce a byte-identical `expression`/`loss`/`pareto_front` whether or
  not a (counting, no-op) callback is attached.
- **The exact seam.** `r-package/rsymbolic2/src/evolutionary_search.cpp`, inside
  `run_evolution`'s outer epoch loop, immediately after `migrate_hof(...)` and before
  the global early-stop check. At this point every island's OpenMP worker for the
  epoch has joined (the `#pragma omp parallel for` region above has completed), and
  both ring migration (`migrate`) and hall-of-fame migration (`migrate_hof`) for the
  epoch are done — so every island's per-island `HallOfFame` is quiescent and it is
  safe to merge them on the single orchestrating thread without touching any
  in-flight island state. The seam merges a local `HallOfFame` from each island's
  `isl.hof` (mirroring the final merge at the end of `run_evolution`), reads its
  `pareto_front()`, and fills a `ProgressSnapshot` (`epoch` is the count of *completed*
  outer iterations, i.e. the loop's own `epoch` counter + 1, since the loop's
  `++epoch` runs after this point in the body).
- **Per-epoch, not per-mutation.** The callback fires once per outer epoch (the unit
  bounded by `migration_interval`, default 28 generations — see docs/28 §C's
  cycle-mapping note), not once per individual mutation/evaluation. This bounds the
  observation overhead to the same cadence as HOF migration itself.

## Why R and Python leave it null (deliberate, for now)

Only the WASM binding (`web/wasm/rsymbolic2_wasm.cpp`) wires `progress_callback`. The
R and Python bindings do not expose it yet. This is a scope decision, not an
oversight:

- The web GUI's motivating use case — a single-threaded, single-shot, in-browser run
  with no way to inspect progress except the page itself — does not apply to R/Python,
  where the process is scriptable and the existing `verbosity=1` per-epoch stderr line
  already gives progress feedback during a long call.
- R and Python callbacks would need to cross a different FFI boundary (R's C API /
  Python's GIL) with their own reentrancy and threading concerns (the seam executes
  once per epoch from whichever thread called `run_evolution`, and R/Python calls are
  typically synchronous/blocking from the caller's perspective already, unlike the
  WASM binding's single dedicated worker thread) — that design is deferred until there
  is a concrete need, per the project's anti-speculative-scope rule.
- Because the core hook already exists and is behaviour-neutral, adding R/Python
  wiring later is a small, additive, non-breaking change: no core rework is implied by
  today's WASM-only scope.

## The WASM / worker / main-thread protocol

1. **`web/wasm/rsymbolic2_wasm.cpp`** (`run(val opts)`): if `opts.on_progress` is
   present, it is captured **by value** into a C++ lambda assigned to
   `o.progress_callback`. This is safe because `run_evolution()` completes
   synchronously inside this same `run()` call — the captured `emscripten::val` never
   outlives the call that created it, so there is no dangling-reference risk despite
   the WASM/JS value not being reference-counted across calls. The WASM build is
   single-threaded (no pthreads, no OpenMP — see docs/51 "Feasibility"), so the
   callback fires synchronously, on the same thread that is running `Module.run()`
   (the worker thread — see below), never concurrently with the search itself.
2. **`web/app/js/worker.js`**: before calling `Module.run(opts)`, sets
   `opts.on_progress = (snap) => self.postMessage({ type: "progress", epoch, complexity, loss })`.
   Each snapshot becomes one extra `postMessage` alongside the existing `ready` /
   `result` / `error` message types — no new message channel.
3. **`web/app/js/main.js`**: the worker's `onmessage` dispatch gains a `"progress"`
   case → `onProgress(msg)`. To avoid flooding the main thread with Chart.js
   rebuilds on a run with many small epochs, redraws are **throttled to at least
   250 ms apart** (`state.lastProgressDraw`, a `performance.now()` timestamp reset at
   the start of each run). Every snapshot still marks the Pareto card `.live`
   (cheap, unthrottled), but only a throttle-eligible snapshot triggers
   `drawPareto()`.
4. `drawPareto()` (`web/app/js/plots.js`) is reused unchanged for both the live and
   final render paths; a live redraw passes a minimal front
   `{ complexity, loss, score: null }` with no `bestIndex` / `selectedIndex` /
   `onSelect`. The function was made robust to those omissions: `front.score`
   absent/null degrades the tooltip to `complexity, loss` (no `score` term), and a
   missing `onSelect`/`bestIndex`/`selectedIndex` simply means no point is
   highlighted and clicks are inert — which matches the CSS (`pointer-events: none`
   on `#results-area` while `body.running`; live points carry no expression to select
   anyway). The final-result draw path is unchanged pixel-for-pixel: it always passes
   a full `front.score` array, so its tooltip and highlighting are untouched.

## GUI states

- **(a) Idle with an old result.** Normal: the previous run's cards are fully opaque
  and interactive (`renderResult()` has already run; no `body.running`, no `.live`).
- **(b) Running, no snapshot yet.** `body.running` is set (`main.js` `run()`); the CSS
  rule in `web/app/css/style.css` (`body.running #results-area .card { opacity:
  0.45; }`, with `pointer-events: none` on the container) dims every result card and
  blocks interaction — the stale previous-run numbers are visibly inert.
- **(c) Running with snapshots.** `main.js` `onProgress()` adds a `.live` class to the
  Pareto card (`#pareto-card` in `index.html`) the moment the first snapshot arrives.
  A sibling CSS rule (`body.running #results-area .card.live { opacity: 1; }`)
  restores that one card to full opacity — note the dim had to move from the
  `#results-area` *container* onto each `.card` individually, because a parent's
  `opacity` composites its children regardless of their own `opacity` value, so a
  child could never "opt back out" of a dimmed ancestor. `pointer-events: none`
  still applies (inherited from the container, not per-card), so the live points
  are visibly updating but not clickable — a `::after` badge reading "updating…" is
  shown on the card only while both `body.running` and `.live` hold.
- **(c′) Running with snapshots, but no result to fall back on.** States (a)–(c)
  above silently assume a previous run: the progressive-disclosure rule
  `#results-area:not(.has-result) .card:not(.placeholder) { display: none; }` keeps
  every result card hidden behind the empty-state placeholder until the first run
  *finishes*, so on the first run of a session — and on any run after the data
  changed and `clearResults()` took `.has-result` off again — the live snapshots were
  being drawn into a `display: none` canvas. The user watched an unchanged "Results
  appear here after a run." placeholder for the whole search (the header progress bar
  and the epoch/ETA status line were the only signs of life), and the live front
  appeared only from the *second* run on — i.e. the feature was suppressed exactly
  where there was nothing else to look at. `onProgress()` therefore also adds
  `.has-live` to `#results-area`, which promotes the Pareto card alone
  (`#results-area:not(.has-result).has-live #pareto-card { display: flex; }`, plus
  hiding the placeholder and collapsing `.charts-row` to one column so the card is
  not left beside an empty grid cell). Every other result card stays hidden because
  it has nothing in it until the run ends. The class is scoped under
  `:not(.has-result)`, so once a result exists it changes nothing and states (a)–(c)
  are byte-for-byte the behaviour described above.
- **(c′) addendum: naming the panels that are still missing.** Hiding the other cards
  is correct — a snapshot carries complexity and loss but no equations, and each of
  those cards is about one *chosen equation* — but on its own it left the column
  silently short. A first-time user saw one chart, no indication that three more
  panels exist, and then a four-card layout appearing all at once the instant the run
  ended; the reasonable reading was "the other charts are broken" rather than "they
  need a finished equation". `index.html #pending-cards` (styled in `style.css` beside
  the three rules above) is a dashed note under the promoted Pareto card that names
  the four pending panels as ghost chips — Best formula, Predicted vs actual, All
  equations, Equation tree — and states in one sentence why they are empty. It is
  scoped to the *same* selector as the (c′) reveal
  (`#results-area:not(.has-result).has-live`), so it appears and disappears with that
  reveal and cannot show up in states (a)–(d). It is deliberately **not** a `.card`:
  the two rules that govern result cards mid-run (hidden until `.has-result`, dimmed
  to `0.45` under `body.running`) would both apply and are both wrong for it — it is
  not a stale result, and it exists only while those cards do not. No JS: nothing in
  `main.js` traverses the result cards, so the four `classList` calls on
  `#results-area` remain the whole mechanism.
- **(d) Finished.** `finishRun()` (called from all three end-of-run paths — Stop,
  result, error) removes `body.running`, the Pareto card's `.live` class and the
  `.has-live` reveal of (c′), so a Stop mid-run leaves no live state behind — a first
  run that is stopped or fails goes back to the placeholder, since
  `restoreResultCharts()` has no completed front to put back and a partial one must
  not read as the answer. `onResult()` → `renderResult()` is the
  authoritative full render: it destroys and recreates the Chart.js instance with the
  complete final front (`score`, `bestIndex`, `selectedIndex`, `onSelect` all
  present), fully replacing whatever the live chart last showed; `renderResult()`
  also defensively clears `.live` itself, so it is self-contained even if a future
  caller invokes it without going through `finishRun()` first.

## Phase 2 — the snapshot carries the equations, and the card shows one

Phase 1 (everything above) gave the running search a live *chart*. The equations
themselves stayed invisible until the run ended, because the snapshot carried only
complexity and loss. That is what phase 2 changes.

### What the core sends: a third parallel array, not a "current best"

`ProgressSnapshot` gains `std::vector<Tree> tree`, filled in the same loop as the other
two so that `tree[i]` is the member `complexity[i]`/`loss[i]` describe — and, since
complexity is `tree.size()` everywhere in the core, `tree[i].size() == complexity[i]`.

It is deliberately **not** a single `best_tree`. Deciding which member is best is a
selection rule — `pareto_scores` + a `model_selection` accuracy band
(`hall_of_fame.cpp` `select_best`) — and docs/48 put that rule in C++ precisely so the
project has exactly one answer to "which equation is recommended". A "best" field here
would either move that rule into what is documented above as pure observation data, or
invite a caller to re-derive it and diverge. The snapshot reports the front; callers
choose.

Nothing else changes: the fill is inside the existing `if (options.progress_callback)`
branch, so the R and Python bindings — which still leave the callback unset — execute
the same untaken branch as before and are unaffected, bit-for-bit.

### What the binding sends: one string, the lowest-loss member, raw

`rsymbolic2_wasm.cpp` adds one field to the JS object, `expression`, taken from
`s.tree.back()`. Three choices are worth recording:

- **The last member** is the lowest-loss one — the front is strictly increasing in
  complexity and strictly decreasing in loss, the same invariant `select_best` relies on
  when `ModelSelection::Accuracy` returns `front.size() - 1`. It needs no scores and no
  model_selection, so the GUI gets a well-defined equation without a second copy of the
  recommendation rule.
- **One, not all.** Stringifying the whole front costs ~28x more (see below) for output
  the GUI does not display.
- **Raw `to_string`, never `display_simplify`.** The simplifier is an e-graph with a
  10 ms per-call budget (docs/54): up to ~300 ms per epoch over a full front, taken
  straight out of a single-threaded search. The consequence is deliberate and visible:
  a mid-run string can differ cosmetically from the finished hero card's rendering of
  the same tree (`(x0 * x0)` vs `(x0 ^ 2)`).

### What the GUI shows

`#live-expr` on the Pareto card: the label "lowest loss so far" and the raw expression in
a monospace box that ellipsises. Shown on exactly the condition that shows the
"updating…" badge (`body.running` + `.live`), written under the same 250 ms throttle as
the chart redraw so the line and the points always describe the same epoch, and cleared
by `finishRun()` and `clearResults()`.

What it is *not* is the point. No KaTeX, no copy button, no metrics, and not the hero
card: a mid-run front is usually immature and the hero card's typography is the page
saying "this is the answer". Lending that authority to an equation that will be
overwritten a hundred times is how a screenshot of epoch 3 becomes someone's result. The
Pareto chart works as a live display because moving points read as convergence and cannot
be misread as a claim; a formula needs the weight dialled down to earn the same licence.

It sits **above** the chart rather than below it. Below, it shared a line with the
"updating…" badge (pinned bottom-right) and had to reserve ~100 px for it, which at phone
width left about a dozen characters before the ellipsis — measured 119 px of usable width
against 223 px in the final placement, on a 375 px viewport.

### Cost (measured)

Micro-benchmark, g++ 14.3.0 `-O2`, Windows, on a front built by the engine's own
`gen_random_tree_fixed_size` in the real shape (30 members, complexities 1..30, 465 nodes
total), median of two runs:

| per epoch | cost |
| --- | --- |
| (B) copying the front's trees into the snapshot | **1.2 µs** |
| (C) `to_string` on one member (what ships) | **2.5 µs** |
| `to_string` on all 30 members (rejected) | 71 µs |

Against epoch times measured in the browser on the shipped examples — 165 ms (Quadratic,
40 rows; the shortest epoch of any bundled dataset), 232 ms (Noisy linear), 286 ms
(Gravity) — the shipped combination is **~0.002 % of the shortest epoch**, or ~0.01 % if
WASM is assumed 5x slower than native. Stringifying the whole front would still have been
only ~0.04 %, so this was a tidiness decision, not a rescue. R/Python/native pay nothing
at all: the callback is unset there and the whole block is skipped.

## Tests

- `standalone/tests/test_progress_callback.cpp` (registered in
  `standalone/CMakeLists.txt` as ctest `progress_callback`): bit-identity with the
  callback unset vs. attached; fire-count bounded by `ceil(generations /
  migration_interval)`; per-snapshot shape (`complexity`/`loss`/`tree` equal length,
  complexity strictly increasing — the Pareto-front invariant — and `tree[i].size() ==
  complexity[i]`, which pins the three arrays to one another and would catch a fill loop
  that fell out of step).
- `web/wasm/test/parity_test.cjs`: `on_progress` fires at least once on a
  multi-iteration config, and attaching it does not change the recovered expression
  or Pareto losses (same assertions as the standalone test, exercised through the
  WASM binding). Phase 2 adds: every snapshot carries a non-empty `expression`, its
  `complexity`/`loss` arrays are non-empty and equal length, and the last front member
  is the lowest-loss one — the invariant the binding relies on when it takes
  `front.back()` as the equation to show. This run also covers the docs/52 display-simplification fields
  (`expression_simplified`/`latex_simplified`) for the WASM binding, which had no
  dedicated WASM-level test before.
