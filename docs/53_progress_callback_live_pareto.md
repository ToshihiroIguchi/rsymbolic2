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
- **(d) Finished.** `finishRun()` (called from all three end-of-run paths — Stop,
  result, error) removes both `body.running` and the Pareto card's `.live` class, so
  a Stop mid-run leaves no live state behind. `onResult()` → `renderResult()` is the
  authoritative full render: it destroys and recreates the Chart.js instance with the
  complete final front (`score`, `bestIndex`, `selectedIndex`, `onSelect` all
  present), fully replacing whatever the live chart last showed; `renderResult()`
  also defensively clears `.live` itself, so it is self-contained even if a future
  caller invokes it without going through `finishRun()` first.

## Tests

- `standalone/tests/test_progress_callback.cpp` (registered in
  `standalone/CMakeLists.txt` as ctest `progress_callback`): bit-identity with the
  callback unset vs. attached; fire-count bounded by `ceil(generations /
  migration_interval)`; per-snapshot shape (`complexity`/`loss` equal length,
  complexity strictly increasing — the Pareto-front invariant).
- `web/wasm/test/parity_test.cjs`: `on_progress` fires at least once on a
  multi-iteration config, and attaching it does not change the recovered expression
  or Pareto losses (same assertions as the standalone test, exercised through the
  WASM binding). This run also covers the docs/52 display-simplification fields
  (`expression_simplified`/`latex_simplified`) for the WASM binding, which had no
  dedicated WASM-level test before.
