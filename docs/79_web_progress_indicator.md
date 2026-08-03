<!--
SPDX-License-Identifier: Apache-2.0
Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
-->

# 79 · Web GUI progress indicator: what it answers, and what it did not

Status: implemented. Six changes to the in-run feedback of the web GUI, all confined to
`web/app/` (JS/HTML/CSS). **The search engine, the WASM binding and the vendored
`rsymbolic2.wasm` are untouched**, so nothing here can move a search trajectory, and the
PySR default-parity rule (CLAUDE.md) is not engaged at all: no default changes, no core
change, no rebuild.

## The starting point (audit)

Progress had exactly one source: the per-epoch `progress_callback` of docs/53, forwarded
by `web/wasm/rsymbolic2_wasm.cpp` as `{epoch, total_epochs, complexity, loss, expression}`
and consumed by `main.js onProgress()`. An epoch is `migration_interval = 28` generations
(`evolutionary_search.hpp:142`), so at the shipped `generations = 2800` the UI receives
100 snapshots — one per 1 % of the budget.

That fed five displays: the header bar (indeterminate sweep until the first snapshot, then
`epoch/total_epochs`), the status chip (`31.8s | epoch 63/100 · ≤ 12m 20s left`), the live
Pareto redraw, the provisional lowest-loss expression, and the first-run `#pending-cards`
note.

The structure was sound — one writer for the ticking chip, the abort paths restore the
completed run's chart, the ticking text deliberately stays out of the live region, the ETA
is honestly an upper bound. Six gaps were real:

1. **Mixed units.** The sidebar says "2800 generations", the chip said "epoch 63/100", and
   the factor 28 (`migration_interval`) appears nowhere in the UI — it is not even an
   exposed setting. Two numbers the user cannot connect.
2. **Budget consumed ≠ progress toward an answer.** The only mid-run decision a user makes
   is whether to press Stop, and that needs "is it still improving?". The live Pareto is
   redrawn from scratch each epoch and carries **no history**, so a plateau is invisible.
3. **One budget tracked, several enforced.** With `timeout_seconds > 0` the run ends at
   whichever of the epoch budget and the wall clock arrives first; the bar tracked only the
   former and could sit at 40 % when the run ended.
4. **A blind opening.** Between Run and the first snapshot lie the worker spawn, a 496 KB
   WASM fetch + compile + instantiate, the data transfer and the first 28 generations — all
   of it one undifferentiated sweep. `worker.js` already posted `{type:"ready"}`; `main.js`
   discarded it, and `getModule()` is lazy so `ready` did not even mean the engine was there.
5. **Silence for screen readers.** The bar is `aria-hidden`, the chip opts out of
   announcement (correctly — it rewrites 5×/s), and `#status-live` only ever received
   discrete messages. A blind user heard "running…" and then nothing for the length of the
   run, which the code's own example puts at 12 minutes.
6. **No signal in a background tab.** Nothing in the tab title or favicon; a long run was
   invisible unless the tab was in front.

## What was implemented

### 1 · Generations, everywhere (gap 1)

The chip now reads `31.8s | generation 1764/2800 · ≤ 12m 20s left`. Epochs remain the
engine's unit internally (`state.epoch`, `state.totalEpochs` are still exactly what the
snapshot said); the *display* converts, and only for display:

```js
gen = min(total, round(epoch * generations / total_epochs))
```

`generations / total_epochs` recovers `migration_interval` exactly whenever the budget
divides by it (2800/100 = 28) and approximates it otherwise; the `min` clamps the last
epoch so a rounding artefact can never print `2801/2800`. Deriving it in JS is deliberate:
adding `generations_per_epoch` to the snapshot would have meant editing the WASM binding
and rebuilding the vendored `.wasm` for one integer — and docs/75 records how easily a
locally rebuilt WASM goes stale against CI's.

The chip's `title` states the cadence in the same breath: *"Updated every 28 generations
(one migration epoch)."* The early-stop chip and the report's "This run" block converted
to the same unit, so no surface says "epoch" to a user any more.

### 2 · A convergence trace (gap 2)

`state.lossTrace` accumulates one point per epoch — the front's minimum loss, which is
already in the message — and a small sparkline is drawn above the live Pareto chart:
`plots.js drawLossTrace()`, a plain 2D-canvas polyline on log10(loss), **not** Chart.js.
Three reasons: it carries history, so the destroy-and-rebuild the other charts use
(`plots.js:139`) is exactly wrong for it; a 40 px sparkline needs none of Chart.js's axis
machinery; and it keeps the ≤4 Hz redraw cheap. Non-finite and non-positive losses are
floored (`1e-300`) rather than dropped, so the polyline never gains a phantom gap, and a
flat trace draws down the middle instead of dividing by a zero range.

Beside it, the fact the sparkline exists to convey, in words:

- `best loss 3.42e-04 · improving`
- `best loss 3.42e-04 · no improvement for 252 generations`

The stall counter is the honest answer to "should I stop?", and it is readable without
being able to see the sparkline at all (which is also what makes it screen-reader-visible
in the DOM, unlike the canvas).

### 3 · Progress is the *nearest* deadline (gap 3)

One `progressFraction()` now feeds both the bar and the ETA:

```
frac = max(epoch/total_epochs, elapsedSearch/timeout_seconds)
eta  = min(recent-epoch-rate estimate, timeout_seconds - elapsedSearch)
```

with each term dropped when its budget is off. Two consequences: with a timeout set, the
bar goes determinate as soon as the engine starts — before any epoch has completed, since
the wall clock needs no snapshot — and it cannot be caught at 40 % by a timeout stop.
`elapsedSearch` is measured
from the worker's `stage: "search"` message, not from the Run click, because the engine's
timeout clock starts when `run_evolution` does — counting the WASM load against it would
retire the bar early. `target_loss` (on by default at 1e-10) stays unpredictable by
construction; the trace of change 2 is what makes an imminent target-loss stop visible.

Bar updates moved out of `onProgress` into the 200 ms timer that already owns the chip.
That is the existing "one writer" rule applied to the second display: with two budgets the
bar has to advance between snapshots, so a snapshot-driven bar could no longer be correct.

### 4 · The opening is narrated (gap 4)

`worker.js` posts `{type:"stage", stage:"engine"}` before `await getModule()` and
`{stage:"search"}` after it, immediately before `Module.run()`. The chip shows
`0.4s | loading engine…` for the first stretch. The unused `ready` message is left alone —
it means "the worker script parsed", which is a different (and weaker) claim.

Note this exposes a real cost rather than hiding one: a fresh worker per run (needed for
Stop == `terminate()`) reloads the engine every run. Making that a visible 0.2–0.5 s beat
is the point; changing it is out of scope here.

### 5 · Coarse announcements (gap 5)

At 25 %, 50 % and 75 % the run writes one sentence to `#status-live`
(`search 50% complete, about 6 min left`). The chip keeps its `announce = false` path, so
the two channels never fight: during a run the milestones own the live region, and
`setStatus()`'s discrete messages own it outside one. Three announcements over a run is the
most a live region can carry without becoming what the original comment refused to build.

### 6 · The tab title tracks the run (gap 6)

`42% · rsymbolic2` while determinate, `running · rsymbolic2` before that, restored by
`finishRun()`. Written only when the integer percent changes. The Notification API was
rejected: a permission prompt is a poor trade for a signal the tab title already carries.

## Verification

`web/serve.py` + Playwright (Chromium), on the bundled damped-oscillation example, sampling
the live DOM every 100–120 ms through each run.

**Full-budget run** (`target_loss = 0`, so the epoch budget is what ends it):

```
[0.2s] 0.2s | loading engine…                                    det=false  title="running · rsymbolic2…"
[1.0s] 1.0s | generation   84/2,800 · ≤ 9 s left   bar=  9.0%     det=true   title="9% · rsymbolic2…"
[5.4s] 5.4s | generation 1,540/2,800 · ≤ 5 s left  bar= 55.0%     det=true   best loss 5.190e-12 · no improvement for 532 generations
[9.8s] 9.8s | generation 2,744/2,800 · ≤ 1 s left  bar= 98.0%     det=true
       9.54s | generations: 2,800                                 det=false  title restored
announced to #status-live: "search 25% complete, about 8 s left" / "50% … 5 s" / "75% … 3 s"
status chip title: "Updated every 28 generations (one migration epoch)."
```

**`timeout_seconds = 4`** — the wall clock is the nearer deadline throughout, and the epoch
term (generation 84/2,800 = 3 %) never wins:

```
[1.0s] generation    84/2,800 · ≤ 4 s left  bar=  9.8%   det=true
[3.0s] generation   616/2,800 · ≤ 2 s left  bar= 59.6%   det=true
[4.0s] 4.01s | stopped early: generation 1,120/2,800     bar= 99.6%
```

The bar reaches 99.6 % exactly as the run ends — the case that used to stop at 40 %.

**Other paths.** Stop mid-run (`2.4s | generation 532/2,800 · ≤ 8 s left` → chip "stopped",
title restored, stall counter and provisional formula both cleared, live card taken down).
Early stop on the shipped `target_loss = 1e-10`: chip `1.21s | stopped early: generation
364/2,800`, and the printed report's "This run" block agrees — *"ended: early, at generation
364 of 2,800 (target loss, timeout or max evals)"*. First run of a session (no previous
result to fall back on): the promoted Pareto card carries the sparkline and the note.
Mid-run theme toggle repaints the sparkline synchronously through `redrawCharts()` — the
stroke sampled off the canvas moves `#4f8dff` → `#2563eb` while `body.running` is still set.
390 px viewport: chip, sparkline and stall counter all wrap and remain legible.

Console clean (0 errors, 0 warnings) across every run; the one warning seen was
`getImageData … willReadFrequently` from the test harness's own pixel sampling, not the app.

**Search behaviour is untouched by construction**: nothing outside `web/app/` changed, no
file under `web/wasm/` was touched, the vendored `.wasm` is the same binary, and no new code
reaches the engine's options — the WASM binding is called with exactly the object it was
called with before.
