# 59. Web GUI: behaviour on large data, and the limits it now enforces

*Status: measured. Supersedes nothing; extends `docs/51` (web GUI) and `docs/53`
(progress callback).*

The browser build (`web/`) runs the same C++ engine as the R and Python packages, but
under two constraints the native builds do not have: it is **single-threaded**, and its
WASM heap is **fixed at 128 MB** with `ALLOW_MEMORY_GROWTH` off. This note records what
actually happens when a large table is loaded, the measurements behind the limits the GUI
now applies, and the options that were considered and rejected.

Everything here is measurement, not estimation, except where marked "extrapolated". All
numbers come from the Emscripten 6.0.2 build of the current core, run under Node v24 on
the development machine (Windows 11); the browser runs the same module on the same V8, and
the GUI figures quoted at the end match the Node ones.

## 1. The wall you hit first is time, not memory

A default-budget run is `generations=2800 x n_populations=31`, about 2.83M evaluations,
and **every evaluation is O(rows)**. Full runs on non-convergent data (so early stopping
cannot cut them short), 2 columns:

| rows | full default-budget run |
|------:|------------------------|
| 200 | 18.5 s |
| 2,000 | 153.7 s |
| 10,000 | ~13 min (extrapolated, `t ~= 3.5 + 0.075n` s) |
| 100,000 | ~2.1 h (extrapolated) |

So the browser is comfortable to a few thousand rows and unusable well before it runs out
of memory. This ordering is why the memory ceiling is *not* the primary thing to fix.

## 2. Batching is the lever for row count

PySR's `batching` (default off there too, so enabling it is not a parity divergence — see
`docs/28` B5) evaluates the evolution and constant-optimisation passes on `batch_size`
rows per iteration, while the hall of fame, the early-stop test and the reported result
stay on the full data. Full default-budget runs, non-convergent data:

| rows | batching off | batching on | speedup |
|------:|-------------:|------------:|--------:|
| 10,000 | ~754 s (extrapolated) | 41.2 s (measured, 3.0M evals) | ~18x |
| 100,000 | ~2.1 h (extrapolated) | 341 s (measured, 3.0M evals) | ~22x |

Measured through the GUI on a 20,000-row table at a 112-generation budget: **55.1 s -> 3.2 s
(17.5x)**, with a comparable evaluation count — and on that dataset the batched run
recovered the generating formula while the unbatched one had not yet.

Batching does **not** move the memory ceiling: every row still lives in the heap.

## 3. The memory ceiling, and how it is easy to measure wrongly

Over-allocation is not a recoverable error here. Emscripten's default `ABORTING_MALLOC`
aborts the module (`Aborted(OOM). Build with -sASSERTIONS for more info.`), which surfaces
as a thrown JS error and leaves that module instance dead. So the row limit has to be
*predicted* before the run.

**The pitfall:** the first measurements were taken at `n_populations=1` and suggested a
ceiling around 500,000 rows. At the shipped default of 31 populations the true ceiling is
3-5x lower, because the self-LM optimiser holds O(rows) scratch **per island**:

| n=300,000, 1 column | result |
|---|---|
| 4 populations | OK |
| 8 populations | OK |
| 16 populations | `Aborted(OOM)` |

At the default 31 populations:

| dataset (columns incl. target) | result |
|---|---|
| 120,000 x 1 / 100,000 x 5 / 100,000 x 10 | OK |
| 300,000 x 1 / 200,000 x 5 / 200,000 x 10 | `Aborted(OOM)` |

A model consistent with all ten points:

```
peak bytes ~= n * (24*p + 80 + 16*n_populations)
```

- `24p + 80` — three live copies of the data: `Xflat` and the `vector<vector>` `X` in
  `web/wasm/rsymbolic2_wasm.cpp`, plus the copy `run_evolution()` makes into its shared
  `Dataset` (`evolutionary_search.cpp`). The `+80` is the per-row allocation overhead of
  the row vectors.
- `16 * n_populations` — per-island optimiser scratch. At 31 populations this term
  dominates: for `p=5` it is 496 of 696 bytes per row.

`web/app/js/data.js: maxRowsForBrowser()` inverts this against a deliberately conservative
64 MB budget (the smallest measured OOM point is ~139 MB by the model, the largest measured
success ~82 MB), giving ~112,000 rows at 1 column, ~96,000 at 5, ~82,000 at 10 — all with
the default 31 populations, all recomputed when the user changes the population count.

The bridge now releases `Xflat` as soon as `X` is built, which removes one of the three
copies. At the default 31 populations that is only ~6% of the peak (the island term
dominates); it matters more at low population counts. Behaviour is unchanged either way.

## 4. Bugs that only appeared at scale (fixed)

1. `plots.js` computed the prediction chart's axis range with `Math.min(...y, ...yhat)`.
   Spreading the rows onto the call stack throws `RangeError` past ~100,000 rows (measured:
   50,000 fine, 100,000 throws), and the throw was swallowed by the caller's `try/catch`,
   so the fit chart went **blank with no explanation** for large multi-feature data. It is
   a loop now, and it skips non-finite predictions instead of poisoning the range.
2. `loadFile()` had no `onerror` and no size check: a file too large to read failed
   **silently**, with no message at all.
3. An OOM abort reached the user as the raw Emscripten string, and the dead worker was kept
   alive until the next run.

## 5. What the GUI does now

- **Byte cap before parsing** (64 MB, all three intake paths — file, drop, paste). Parsing
  is synchronous on the main thread, so this has to be checked before the file is read.
- **Row policy on load** (`main.js: applyRowPolicy`): quiet below 5,000 rows; a warning
  above it; and above `maxRowsForBrowser()` the table is **sampled down automatically**,
  because the alternative is a certain abort. The sample is deterministic (seeded
  selection sampling, `data.js: sampleRowIndices`), the checkbox is locked on while the
  full table cannot fit, and the fact is stated in the data summary, the preview dialog and
  the copied Python/R snippets.
- **The limit is rechecked at Run**, since `n_populations` and the tick-box feature count
  both move it after loading.
- **Real progress**: the WASM bridge now reports `total_epochs` alongside each progress
  snapshot (derived from `generations / migration_interval` in the bridge — the core and
  the R/Python bridges are untouched), so the header bar becomes determinate and the status
  line reads `epoch 12/100 · <= 3 min left`. The estimate uses the recent-epoch rate, not
  the whole-run average, because early epochs carry smaller trees and are much cheaper; it
  is an upper bound, since `target_loss`/timeout/`max_evals` can all end a run sooner.
- **Charts draw at most 5,000 points** (deterministic stride subset), and the equation is
  evaluated only on those rows. Nothing reported depends on it: loss comes from the engine
  and R² from `res.sst`.
- **Batching is exposed** in Search settings under "Large data", off by default, and shows
  up in the settings summary line when on so the divergence is never silent.

## 6. Rejected

- **Raising `INITIAL_MEMORY` to 256/512 MB.** The time wall arrives long before the memory
  wall (§1), so this buys little, and it charges every page load — including phones — for
  memory almost no session uses. Predicting the limit and sampling is strictly better.
- **Parsing the CSV in a worker.** Under the 64 MB cap the worst main-thread stall measured
  is ~1 s (0.46 s parse + 0.13 s matrix build for a 32 MB, 500,000-row file). Not worth a
  second worker and a message protocol.
- **Calibrating a pre-run time estimate by test-running the engine.** The live ETA already
  measures the user's own machine; a second mechanism to predict what the first one
  observes is not worth its complexity. The pre-run notice is therefore qualitative.
- **Changing the GUI's `timeout_seconds` default.** The web GUI's exemption from PySR
  default parity (CLAUDE.md) covers presentation choices such as `model_selection`; a
  timeout changes when the search stops, i.e. the search itself. It stays 0, and the
  warning points at the setting instead.
- **Lowering `n_populations` to fit more rows.** It works (§3) but diverges from a PySR
  parity default, so it is documented, not recommended.
