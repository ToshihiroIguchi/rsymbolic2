// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Web Worker that runs the symbolic-regression search off the main thread. run_evolution
// blocks its thread for the whole search, so keeping it here keeps the UI responsive.
// The main thread stops a run by terminating this worker (worker.terminate()); a
// graceful wall-clock budget is available via the `timeout_seconds` option.
//
// Loaded as a module worker: `new Worker(url, { type: "module" })`.

import createRsymbolic2Module from "../vendor/rsymbolic2.js";

let modulePromise = null;
function getModule() {
  if (!modulePromise) modulePromise = createRsymbolic2Module();
  return modulePromise;
}

// The engine runs on a fixed 128 MB WASM heap with Emscripten's default ABORTING_MALLOC,
// so an over-large dataset does not fail gracefully: the module aborts and throws
// "Aborted(OOM). Build with -sASSERTIONS for more info." at us. Allocating the flat input
// on the JS side can fail first, with a RangeError. Neither string means anything to a
// user, and both have the same remedy. main.js caps the row count before it gets here
// (data.js maxRowsForBrowser), so this is the backstop, not the first line of defence.
function humanError(e) {
  const raw = String((e && e.message) || e);
  if (/\bOOM\b|out of memory|allocation failed|Invalid array length|Aborted/i.test(raw)) {
    return "Dataset too large for the in-browser engine (fixed 128 MB heap). " +
           "Use fewer rows, or run the R or Python package.";
  }
  return raw;
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type !== "run") return;
  try {
    // Stage notices (docs/79). Everything between the Run click and the first progress
    // snapshot used to be one undifferentiated sweep: worker spawn, a ~500 KB WASM fetch,
    // its compile and instantiate, the data transfer, then a full first epoch. The engine
    // load is the part that is neither instant nor attributable to the search, and it
    // happens on EVERY run — the worker is recreated per run so that Stop can be a clean
    // terminate(), which also throws away the cached module. Naming it is the honest
    // alternative to hiding it. (`ready`, posted at the bottom of this file, is a weaker
    // claim: the worker script parsed. getModule() is lazy, so at that point the engine
    // has not been fetched.)
    self.postMessage({ type: "stage", stage: "engine" });
    const Module = await getModule();
    // Posted before Module.run() and after the module resolved, so the main thread can time
    // the wall clock the way the ENGINE does: run_evolution's timeout_seconds starts here,
    // not at the Run click, and charging the engine load against it would retire the
    // progress bar before the run it is describing.
    self.postMessage({ type: "stage", stage: "search" });

    // X arrives already flattened row-major (main.js), transferred rather than copied:
    // structured-cloning an array of row arrays cost a full duplicate of the dataset and
    // ~0.3 s of main-thread time at 500k rows (docs/59).
    const opts = Object.assign({}, msg.options, {
      X: msg.X,
      y: msg.y,
      nrow: msg.nrow,
      ncol: msg.ncol,
    });
    if (msg.weights && msg.weights.length) opts.weights = msg.weights;

    // Per-epoch progress observer (docs/53): the WASM build is single-threaded, so this
    // fires synchronously inside Module.run() below, on this same worker thread. Purely
    // additional postMessage traffic — never touches the search itself.
    opts.on_progress = (snap) => {
      self.postMessage({
        type: "progress",
        epoch: snap.epoch,
        total_epochs: snap.total_epochs, // epoch budget, so the UI can show real progress
        complexity: snap.complexity,
        loss: snap.loss,
        expression: snap.expression, // lowest-loss member, raw (undefined on an empty front)
      });
    };

    const t0 = performance.now();
    const result = Module.run(opts); // returns a plain JS object (or {error})
    const elapsed = (performance.now() - t0) / 1000;

    if (result && result.error) {
      self.postMessage({ type: "error", message: result.error });
    } else {
      self.postMessage({ type: "result", result, elapsed });
    }
  } catch (e) {
    self.postMessage({ type: "error", message: humanError(e) });
  }
};

// Let the main thread know the worker script loaded (module fetch succeeded).
self.postMessage({ type: "ready" });
