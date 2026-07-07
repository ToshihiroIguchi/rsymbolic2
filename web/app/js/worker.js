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

self.onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.type !== "run") return;
  try {
    const Module = await getModule();

    // Flatten X (array of row arrays) to a row-major Float64Array for the bridge.
    const X = msg.X;
    const nrow = X.length;
    const ncol = nrow > 0 ? X[0].length : 0;
    const flat = new Float64Array(nrow * ncol);
    for (let i = 0; i < nrow; i++) {
      const row = X[i];
      for (let j = 0; j < ncol; j++) flat[i * ncol + j] = row[j];
    }

    const opts = Object.assign({}, msg.options, {
      X: flat,
      y: Float64Array.from(msg.y),
      nrow,
      ncol,
    });
    if (msg.weights && msg.weights.length) opts.weights = Float64Array.from(msg.weights);

    const t0 = performance.now();
    const result = Module.run(opts); // returns a plain JS object (or {error})
    const elapsed = (performance.now() - t0) / 1000;

    if (result && result.error) {
      self.postMessage({ type: "error", message: result.error });
    } else {
      self.postMessage({ type: "result", result, elapsed });
    }
  } catch (e) {
    self.postMessage({ type: "error", message: String((e && e.message) || e) });
  }
};

// Let the main thread know the worker script loaded (module fetch succeeded).
self.postMessage({ type: "ready" });
