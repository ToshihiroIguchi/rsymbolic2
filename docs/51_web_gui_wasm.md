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
  expression,latex}`). It includes only `rsymbolic/...` headers — no R/Python headers.
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
  (`linear_scaling`, `eval_cache`, dimensional analysis) are surfaced but off by default,
  mirroring the library's second layer.
- **The shared C++ core is not modified.** No change to `r-package/rsymbolic2/src/`, so the
  standalone ctest suite, R `testthat` and Python `pytest` remain valid. (A future Phase-3
  live-progress callback would touch the core and must be behaviour-neutral and guarded.)

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

## Deferred (later)

Live per-epoch Pareto/best-loss updates (needs a behaviour-neutral core progress callback);
warm-start "continue"; feature-impact; full candidate-pool query (the engine currently
returns only the Pareto front).
