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
search. For large or long searches, prefer the R or Python package.

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
    js/*.js                 UI (main, worker, predict, data, plots, latex, export)
    vendor/                 KaTeX + Chart.js (vendored, MIT) + built rsymbolic2.{js,wasm}
    examples/ (inline)      example datasets live in js/examples.js
```

## Custom (macro) operators

The sidebar's **Custom operators (macros)** disclosure (under the operator checkboxes) lets a
user define one-argument templates over the primitives — `gauss = exp(-square(x))` — which the
engine expands as it builds expressions (docs/57). A preset list seeds the common ones. The
feature is **off by default**: with no macro rows the two option arrays sent to the bridge are
empty and the search is bit-identical to the PySR-parity run.

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
