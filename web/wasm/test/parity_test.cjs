// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Phase-0 correctness gate for the WebAssembly build. Requires the Node variant built by
// web/wasm/CMakeLists.txt (rsymbolic2_node.cjs). Runs three checks:
//   1. Recovery  — the quadratic example y = 2.5 x^2 - 1.3 is recovered to tiny loss.
//   2. Determinism — the same (X, y, seed, options) yields a bit-identical Pareto front
//                    on two runs (the WASM build is single-threaded and seed-deterministic).
//   3. Cross-build equivalence (best-effort) — if the Python package `rsymbolic2` is
//                    importable, the same inputs recover the same target to comparably tiny
//                    loss. NOTE: the returned *expression* is NOT expected to be
//                    bit-identical to the native build. The WASM and Python bridges wrap the
//                    identical C++ core with identical defaults, and each build is
//                    deterministic on its own, but the evolutionary trajectory is sensitive
//                    to last-bit floating-point differences between Emscripten's libm and
//                    the native (MinGW) libm — so the two builds can converge to different
//                    (equally valid) expressions. We therefore assert outcome equivalence
//                    (both recover to loss < 1e-6), not string equality. Skipped with a note
//                    if Python is unavailable.
//
// Run:  node web/wasm/test/parity_test.cjs   (after building the rsymbolic2_node target)

const path = require("path");
const { execFileSync } = require("child_process");

const createModule = require(path.join(__dirname, "rsymbolic2_node.cjs"));

// Quadratic example, identical to web/app/js/examples.js.
function quadratic() {
  const X = [];
  const y = [];
  for (let k = 0; k < 40; k++) {
    const x = -3 + (6 * k) / 39;
    X.push([Math.round(x * 1e6) / 1e6]);
    y.push(Math.round((2.5 * x * x - 1.3) * 1e6) / 1e6);
  }
  return { X, y };
}

function flatten(X) {
  const nrow = X.length;
  const ncol = X[0].length;
  const flat = new Float64Array(nrow * ncol);
  for (let i = 0; i < nrow; i++) for (let j = 0; j < ncol; j++) flat[i * ncol + j] = X[i][j];
  return { flat, nrow, ncol };
}

const OPTIONS = {
  unary_ops: [],
  binary_ops: ["add", "sub", "mul"],
  generations: 200,
  n_populations: 6,
  population_size: 27,
  max_nodes: 30,
  seed: 1,
};

function runWasm(Module, X, y) {
  const { flat, nrow, ncol } = flatten(X);
  const opts = Object.assign({}, OPTIONS, { X: flat, y: Float64Array.from(y), nrow, ncol });
  const res = Module.run(opts);
  if (res && res.error) throw new Error("WASM run error: " + res.error);
  return res;
}

function assert(cond, msg) {
  if (!cond) {
    console.error("FAIL: " + msg);
    process.exitCode = 1;
    throw new Error(msg);
  }
  console.log("ok  : " + msg);
}

(async () => {
  const Module = await createModule();
  const { X, y } = quadratic();

  // 1. Recovery.
  const r1 = runWasm(Module, X, y);
  console.log(`   best expression: ${r1.expression}`);
  console.log(`   best loss: ${r1.loss}`);
  assert(Number.isFinite(r1.loss), "best loss is finite");
  assert(r1.loss < 1e-6, "quadratic recovered to loss < 1e-6");
  assert(r1.pareto_front.complexity.length > 0, "Pareto front is non-empty");

  // 2. Determinism (same seed => identical front).
  const r2 = runWasm(Module, X, y);
  assert(r1.expression === r2.expression, "expression identical across runs (determinism)");
  const l1 = r1.pareto_front.loss.join(",");
  const l2 = r2.pareto_front.loss.join(",");
  assert(l1 === l2, "Pareto losses identical across runs (determinism)");

  // 3. Cross-build equivalence vs Python (best-effort; outcome, not string equality).
  let py = null;
  try {
    const script = `
import json, numpy as np
from rsymbolic2 import symbolic_regression
X = np.array(${JSON.stringify(X)}, dtype=float)
y = np.array(${JSON.stringify(y)}, dtype=float)
res = symbolic_regression(X, y, unary_ops=[], binary_ops=["add","sub","mul"],
    generations=200, n_populations=6, population_size=27, max_nodes=30, seed=1,
    verbosity=0)
print(json.dumps({"expression": res.expression, "loss": res.loss}))
`;
    const out = execFileSync("python", ["-c", script], { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    py = JSON.parse(out.trim().split("\n").pop());
  } catch (e) {
    console.log("skip: Python cross-check (rsymbolic2 not importable: " + String(e.message).split("\n")[0] + ")");
  }
  if (py) {
    console.log(`   Python best expression: ${py.expression}`);
    console.log(`   Python best loss: ${py.loss}`);
    // Outcome equivalence: both builds recover the target to tiny loss. The expressions
    // themselves may differ (cross-toolchain libm ULP divergence) and are shown above for
    // manual inspection, not asserted equal.
    assert(Number.isFinite(py.loss) && py.loss < 1e-6, "Python also recovers to loss < 1e-6");
    assert(r1.loss < 1e-6, "WASM recovers to loss < 1e-6 (same outcome as Python)");
  }

  console.log(process.exitCode ? "\nPARITY TEST FAILED" : "\nPARITY TEST PASSED");
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
