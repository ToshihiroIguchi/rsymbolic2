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

  // 2b. Display-simplification fields (docs/52), covering the feature for the WASM
  // binding: the Pareto front and the top-level result both carry a *_simplified
  // companion alongside the raw (evaluatable) expression.
  assert(typeof r1.expression_simplified === "string" && r1.expression_simplified.length > 0,
    "top-level expression_simplified is a non-empty string");
  const pf = r1.pareto_front;
  assert(Array.isArray(pf.expression_simplified) && Array.isArray(pf.latex_simplified),
    "pareto_front carries expression_simplified/latex_simplified arrays");
  assert(pf.expression_simplified.length === pf.complexity.length,
    "pareto_front.expression_simplified has one entry per front member");
  assert(pf.latex_simplified.length === pf.complexity.length,
    "pareto_front.latex_simplified has one entry per front member");
  assert(pf.expression_simplified.every((s) => typeof s === "string" && s.length > 0),
    "every pareto_front.expression_simplified entry is a non-empty string");
  assert(pf.latex_simplified.every((s) => typeof s === "string" && s.length > 0),
    "every pareto_front.latex_simplified entry is a non-empty string");
  // complexity_simplified is the node count of expression_simplified. display_simplify()
  // adopts its rewrite only when it shrinks the tree, so it is never larger than the raw
  // complexity — the UI relies on that to render "10 -> 7" only where it is meaningful.
  assert(Array.isArray(pf.complexity_simplified)
      && pf.complexity_simplified.length === pf.complexity.length,
    "pareto_front.complexity_simplified has one entry per front member");
  assert(pf.complexity_simplified.every((c, i) => Number.isInteger(c) && c >= 1
      && c <= pf.complexity[i]),
    "every complexity_simplified is a positive integer <= the raw complexity");

  // 2c. Progress callback (docs/53): purely observational — attaching one must not
  // change the result — and it fires at least once on this multi-iteration config
  // (generations=200 with the default migration_interval=28 gives multiple epochs).
  let fireCount = 0;
  const { flat, nrow, ncol } = flatten(X);
  const optsWithProgress = Object.assign({}, OPTIONS, {
    X: flat, y: Float64Array.from(y), nrow, ncol,
    on_progress: () => { fireCount++; },
  });
  const r3 = Module.run(optsWithProgress);
  if (r3 && r3.error) throw new Error("WASM run error (with on_progress): " + r3.error);
  assert(fireCount >= 1, "on_progress fired at least once on a multi-iteration run");
  assert(r3.expression === r1.expression,
    "on_progress does not change the recovered expression (bit-identical, same seed)");
  assert(r3.pareto_front.loss.join(",") === l1,
    "on_progress does not change the Pareto front losses (bit-identical, same seed)");

  // 2d. Opt-in search-time strong simplification (strong_simplify; docs/55): an unset
  // flag must behave identically to an explicit false (default-off parity), and
  // enabling it must complete with a finite loss and populate the
  // strong_simplify_attempts/adopted eval_counts entries.
  const optsExplicitFalse = Object.assign({}, OPTIONS, {
    X: flat, y: Float64Array.from(y), nrow, ncol, strong_simplify: false,
  });
  const r4 = Module.run(optsExplicitFalse);
  if (r4 && r4.error) throw new Error("WASM run error (strong_simplify=false): " + r4.error);
  assert(r4.expression === r1.expression,
    "strong_simplify=false is identical to the flag being unset (same seed)");
  assert(r4.pareto_front.loss.join(",") === l1,
    "strong_simplify=false Pareto losses identical to the flag being unset");
  assert(r4.eval_counts.strong_simplify_attempts === 0 &&
    r4.eval_counts.strong_simplify_adopted === 0,
    "strong_simplify counters are zero when the option is off");

  const optsStrongSimplify = Object.assign({}, OPTIONS, {
    X: flat, y: Float64Array.from(y), nrow, ncol, strong_simplify: true,
    unary_ops: ["neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"],
    binary_ops: ["add", "sub", "mul", "div", "pow"],
  });
  const r5 = Module.run(optsStrongSimplify);
  if (r5 && r5.error) throw new Error("WASM run error (strong_simplify=true): " + r5.error);
  assert(Number.isFinite(r5.loss), "strong_simplify=true run completes with a finite loss");
  assert(r5.eval_counts.strong_simplify_attempts > 0,
    "strong_simplify=true records at least one attempt with a generous operator set");
  assert(r5.eval_counts.strong_simplify_adopted <= r5.eval_counts.strong_simplify_attempts,
    "strong_simplify adopted count never exceeds the attempt count");

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
