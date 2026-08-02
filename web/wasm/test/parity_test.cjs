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
//                    the native libm (UCRT on Windows since docs/68, glibc on Linux) — so
//                    the two builds can converge to different
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

  // 2b-2. SymPy renderings (docs/70). The GUI's "SymPy" button and the Pareto CSV both read
  // these, and the reason they exist is `^`: it is the engine's power operator on every
  // display surface, and Python reads it as xor, so eval()/NumPy/lambdify() compute the
  // wrong function without complaining. The property asserted here is that no such token
  // survives into the export.
  //
  // This run's operator set has no pow, square, inv or neg at all, and the check still
  // matters: the DISPLAY simplifier introduces a Square node on its own (x*x -> square(x)),
  // which the renderer prints as `(x ^ 2)` (docs/71), so the equation on screen can carry a
  // power the user never enabled.
  assert(Array.isArray(pf.sympy) && Array.isArray(pf.sympy_simplified),
    "pareto_front carries sympy/sympy_simplified arrays");
  assert(pf.sympy.length === pf.complexity.length
      && pf.sympy_simplified.length === pf.complexity.length,
    "pareto_front.sympy/sympy_simplified have one entry per front member");
  assert(pf.sympy.every((s) => typeof s === "string" && s.length > 0)
      && pf.sympy_simplified.every((s) => typeof s === "string" && s.length > 0),
    "every sympy/sympy_simplified entry is a non-empty string");
  const notPython = (s) => /\b(square|inv|neg)\s*\(/.test(s) || s.includes("^");
  assert(!pf.sympy.some(notPython) && !pf.sympy_simplified.some(notPython),
    "no sympy rendering spells square()/inv()/neg()/^");
  assert(pf.expression_simplified.some((s) => s.includes(" ^ 2)")),
    "the display simplifier did introduce a Square node (so the check above was exercised)");
  // The renderer never spells an engine-internal operator name, on any surface: those
  // three nodes print as `(a ^ 2)`, `(1 / a)` and `(-a)` (docs/71).
  const engineName = (s) => /\b(square|inv|neg)\s*\(/.test(s);
  assert(!pf.expression.some(engineName) && !pf.expression_simplified.some(engineName),
    "no expression string spells square()/inv()/neg()");

  // 2c. Progress callback (docs/53): purely observational — attaching one must not
  // change the result — and it fires at least once on this multi-iteration config
  // (generations=200 with the default migration_interval=28 gives multiple epochs).
  let fireCount = 0;
  const snapshots = [];
  const { flat, nrow, ncol } = flatten(X);
  const optsWithProgress = Object.assign({}, OPTIONS, {
    X: flat, y: Float64Array.from(y), nrow, ncol,
    on_progress: (s) => { fireCount++; snapshots.push(s); },
  });
  const r3 = Module.run(optsWithProgress);
  if (r3 && r3.error) throw new Error("WASM run error (with on_progress): " + r3.error);
  assert(fireCount >= 1, "on_progress fired at least once on a multi-iteration run");

  // Each snapshot carries the epoch budget (docs/59) so a caller can show real progress
  // instead of an indeterminate bar. It is derived from generations/migration_interval, so
  // it is constant across the run and never smaller than the epoch it accompanies.
  const expectedEpochs = Math.ceil(OPTIONS.generations / 28); // core default migration_interval
  assert(snapshots.every((s) => s.total_epochs === expectedEpochs),
    `every snapshot reports total_epochs === ${expectedEpochs}`);
  assert(snapshots.every((s) => Number.isInteger(s.epoch) && s.epoch >= 1
                                && s.epoch <= s.total_epochs),
    "epoch is a positive integer within total_epochs");
  // Each snapshot also carries ONE expression: the lowest-loss member of the front so far,
  // printed raw (docs/53 phase 2). It is what the GUI shows as a provisional line under the
  // live chart, so it must be a real, non-empty expression string every epoch — and it must
  // be the LAST member of the front the same snapshot reports, which is the invariant the
  // binding relies on when it picks front.back() as the lowest-loss one.
  assert(snapshots.every((s) => typeof s.expression === "string" && s.expression.length > 0),
    "every snapshot carries a non-empty provisional expression");
  assert(snapshots.every((s) => s.complexity.length === s.loss.length && s.complexity.length > 0),
    "snapshot complexity/loss arrays are non-empty and equal length");
  assert(snapshots.every((s) => s.loss[s.loss.length - 1] === Math.min(...s.loss)),
    "the last front member is the lowest-loss one (what expression is taken from)");
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

  // 2e. Opt-in macro operators (docs/57): one-argument templates over the primitives,
  // expanded when a growth mutation creates a unary node. Passed to this bridge as two
  // parallel arrays, exactly like the R/Python bridges take them.
  const runOpts = (extra) =>
    Module.run(Object.assign({}, OPTIONS, { X: flat, y: Float64Array.from(y), nrow, ncol }, extra));

  // Default parity: empty arrays must be indistinguishable from the fields being absent.
  // This is the load-bearing assertion — it proves the macro code is inert when unused.
  const rNoMacro = runOpts({ macro_names: [], macro_bodies: [] });
  if (rNoMacro && rNoMacro.error) throw new Error("WASM run error (empty macros): " + rNoMacro.error);
  assert(rNoMacro.expression === r1.expression,
    "empty macro arrays are identical to the fields being unset (same seed)");
  assert(rNoMacro.pareto_front.loss.join(",") === l1,
    "empty macro arrays leave the Pareto losses identical");

  // Validation: one parser (make_macro_op / parse_expression.hpp) serves every interface, so
  // the browser rejects exactly what R and Python reject, with the same message.
  const badBody = runOpts({ macro_names: ["gauss"], macro_bodies: ["exp(foo(x))"] });
  assert(badBody && typeof badBody.error === "string" && badBody.error.includes("gauss")
      && badBody.error.includes("foo"),
    "an unknown function in a macro body is rejected, naming the macro and the function");
  const twoArgs = runOpts({ macro_names: ["dbl"], macro_bodies: ["x * x"] });
  assert(twoArgs && typeof twoArgs.error === "string" && twoArgs.error.includes("exactly once"),
    "a macro body using the argument twice is rejected");
  const mismatched = runOpts({ macro_names: ["gauss"], macro_bodies: [] });
  assert(mismatched && typeof mismatched.error === "string"
      && mismatched.error.includes("same length"),
    "mismatched macro name/body array lengths are rejected");

  // A working macro: the run completes, is deterministic, and the macro is INVISIBLE in the
  // results — the front prints the expanded primitive form, which is what keeps the reported
  // expression evaluatable by predict() with no macro knowledge (docs/57 §2).
  const macroOpts = { macro_names: ["gauss"], macro_bodies: ["exp(-square(x))"] };
  const rMacro = runOpts(macroOpts);
  if (rMacro && rMacro.error) throw new Error("WASM run error (macro): " + rMacro.error);
  assert(Number.isFinite(rMacro.loss), "a run with a macro operator completes with a finite loss");
  assert(rMacro.pareto_front.complexity.length > 0, "the macro run returns a non-empty front");
  assert(rMacro.pareto_front.expression.every((e) => !e.includes("gauss"))
      && !rMacro.expression.includes("gauss"),
    "the macro name never appears in a returned expression (expanded primitive form)");
  const rMacro2 = runOpts(macroOpts);
  assert(rMacro2.expression === rMacro.expression
      && rMacro2.pareto_front.loss.join(",") === rMacro.pareto_front.loss.join(","),
    "a macro run is deterministic across runs (same seed)");

  // 2f. Every macro preset the web GUI offers must be accepted by the engine. A preset that
  // the parser rejects is a button that produces an error message, and nothing else in the
  // build would notice: the GUI deliberately carries no copy of the grammar (docs/57 §5), so
  // the bodies are only ever validated at Run. The bodies are READ OUT of main.js rather than
  // restated here — a second copy of the list is exactly the drift this is meant to catch.
  // Cheap on purpose: validation happens before the search, so one generation suffices.
  const mainJs = require("fs").readFileSync(
    path.join(__dirname, "..", "..", "app", "js", "main.js"), "utf8");
  const presetBlock = /const MACRO_PRESETS = \[([\s\S]*?)\n\];/.exec(mainJs);
  assert(presetBlock !== null, "MACRO_PRESETS block found in web/app/js/main.js");
  const presets = [...presetBlock[1].matchAll(/name: "([^"]+)",\s*body: "([^"]+)"/g)]
    .map((m) => ({ name: m[1], body: m[2] }));
  // A lower bound, not an exact count: reading source text means a reformatted list could
  // match nothing, and this must FAIL in that case rather than vacuously pass on zero presets.
  assert(presets.length >= 5, `parsed ${presets.length} macro presets out of main.js`);
  const badPresets = presets.filter((p) => {
    const res = runOpts({
      macro_names: [p.name], macro_bodies: [p.body], generations: 1, n_populations: 2,
    });
    if (res && res.error) console.error(`   preset '${p.name} = ${p.body}': ${res.error}`);
    return Boolean(res && res.error);
  });
  assert(badPresets.length === 0,
    `all ${presets.length} shipped macro presets are accepted by the engine`);

  // 2f-bis. Input validation at the bridge boundary, matching the R and Python bridges
  // (docs/74). Every count below is cast to an unsigned type inside the bridge, so a negative
  // one does not fail — it wraps to ~1.8e19. Before these guards `population_size = -1` came
  // back as the bare std::length_error message "vector" and `generations = -1` started a run
  // of 1.8e19 generations, which in a browser tab is indistinguishable from a hang. The
  // assertions check the MESSAGE, not just that something failed: the message naming the
  // argument is the whole point, and "vector" would still be an error.
  for (const name of ["population_size", "generations", "tournament_size", "max_nodes",
                      "max_depth", "n_populations"]) {
    for (const bad of [0, -1]) {
      const res = runOpts({ [name]: bad });
      assert(res && typeof res.error === "string" && res.error.includes(name)
          && res.error.includes("positive"),
        `${name} = ${bad} is rejected by name (got: ${res && res.error})`);
    }
  }
  // NaN/Inf in the data starve a run rather than stopping it: the core gives a non-finite
  // prediction an infinite loss per candidate, so the search runs to completion and reports a
  // loss that looks ordinary. The GUI's intake cannot produce one (a column with a non-finite
  // cell is offered neither as a feature nor as the target), so this guards every other caller.
  const nanX = Float64Array.from(flat);
  nanX[0] = NaN;
  const rNanX = runOpts({ X: nanX });
  assert(rNanX && typeof rNanX.error === "string" && rNanX.error.includes("X must not contain"),
    "NaN in X is rejected");
  const infY = Float64Array.from(y);
  infY[0] = Infinity;
  const rInfY = runOpts({ y: infY });
  assert(rInfY && typeof rInfY.error === "string" && rInfY.error.includes("y must not contain"),
    "Inf in y is rejected");

  // 2g. The browser's own evaluator (web/app/js/predict.js) must agree with the engine on
  // every operator it re-implements. This matters most for `erf`: R borrows pnorm and Python
  // borrows math.erf, but JavaScript has no error function at all, so predict.js carries its
  // own series/continued-fraction implementation (docs/62 §5) — the one place where a
  // prediction could silently disagree with the search that produced the expression.
  // The check is end-to-end: fit data generated FROM erf/sinh/cosh, then recompute the
  // engine's reported loss from the returned expression using predict.js alone.
  const { predict, parseExpression } = await import(
    require("url").pathToFileURL(path.join(__dirname, "..", "..", "app", "js", "predict.js")).href);

  // First, directly: erf at points spanning both branches of the implementation (the series
  // below |x| = 3 and the continued fraction above it), against values printed by C's erf —
  // the same function the engine calls. 1e-14 relative is far tighter than the few-ulp
  // agreement claimed, and loose enough not to depend on the last bit of Math.exp.
  const ERF_REF = [
    [-2.5, -0.999593047982555], [-0.13, -0.14586711483569575],
    [0.0, 0.0], [0.5, 0.5204998778130465], [1.0, 0.8427007929497149],
    [2.9, 0.9999589021219005], [3.0, 0.9999779095030014], [4.0, 0.9999999845827421],
  ];
  for (const [x, want] of ERF_REF) {
    const got = predict(`erf(${x})`, [[0]])[0];
    assert(Math.abs(got - want) <= 1e-14 * Math.max(1, Math.abs(want)),
      `predict.js erf(${x}) = ${got}, expected ${want}`);
  }
  assert(parseExpression("(sinh(x0) + cosh(x0))") !== null,
    "predict.js parses the new unary operators");

  const Xs = [];
  const ys = [];
  for (let k = 0; k < 40; k++) {
    const x = -2 + (4 * k) / 39;
    Xs.push([Math.round(x * 1e6) / 1e6]);
    // A target shaped by all three operators, so the search space is exercised. The values
    // come from predict.js itself only to SHAPE the data; the assertion below compares
    // predict.js against the ENGINE's loss, so this cannot make a broken erf look correct.
    ys.push(Math.round(
      (1.7 * predict(`erf(${x})`, [[0]])[0] + 0.4 * Math.sinh(x) + 0.3 * Math.cosh(x)) * 1e6) / 1e6);
  }
  const fs2 = flatten(Xs);
  const rSpecial = Module.run(Object.assign({}, OPTIONS, {
    X: fs2.flat, y: Float64Array.from(ys), nrow: fs2.nrow, ncol: fs2.ncol,
    unary_ops: ["erf", "sinh", "cosh"], generations: 300,
  }));
  if (rSpecial && rSpecial.error) throw new Error("WASM run error (erf/sinh/cosh): " + rSpecial.error);
  assert(Number.isFinite(rSpecial.loss), "a run over erf/sinh/cosh completes with a finite loss");

  const yhat = predict(rSpecial.expression, Xs);
  let sse = 0;
  for (let i = 0; i < ys.length; i++) sse += (ys[i] - yhat[i]) * (ys[i] - yhat[i]);
  // The engine's loss is the same SSE over the same rows. Exact equality is NOT available:
  // to_string prints constants with "%.6g", so re-evaluating the printed expression fits
  // slightly rounded constants (docs/48 D2) — worth ~1e-5 relative on this loss. A relative
  // 1e-3 is therefore the honest bar, and it is still orders of magnitude tighter than any
  // wrong erf would survive; the exactness of erf itself is the reference check above.
  assert(Math.abs(sse - rSpecial.loss) <= 1e-3 * Math.max(1e-12, Math.abs(rSpecial.loss)),
    `predict.js reproduces the engine loss for an erf/sinh/cosh expression ` +
    `(js ${sse}, engine ${rSpecial.loss}): ${rSpecial.expression}`);

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
