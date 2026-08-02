// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// UI wiring and application state for the rsymbolic2 web GUI. Ties together data intake
// (data.js), the WASM search run (worker.js), the result plots (plots.js), equation
// rendering (latex.js) and export (export.js). No framework — plain DOM + ES modules.

import {
  parseTable, numericColumns, toMatrix, maxRowsForBrowser, sampleTable, strideIndices,
  firstNonNumericCell,
} from "./data.js";
import { EXAMPLES } from "./examples.js";
import {
  drawPareto, drawPrediction, drawResidual, destroyPlots, destroyPrediction,
  paretoImage, fitImage, residualImage,
} from "./plots.js";
import { renderInto } from "./latex.js";
import { renderTree, treeSvgStandalone } from "./tree.js";
import { predict } from "./predict.js";
import {
  copyText, pythonCall, rCall, paretoCsv, downloadText,
} from "./export.js";
import { buildReport } from "./report.js";
import { fmt, fmtInt, fmtComplexity } from "./format.js";

// Printed on the report (docs/64) so a saved PDF says which build produced it. Must track
// r-package/rsymbolic2/DESCRIPTION and python/pyproject.toml, which ship the same version:
// the web GUI is a fourth binding over the same core, not a separately versioned product.
// Stamping it at build time was rejected as disproportionate — it would have to work in the
// GitHub Actions build and in web/serve.py, for one string.
const APP_VERSION = "0.1.0";

// Canonical unary order sent to the engine. The core picks operators by a random index into
// this list (mutation.cpp: space.unary_ops[op(rng)]), so the list ORDER is part of the search
// trajectory for a fixed seed. The checkboxes are grouped only for readability (UNARY_GROUPS);
// checkedOps("un") re-emits in this canonical order so the visual grouping never changes the
// search. Keep this array's order stable.
const UNARY = ["neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square", "inv",
               "erf", "sinh", "cosh"];
const UNARY_GROUPS = [
  { label: "Trigonometric", ops: ["sin", "cos", "tanh", "sinh", "cosh"] },
  { label: "Exp / Log", ops: ["exp", "log", "erf"] },
  { label: "Power / other", ops: ["sqrt", "square", "inv", "abs", "neg"] },
];
const UNARY_DEFAULT = new Set(["neg", "exp", "log", "sin", "cos"]);
const BINARY = ["add", "sub", "mul", "div", "pow"];
const BINARY_DEFAULT = new Set(["add", "sub", "mul"]);

// --- Operator display labels ------------------------------------------------------
// Every pill is one token: the operator in ordinary mathematical notation. Nothing on the pill
// repeats itself, so the row stays scannable and a pill is as wide as the notation needs.
//
// The engine identifier ("mul", "sqrt") is NOT shown beside it. It is the machine spelling of
// the same thing, and there is nowhere the user has to type it from memory: an export snippet
// is generated code they copy, results print the names back (`sqrt(x0)`), and the macro
// disclosure spells out the one place a name is written by hand, in its own example
// (`exp(-square(x))`). The identifier lives in each pill's tooltip instead (OP_HINT), which is
// where it is wanted - when reading a snippet, not when picking operators.
//
// square, inv and neg are the exception in both directions: a result never prints their
// names (it prints `(x0 ^ 2)`, `(1 / x0)` and `(-x0)` — tree.hpp to_string, docs/71), while
// a macro body still has to spell them, so their tooltips say both.
//
// sin/cos/tanh/exp keep their word form because that IS the notation for them; sqrt/square/
// inv/abs/neg/log take the symbolic form, which is both shorter and unambiguous (`log` in
// particular is the NATURAL log in the core, and many readers take the word for base 10).
//
// BINARY_GLYPH must stay character-identical to tree.js's BINARY_LABEL, which maps the same
// glyphs from the parsed SYMBOL ("*") instead of the engine id ("mul"): showing one operator
// under two names on one screen is the defect this fixes. Two five-entry tables are cheaper
// than bridging the two key spaces; change them together.
const BINARY_GLYPH = { add: "+", sub: "-", mul: "×", div: "÷", pow: "^" };
const UNARY_GLYPH = {
  log: "ln", sqrt: "√x", square: "x²", inv: "1/x", abs: "|x|", neg: "-x",
};
// Pill tooltip: the engine identifier first (this is the only place it appears now, and the
// form a macro body calls for the unary ops), then the meaning. Guard behaviour is stated only
// where it was read off the core: sqrt and pow reject their out-of-domain arguments with NaN,
// which discards the candidate (safe_sqrt / safe_pow in dual.hpp, SR.jl semantics — docs/69),
// while div/inv/log are unguarded (multi_dual.hpp recip, "Unguarded, like
// operator/"). Absence of a guard clause here means the operator has no guard.
const OP_HINT = {
  add: 'add — a + b', sub: 'sub — a - b', mul: 'mul — a × b',
  div: 'div — a ÷ b; no divide-by-zero guard',
  pow: 'pow — a raised to b; safe_pow, so a negative base with a fractional exponent is NaN and the candidate is rejected',
  neg: 'neg(x) — sign flip; results print it as -x',
  exp: 'exp(x) — e to the power x',
  log: 'log(x) — natural logarithm, base e (not base 10)',
  sin: 'sin(x) — sine, in radians',
  cos: 'cos(x) — cosine, in radians',
  tanh: 'tanh(x) — hyperbolic tangent',
  sqrt: 'sqrt(x) — square root; guarded, a negative argument is NaN and the candidate is rejected',
  square: 'square(x) — x times x; results print it as x ^ 2',
  inv: 'inv(x) — reciprocal; no divide-by-zero guard; results print it as 1 / x',
  abs: 'abs(x) — absolute value',
  erf: 'erf(x) — error function, the integral of a Gaussian: diffusion profiles, Maxwell-Boltzmann fractions, the normal CDF. Bounded between -1 and 1, so no guard is possible or needed',
  sinh: 'sinh(x) — hyperbolic sine: Butler-Volmer current, catenary. No overflow guard; a large argument gives a non-finite value and the candidate is rejected',
  cosh: 'cosh(x) — hyperbolic cosine, same overflow behaviour as sinh',
};

// --- Data-size policy (docs/59) ---------------------------------------------------
// Three measured facts drive every constant here. (1) A default-budget run costs ~2.83M
// evaluations, each O(rows): 200 rows take 18.5 s, 2,000 rows 154 s, and 100,000 rows
// about two hours — single-threaded, with no way for the user to know that in advance.
// (2) The WASM heap is fixed at 128 MB and aborts on over-allocation, so the row ceiling
// must be predicted before the run (data.js maxRowsForBrowser). (3) Reading and parsing
// happen synchronously on the main thread, so an enormous file freezes the UI before any
// of our checks could run — hence the byte cap, tested before the file is read.
const MAX_INPUT_BYTES = 64 * 1024 * 1024;
// Above this a default-budget run is minutes, not seconds. Counted in CELLS (rows x fitted
// columns), not rows: an evaluation walks every fitted value, so 5,000 rows x 20 columns is
// an order of magnitude worse than the 5,000 x 2 the old row-only threshold was calibrated
// on and used to pass without a word. 10,000 keeps that calibration point exactly (the
// docs/59 measurements are all 2-column) while making the wide case say something.
const SLOW_CELL_THRESHOLD = 10000;
const DEFAULT_SAMPLE_ROWS = 5000;  // what auto-sampling falls back to
const SAMPLE_SEED = 1;             // fixed: the same table must always yield the same sample
// Chart.js allocates an object per point and both charts are rebuilt on every equation
// click and theme toggle, so plots draw a fixed-size stride subset of the data.
const DISPLAY_POINT_CAP = 5000;

// Macro operators (docs/57): one-argument expression templates the engine expands as it
// builds a tree. Ready-made bodies, offered because a feature nobody can guess the syntax of
// is a feature nobody uses. Each body is written in the grammar the C++ parser accepts
// (parse_expression.hpp): binary operators in infix, unary operators in call form, and the
// argument `x` exactly once.
//
// What earns a preset a place here is DISTANCE: how many lucky mutations the primitive set
// needs to reach the motif, which is the only thing a macro buys (docs/57 §2). `extra` below
// is the node count the macro adds over its argument, measured with make_macro_op() — a
// 5-or-6-node motif is one a random walk essentially never assembles, while a 2-node one the
// search finds unaided. That is also why an earlier `cube = x^3` entry is gone: a numeric
// literal in a body becomes a TUNABLE constant, so it was never a cube — it was `x^c` seeded
// at 3, two nodes from reach and misnamed. `powlaw` is the same template under its true name.
//
// The groups are the shapes a user is actually looking for, not operator families; each
// preset's `hint` says where the motif comes from and which numbers end up fitted, since a
// name alone cannot warn that `lorentz`'s 1s are free parameters.
const MACRO_PRESETS = [
  { group: "Peaks", name: "gauss", body: "exp(-square(x))", extra: 3,
    hint: "Gaussian bump. The width and centre come from what you feed it: exp(-square((x0 - c1) / c2))." },
  { group: "Peaks", name: "lorentz", body: "1 / (1 + square(x))", extra: 5,
    hint: "Lorentzian / Cauchy peak — a resonance lineshape. Both 1s are fitted, giving amplitude and width." },
  { group: "Peaks", name: "semicircle", body: "sqrt(1 - square(x))", extra: 4,
    hint: "Semicircular dome: circular and spherical cross-sections, the field of a charged disc, Wigner's semicircle. sqrt is guarded, so it reads 0 outside the fitted radius rather than failing." },
  { group: "Peaks", name: "lognormal", body: "exp(-square(log(x)))", extra: 4,
    hint: "Log-normal peak, skewed on a linear axis: particle, droplet and grain size distributions. Needs x > 0." },
  { group: "S-curves", name: "sigmoid", body: "1 / (1 + exp(-x))", extra: 6,
    hint: "Logistic S-curve. Both 1s are fitted, so this is the general logistic, not only the 0-to-1 one." },
  { group: "S-curves", name: "softplus", body: "log(1 + exp(x))", extra: 4,
    hint: "Smooth hinge: ~0 for negative x, ~x for positive x." },
  { group: "S-curves", name: "log1p", body: "log(1 + x)", extra: 3,
    hint: "Diminishing returns that stays finite at x = 0, unlike log(x)." },
  { group: "S-curves", name: "saturate", body: "1 / (1 + 1 / x)", extra: 6,
    hint: "Rise to a plateau: Langmuir adsorption, Michaelis-Menten kinetics, saturation magnetisation. This is x / (1 + x), rewritten because that form would use x twice; both 1s are fitted, giving the plateau and the half-way point." },
  { group: "S-curves", name: "erfc", body: "1 - erf(x)", extra: 3,
    hint: "Complementary error function: the concentration profile of one-dimensional diffusion (Fick's second law) and the tail of a Maxwell-Boltzmann distribution. Needs no erf checkbox — declaring the body is what enables it. For a very large x, 1 - erf(x) cancels to a few digits; the shape is what the search uses it for." },
  { group: "Growth / decay", name: "decay", body: "exp(-1.0 * x)", extra: 3,
    hint: "Exponential decay with a fitted rate, seeded at -1. The rate may fit positive, giving growth." },
  { group: "Growth / decay", name: "arrhenius", body: "exp(-1.0 / x)", extra: 3,
    hint: "Rate law exp(-Ea / kT): activated processes, solubility, viscosity. The barrier is fitted." },
  { group: "Growth / decay", name: "planck", body: "1 / (exp(x) - 1)", extra: 5,
    hint: "Bose-Einstein occupation, the denominator of Planck's law. The fitted -1 also reaches Fermi-Dirac (+1)." },
  { group: "Growth / decay", name: "stretchexp", body: "exp(-x^2.0)", extra: 4,
    hint: "Stretched-exponential (Kohlrausch) relaxation: dielectric, structural and luminescence decay. The exponent is fitted from 2, so the family spans the Gaussian (2) and ordinary decay (1)." },
  { group: "Growth / decay", name: "coth", body: "1 / tanh(x)", extra: 3,
    hint: "Hyperbolic cotangent: the leading term of the Langevin and Brillouin magnetisation curves and of the Debye heat capacity. Full Langevin coth(x) - 1/x is not expressible, as x may appear only once." },
  { group: "Powers / roots", name: "powlaw", body: "x^2.0", extra: 2,
    hint: "Power law with a fitted exponent, seeded at 2 — not a square. The exponent is free to land on 0.5 or -3." },
  { group: "Powers / roots", name: "rsqrt", body: "1 / sqrt(x)", extra: 3,
    hint: "Inverse square root: periods, orbital and wave speeds, standard errors." },
  { group: "Powers / roots", name: "relgamma", body: "1 / sqrt(1 - square(x))", extra: 6,
    hint: "Relativistic Lorentz factor 1/sqrt(1 - (v/c)^2). Six nodes deep, so the search rarely builds it alone." },
];

const $ = (id) => document.getElementById(id);

// The shipped search defaults, PySR-identical (docs/28). Single source for four consumers:
// readConfig()'s fallback when a field is blank, resetDefaults(), the "modified" marker in the
// Settings summary, and the per-field hint printed in the settings dialog — so those can never
// drift apart. Keys are the input element ids; `int` marks the fields parsed with parseInt;
// `note` is the one-line description shown under the field, kept here rather than in the HTML
// so a constant's meaning and its parity value live in the same place. model_selection is
// deliberately absent: it is a results-side display control (score, the web GUI's sanctioned
// divergence — CLAUDE.md "the web GUI is exempt"), not a search setting, and it lives outside
// this panel.
const DEFAULTS = {
  generations: { value: 2800, int: true, note: "Evolution cycles per population." },
  n_populations: { value: 31, int: true, note: "Independent populations evolved in parallel." },
  population_size: { value: 27, int: true, note: "Expressions held in each population." },
  max_nodes: { value: 30, int: true, note: "Largest expression the search may build." },
  seed: { value: 1, int: true, note: "Fixed seed; the same seed reproduces the run exactly." },
  timeout_seconds: { value: 0, note: "Wall-clock limit; 0 runs the full budget." },
  tournament_size: { value: 15, int: true, note: "Candidates entering each selection tournament." },
  tournament_selection_p: { value: 0.982, note: "Chance the tournament picks its best member." },
  parsimony: { value: 0, note: "Fixed complexity penalty added to the loss." },
  adaptive_parsimony_scaling: {
    value: 1040,
    note: "Strength of the frequency-adaptive penalty on crowded complexities.",
  },
  optimize_probability: { value: 0.14, note: "Chance a candidate gets its constants optimised." },
  crossover_probability: { value: 0.0259, note: "Chance a mutation step is a crossover instead." },
  fraction_replaced_hof: { value: 0.0614, note: "Population fraction reseeded from the hall of fame." },
  max_depth: { value: 30, int: true, note: "Deepest expression tree allowed." },
  target_loss: { value: 1e-10, note: "Stop early once the loss reaches this." },
  max_evals: { value: 0, note: "Evaluation-count limit; 0 leaves it uncapped." },
  warmup_maxsize_by: {
    value: 0,
    note: "Fraction of the run spent ramping the size limit up; 0 disables the ramp.",
  },
  batch_size: { value: 50, int: true, note: "Rows drawn per iteration when batching is on." },
};

// The dialog's non-numeric fields, kept beside DEFAULTS for the same reason: the settings
// snapshot (Cancel/Esc restore) and "Reset to PySR defaults" must see every field the dialog
// owns. Batching is off in PySR too, so the default here is still the parity value.
// Deliberately excludes the sidebar's high-accuracy opt-ins (linear_scaling, eval_cache):
// they live outside this dialog, and a button inside it must not rewrite state the user
// cannot see from where they clicked.
const CHECKBOX_DEFAULTS = { batching: false };

const state = {
  sourceTable: null, // { columns, rows } exactly as parsed
  table: null, // the table actually fitted — sourceTable, or a sample of it
  rowLimit: 0, // rows the engine can hold for this shape (data.js maxRowsForBrowser)
  policyShape: "", // the "fitted width|populations" the current rowLimit/sample was derived from
  numeric: [], // bool[] per column
  targetIndex: -1,
  featureIndices: [],
  result: null, // last WASM result
  config: null, // last config used
  X: null,
  y: null,
  featureNames: null,
  targetName: null,
  sampling: null, // {fitted, total, seed} when the run used a sample; snapshotted at run time
  dataSource: null, // {label, kind} of the loaded table — where the data came from
  // Facts about the finished run that nothing on screen keeps: the report needs them, the
  // live UI only ever turned them into status-chip text. Snapshotted at result time for the
  // same reason as featureNames/targetName/sampling above — a displayed result describes the
  // run that produced it, and controls the user can still move afterwards must not rewrite it.
  run: null, // {startedAt, elapsed, stoppedEarly, epoch, totalEpochs, source}
  runStartedAt: null, // Date the in-flight run was launched
  selectedIndex: null,
  treeSvg: null, // the rendered <svg> of the selected equation's tree, for the SVG download
  worker: null,
  settingsSnapshot: null, // field values captured when the settings dialog opened (Cancel restores them)
  timer: null,
  t0: 0,
  lastProgressDraw: 0, // performance.now() of the last live-Pareto redraw (throttling)
  epoch: 0,            // epochs completed, from the engine's progress snapshots
  totalEpochs: 0,      // epoch budget the engine reports (an upper bound: it can stop early)
  epochMarks: [],      // performance.now() at each completed epoch, for the ETA's recent rate
};

// --- Theme ------------------------------------------------------------------------
// The <head> inline script already resolved data-theme before first paint; here we only
// handle the toggle. Charts read their colors from CSS vars at draw time (plots.js
// themeColors), so a toggle just redraws them.
function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* private mode etc. */ }
  updateThemeToggleIcon();
  redrawCharts();
}
function updateThemeToggleIcon() {
  const dark = document.documentElement.dataset.theme === "dark";
  $("theme-toggle").textContent = dark ? "☀" : "🌙";
}
function redrawCharts() {
  if (!state.result) return;
  drawParetoChart();
  if (state.selectedIndex != null) selectEquation(state.selectedIndex);
}

// --- Setup static controls --------------------------------------------------------
function buildOperatorChecks() {
  const bwrap = $("binary-ops");
  BINARY.forEach((op) =>
    bwrap.appendChild(opCheck("bin", op, BINARY_DEFAULT.has(op), BINARY_GLYPH[op] || op)));
  const uwrap = $("unary-ops");
  UNARY_GROUPS.forEach((g) => {
    const group = document.createElement("div");
    group.className = "op-group";
    const label = document.createElement("span");
    label.className = "op-group-label";
    label.textContent = g.label;
    group.appendChild(label);
    const grid = document.createElement("div");
    grid.className = "checkgrid";
    g.ops.forEach((op) =>
      grid.appendChild(opCheck("un", op, UNARY_DEFAULT.has(op), UNARY_GLYPH[op] || op)));
    group.appendChild(grid);
    uwrap.appendChild(group);
  });
}
// One operator pill: the notation as its only visible token, the engine id and the meaning on
// hover. Only the checkbox's `value` reaches the engine, so all of this is presentation.
function opCheck(kind, op, checked, label) {
  const l = document.createElement("label");
  l.className = "check";
  if (OP_HINT[op]) l.title = OP_HINT[op];
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = checked;
  cb.dataset.kind = kind;
  cb.value = op;
  l.appendChild(cb);
  const text = document.createElement("span");
  text.className = "op-primary";
  text.textContent = label;
  l.appendChild(text);
  return l;
}
function checkedOps(kind) {
  const picked = new Set(
    [...document.querySelectorAll(`input[data-kind="${kind}"]:checked`)].map((e) => e.value),
  );
  // Emit unary ops in the canonical UNARY order (not DOM order) so the grouped layout does not
  // alter the operator-index -> RNG mapping and the search stays reproducible. Binary ops are
  // not regrouped, so DOM order already matches their declared order.
  if (kind === "un") return UNARY.filter((op) => picked.has(op));
  return [...picked];
}

// --- Macro operators --------------------------------------------------------------
// A row is one macro: its name and its body. Rows where BOTH fields are blank are ignored,
// so a leftover empty row can never switch the feature on — with no macros the two option
// arrays are empty and the search stays bit-identical to the PySR-parity run (docs/57 §4).
// `hint` is set only for a preset row: an <option>'s tooltip is gone the moment it is picked,
// and what the motif means is exactly what the user wants to re-read while editing the row.
//
// The body tooltip spells out the two notations, because the operator pills show `x²` and `^`
// side by side and a reader cannot tell from them which one a body may be written in: both,
// and they mean different things. `x^2` is pow with a FITTED exponent seeded at 2 (a numeric
// literal is always tunable, docs/57 §2); `square(x)` is the fixed one. That distinction is
// the same one the paragraph above the list makes, said where a body is actually typed.
const MACRO_BODY_HINT =
  "The template, in terms of x — e.g. exp(-square(x)). Binary operators are infix (x^2, 1/x), " +
  "unary ones are calls (square(x), log(x)); a body may use operators left unchecked above. " +
  "Use x exactly once. Numbers become tunable constants seeded at that value, so x^2 is a " +
  "fitted power — square(x) is the fixed square.";

function addMacroRow(name = "", body = "", hint = "") {
  const row = document.createElement("div");
  row.className = "macro-row";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "name";
  nameInput.title = "What to call this operator. It joins the unary list for the search, but " +
                    "results always print the expanded primitive form.";
  nameInput.value = name;
  nameInput.dataset.macro = "name";
  const bodyInput = document.createElement("input");
  bodyInput.type = "text";
  bodyInput.title = MACRO_BODY_HINT + (hint ? `\n\n${hint}` : "");
  bodyInput.placeholder = "exp(-square(x))";
  bodyInput.value = body;
  bodyInput.dataset.macro = "body";
  const del = document.createElement("button");
  del.type = "button";
  del.className = "btn small";
  del.textContent = "×";
  del.title = "Remove this macro";
  del.addEventListener("click", () => {
    row.remove();
    updateMacroSummary();
    saveSearchSettingsSoon(); // a click, so the input/change funnel never sees this one
  });
  [nameInput, bodyInput].forEach((el) => el.addEventListener("input", updateMacroSummary));
  row.append(nameInput, bodyInput, del);
  $("macro-list").appendChild(row);
  updateMacroSummary();
}

function readMacros() {
  const names = [];
  const bodies = [];
  document.querySelectorAll("#macro-list .macro-row").forEach((row) => {
    const name = row.querySelector('[data-macro="name"]').value.trim();
    const body = row.querySelector('[data-macro="body"]').value.trim();
    if (!name && !body) return; // untouched row
    names.push(name);
    bodies.push(body);
  });
  return { names, bodies };
}

// The checks that need no expression parser, so they can be reported before a run starts.
// Everything about the BODY is left to the engine's make_macro_op() (docs/57 §5): one parser
// serves R, Python and the browser, and duplicating its grammar here would let the two drift.
function macroError(names, bodies) {
  const seen = new Set();
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    if (!name) return `Macro ${i + 1}: give the macro a name (or clear its body).`;
    if (!bodies[i]) return `Macro '${name}': the body is empty, e.g. exp(-square(x)).`;
    if (!/^[A-Za-z_]\w*$/.test(name))
      return `Macro '${name}': the name must be a letter or _ followed by letters/digits.`;
    if (UNARY.includes(name) || BINARY.includes(name))
      return `Macro '${name}' shadows a built-in operator; pick another name.`;
    if (seen.has(name)) return `Macro '${name}' is defined twice.`;
    seen.add(name);
  }
  return null;
}

function updateMacroSummary() {
  const n = readMacros().names.length;
  $("macro-summary").textContent = n ? `— ${n} defined` : "";
}

// One <optgroup> per shape, so sixteen entries stay scannable; the option's tooltip carries the
// hint plus the node cost, which is the number that decides whether a macro is worth adding at
// all (a 2-node motif the search reaches unaided).
function buildMacroPresets() {
  const sel = $("macro-preset");
  let group = null;
  MACRO_PRESETS.forEach((m, i) => {
    if (!group || group.label !== m.group) {
      group = document.createElement("optgroup");
      group.label = m.group;
      sel.appendChild(group);
    }
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `${m.name} = ${m.body}`;
    o.title = `${m.hint} Adds ${m.extra} nodes to the expression.`;
    group.appendChild(o);
  });
  sel.addEventListener("change", () => {
    const m = MACRO_PRESETS[parseInt(sel.value, 10)];
    if (m) addMacroRow(m.name, m.body, m.hint);
    sel.value = ""; // back to the placeholder, so the same preset can be picked again
  });
}

function buildExamples() {
  const sel = $("example-select");
  EXAMPLES.forEach((ex, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = ex.label;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => {
    const ex = EXAMPLES[parseInt(sel.value, 10)];
    if (!ex) return;
    applyExampleOps(ex.ops);
    syncOperatorReset();      // ... and no change event means no delegated listener either
    saveSearchSettingsSoon(); // ticked programmatically, so no change event fires on the boxes
    loadTable(ex.make(), { label: ex.label, kind: "example" });
    // Back to the placeholder, same as #macro-preset. This is a loader, not a statement of
    // what is loaded — the data summary line is that. Leaving the picked entry selected broke
    // it in both directions: a <select> fires no change event when the option already selected
    // is picked again, so the same example could never be re-loaded after other data replaced
    // it; and until then the rail sat there naming an example that was no longer the data.
    sel.value = "";
  });
}

// Check the operator boxes an example's target formula needs (e.g. div for a rational),
// so a loaded example is always searchable without hunting through the Operators panel.
// Never unchecks anything the user enabled. Operators are a problem input, not a PySR
// parity default, so this does not touch any search setting.
function applyExampleOps(ops) {
  if (!ops) return;
  (ops.binary || []).forEach((op) => setOpChecked("bin", op));
  (ops.unary || []).forEach((op) => setOpChecked("un", op));
}
function setOpChecked(kind, op) {
  const cb = document.querySelector(`input[data-kind="${kind}"][value="${op}"]`);
  if (cb) cb.checked = true;
}
// Restoring a saved selection is the one caller that must also be able to UNCHECK: setOpChecked
// only ever turns a box on (an example may add operators, never drop one the user picked), which
// cannot express "this default was deliberately switched off".
function applyOpSelection(kind, ops) {
  const want = new Set(ops);
  document.querySelectorAll(`input[data-kind="${kind}"]`).forEach((cb) => {
    cb.checked = want.has(cb.value);
  });
}

// --- Data intake ------------------------------------------------------------------
// `source` = {label, kind} identifying where the table came from ("file", "example",
// "paste"). It exists for the report, which has to name the data it describes; the file name
// used to be read only to compose an error message and then dropped.
function loadTable(table, source = null) {
  // New data invalidates everything downstream of it, including a search still in flight:
  // its result would render against the table it can no longer be read as describing.
  if (document.body.classList.contains("running")) {
    if (state.worker) { state.worker.terminate(); state.worker = null; }
    finishRun();
  }
  clearResults();
  state.sourceTable = table;
  state.dataSource = source;
  state.table = table; // provisional; applyRowPolicy() below may replace it with a sample
  // Classify columns on the FULL table: a column holding text in a row that sampling happens
  // to drop is still not a numeric column, and this way the target/feature lists do not
  // reshuffle when the sample size changes.
  state.numeric = numericColumns(state.sourceTable);
  // The feature list must exist before the row policy runs: the ceiling is computed from the
  // fitted width, which is read off the ticked feature boxes (fittedWidth()). Both work on the
  // same column set, so the provisional table above is enough to build the lists from.
  buildTargetAndFeatures();
  applyRowPolicy(true);
  renderIntakeNotice();
  // Progressive disclosure: the target/feature controls stay hidden until there is data to
  // model, and loading data collapses the intake block (drop zone + example picker) — the
  // "change" button re-opens it.
  document.body.classList.add("has-data");
  document.body.classList.remove("editing-data");
  $("run-btn").disabled = false;
  $("data-summary").disabled = false;
  setStatus("data loaded — press Run");
}

// Return the results panel to its pre-run placeholder state. A displayed result is an answer
// ABOUT one dataset: the front, the fitted matrices (state.X/y) and the column names were all
// snapshotted at run time, so once the table underneath them changes, leaving them on screen
// invites reading the previous dataset's equation as the new one's answer. There is nothing to
// preserve either — the next run overwrites them anyway. Charts are destroyed rather than left
// stale because the hidden canvases keep their last drawing until something redraws them.
function clearResults() {
  if (!state.result) return;
  destroyPlots();
  state.result = null;
  state.config = null;
  state.X = null;
  state.y = null;
  state.featureNames = null;
  state.targetName = null;
  state.sampling = null;
  state.run = null;
  state.selectedIndex = null;
  state.treeSvg = null;
  $("print-btn").disabled = true; // nothing left to report on
  $("results-area").classList.remove("has-result", "has-live");
  $("pareto-card").classList.remove("live");
  $("pareto-table").querySelector("tbody").innerHTML = "";
  $("eq-latex").innerHTML = "";
  $("eq-string").textContent = "";
  $("eq-string").title = "";
  $("eq-metrics").innerHTML = "";
  $("eval-accounting").textContent = "";
  $("fit-note").textContent = "";
  $("tree-box").textContent = "";
  $("tree-nodes").textContent = "";
}

// Shared by all three intake paths (file input, drag & drop, clipboard paste). Parsing is
// synchronous on the main thread, so the byte/character cap has to be applied to the raw
// input BEFORE parsing — by the time we could count rows the UI has already frozen (and a
// file past the browser's maximum string length fails inside FileReader, where the only
// signal is the error event).
function tooLarge(bytes) {
  if (bytes <= MAX_INPUT_BYTES) return null;
  return `That input is ${(bytes / 1048576).toFixed(0)} MB; the browser build accepts up to ` +
         `${MAX_INPUT_BYTES / 1048576} MB. Use fewer rows, or the R or Python package.`;
}
function loadTextAsTable(text, source = null) {
  const problem = tooLarge(text.length);
  if (problem) { setStatus(problem); return; }
  try { loadTable(parseTable(text), source); }
  catch (err) { setStatus("parse error: " + err.message); }
}
function loadFile(file) {
  const problem = tooLarge(file.size);
  if (problem) { setStatus(problem); return; }
  const reader = new FileReader();
  reader.onload = () => loadTextAsTable(reader.result, { label: file.name, kind: "file" });
  reader.onerror = () => setStatus(
    `could not read ${file.name}: ${(reader.error && reader.error.message) || "read failed"}`);
  reader.readAsText(file);
}

// Decide what is actually fitted: the whole table, or a sample of it. Two independent
// reasons to sample — the user asked, or the table cannot fit in the engine's fixed heap
// at all (data.js maxRowsForBrowser). The second is not a preference, so the checkbox is
// forced on and locked; only the sample size stays adjustable. The parsed source table is
// kept so changing the size re-samples from the full data instead of a sample of a sample.
// `fresh` = called for newly loaded data (initialise the controls); otherwise the user
// changed a control and only the outcome is recomputed.
function applyRowPolicy(fresh) {
  const src = state.sourceTable;
  const n = src.rows.length;
  // The ceiling moves with the island count (per-island O(rows) optimiser scratch), so read
  // the live setting rather than the shipped default: raising Populations lowers it. The width
  // is the FITTED width, not the file's, for the same reason — see fittedWidth().
  state.rowLimit = maxRowsForBrowser(fittedWidth(), configuredPopulations());
  const over = n > state.rowLimit;

  // The pair is exclusive, so only "sampling is on" has to be tracked; `over` decides
  // whether the OTHER option is available. Disabling "All rows" — the thing that cannot be
  // done — rather than the sampling toggle is the whole point of the radio pair: a disabled
  // checkbox stuck in the ticked state said "this control is broken" where this says "that
  // option is out", and it says which one in its own label.
  const box = $("sample-rows");
  const allBox = $("sample-all");
  const sizeInput = $("sample-size");
  sizeInput.max = String(state.rowLimit);
  if (fresh) {
    box.checked = over;
    sizeInput.value = String(Math.min(DEFAULT_SAMPLE_ROWS, state.rowLimit));
  }
  if (over) box.checked = true;
  allBox.checked = !box.checked;
  allBox.disabled = over;
  allBox.title = over
    ? `${fmtInt(n)} rows do not fit the engine's heap for this shape, so the whole table ` +
      `cannot be fitted here.`
    : "Fit every row in the file.";
  $("sample-all-label").textContent = `All ${fmtInt(n)} rows`;
  sizeInput.disabled = !box.checked;
  // Decided after the radios settle, because their final state is part of the condition:
  // visible whenever the row count is worth acting on, and ALWAYS while sampling is in effect
  // or forced. The notice tells the user to adjust the count, and a table being fitted on a
  // sample must keep the control that undoes it. (A small ceiling — many columns, or a raised
  // population count — can force sampling below the warning threshold, which used to hide the
  // control the notice pointed at; releasing the lock later must not hide it either.)
  $("sample-control").hidden = !over && !box.checked && !isSlowShape(n);

  let k = parseInt(sizeInput.value, 10);
  if (!Number.isFinite(k) || k < 1) k = Math.min(DEFAULT_SAMPLE_ROWS, state.rowLimit);
  k = Math.min(k, state.rowLimit);
  sizeInput.value = String(k);

  state.table = box.checked ? sampleTable(src, k, SAMPLE_SEED) : src;
  state.policyShape = policyShape();
  renderDataSummary();
  renderDataNotice(over);
}

// The width the engine actually holds: the ticked feature columns plus the target, NOT the
// file's column count. The difference is not cosmetic — the heap term is 24 bytes per value
// per row, so modelling 2 columns of a 50-column file triples the row ceiling. Judging by the
// file's width used to force-sample (and lock the checkbox on) tables that fit comfortably.
// Falls back to the file's width before the feature list exists.
function fittedWidth() {
  const picked = document.querySelectorAll('input[data-feature="1"]:checked').length;
  if (picked > 0) return picked + 1;
  return state.sourceTable ? state.sourceTable.columns.length : 1;
}

// Everything the row ceiling depends on, as one comparable value.
function policyShape() {
  return `${fittedWidth()}|${configuredPopulations()}`;
}

// Which rows are fitted, as one comparable value: the sample size when sampling is on, -1 for
// the whole table. Equal before and after a policy pass means the fitted row set did not move,
// so a displayed result and a search in flight are still about the same data.
function fittedSignature() {
  if (!$("sample-rows").checked) return -1;
  const k = parseInt($("sample-size").value, 10);
  return Number.isFinite(k) ? k : -1;
}

// The ceiling moves with Populations and with the feature selection, and both can change
// through paths that fire no `change` event on them — the settings dialog restores every field
// by assignment on Cancel/Esc, and the feature list is rebuilt wholesale when the target
// changes. Comparing against the shape the current policy was computed from covers all of them.
function syncRowPolicyIfShapeChanged() {
  if (state.sourceTable && policyShape() !== state.policyShape) reapplyRowPolicy();
}

// Re-derive the fitted table after a control that can move it changed. Only a change to the
// ROWS invalidates what is on screen: re-sampling makes a displayed result an answer about
// data that is no longer loaded, and makes a search in flight one whose result could not be
// rendered honestly. Merely recomputing the ceiling (ticking a feature, retyping Populations)
// leaves the fitted rows alone and must not throw away a result or kill a running search.
function reapplyRowPolicy() {
  const before = fittedSignature();
  applyRowPolicy(false);
  if (fittedSignature() === before) return;
  if (document.body.classList.contains("running")) {
    if (state.worker) { state.worker.terminate(); state.worker = null; }
    finishRun();
    setStatus("stopped — the data changed");
  }
  clearResults();
}

function syncBatchingDependants() {
  const on = $("batching").checked;
  $("eval_cache").disabled = on;
  $("eval-cache-note").hidden = !on;
}

function configuredPopulations() {
  const v = parseInt($("n_populations").value, 10);
  return Number.isFinite(v) && v > 0 ? v : DEFAULTS.n_populations.value;
}

// Re-draw the data notice without touching what is fitted. Needed because the notice now
// carries an "Enable batching" action, and batching can also be switched from the settings
// dialog — leaving a notice that still offers to do what has just been done.
function refreshDataNotice() {
  if (!state.sourceTable) return;
  renderDataNotice(state.sourceTable.rows.length > state.rowLimit);
}

// Is the displayed result fitted on fewer rows than the file holds?
function isSampled() {
  return !!state.sourceTable && state.table.rows.length < state.sourceTable.rows.length;
}

// Is a default-budget run on this table going to take minutes? Cell count, not row count —
// every evaluation walks every fitted value, so the width belongs in the test.
function isSlowShape(rows) {
  return rows * fittedWidth() > SLOW_CELL_THRESHOLD;
}

function renderDataNotice(over) {
  const el = $("data-notice");
  const n = state.sourceTable.rows.length;
  const fitted = state.table.rows.length;
  el.classList.remove("warn", "acted");
  el.textContent = "";
  let text = null;
  if (over) {
    el.classList.add("acted");
    text =
      `${fmtInt(n)} rows exceed what the in-browser engine can hold — about ` +
      `${fmtInt(state.rowLimit)} rows for the ${fittedWidth()} columns being fitted at ` +
      `${configuredPopulations()} populations (fixed 128 MB heap). Fitting a ` +
      `${fmtInt(fitted)}-row sample instead; adjust the count below, or use the R or ` +
      `Python package for the full table.`;
  } else if (isSlowShape(n)) {
    el.classList.add("warn");
    text =
      `${fmtInt(n)} rows x ${fittedWidth()} columns is a lot for the browser: the engine is ` +
      `single-threaded here and a default-budget run evaluates every row ~2.8M times ` +
      `(10,000 rows x 2 columns takes on the order of ten minutes). Sample the rows below, ` +
      `set a Timeout, or run the R or Python package.`;
  }
  if (text === null) {
    el.hidden = true;
    return;
  }
  el.appendChild(document.createTextNode(text));
  // The notice used to end by telling the user to "enable Batching in Search settings" —
  // an instruction three navigation steps from the sentence making it, in a modal section
  // they have no reason to have opened. It is one click here instead. Absent once batching
  // is on, so the notice never asks for something already done.
  if (!$("batching").checked) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn tiny notice-action";
    btn.textContent = "Enable batching";
    btn.title =
      "Score candidates on a fresh random batch of rows each iteration instead of the whole " +
      "table (~18x faster at 10,000 rows). The reported result is still computed on all rows. " +
      "Same as the Batching option in Search settings.";
    btn.addEventListener("click", () => {
      $("batching").checked = true;
      syncBatchingDependants();
      updateSettingsSummary();
      saveSearchSettings();
      renderDataNotice(over); // drops the button, since the thing it offered is done
      setStatus("Batching enabled — it takes effect on the next run.");
    });
    el.appendChild(btn);
  }
  el.hidden = false;
}

// What the intake could not use, said once when the table is loaded. Two independent things
// silently disappeared before this existed:
//   * a column holding ONE blank, "NA" or text cell is not numeric, so it is offered neither as
//     the target nor as a feature — while the summary line still counts it among the columns,
//     which is what made it invisible. Naming the first offending cell is the point: "x1 is not
//     numeric" sends the user looking through the whole column, "row 2 is NA" does not.
//   * a row whose field count does not match the header is dropped by the parser.
// Neither is worth refusing the file over, so this reports and moves on. Rebuilt only on load:
// both facts are properties of the parsed file, and nothing the user does afterwards changes
// them (the target/feature controls cannot reach a column that is not on them).
const NAMED_DROPPED_COLUMNS = 3; // beyond this, count them instead of listing them
function renderIntakeNotice() {
  const el = $("intake-notice");
  const src = state.sourceTable;
  const dropped = src.columns
    .map((name, j) => ({ name, j }))
    .filter(({ j }) => !state.numeric[j]);
  const lines = [];
  if (dropped.length) {
    const shown = dropped.slice(0, NAMED_DROPPED_COLUMNS).map(({ name, j }) => {
      const bad = firstNonNumericCell(src, j);
      if (!bad) return `"${name}"`; // unreachable while `numeric` is derived from the same test
      const cell = bad.value === "" || bad.value == null ? "blank" : `"${bad.value}"`;
      return `"${name}" (row ${fmtInt(bad.row)} is ${cell})`;
    });
    const rest = dropped.length - shown.length;
    lines.push(
      `${dropped.length} of ${src.columns.length} column${src.columns.length === 1 ? "" : "s"} ` +
      `${dropped.length === 1 ? "is" : "are"} not numeric and cannot be modelled: ` +
      `${shown.join(", ")}${rest > 0 ? `, and ${fmtInt(rest)} more` : ""}. ` +
      "A column needs every cell to be a finite number; fix or remove the cell to use it.");
  }
  if (src.skippedRows > 0) {
    lines.push(
      `${fmtInt(src.skippedRows)} row${src.skippedRows === 1 ? "" : "s"} did not have ` +
      `${src.columns.length} fields and ${src.skippedRows === 1 ? "was" : "were"} skipped.`);
  }
  el.textContent = lines.join(" ");
  el.classList.toggle("warn", lines.length > 0);
  el.hidden = lines.length === 0;
}

function renderDataSummary() {
  const cols = state.table.columns.length;
  const fitted = state.table.rows.length;
  const total = state.sourceTable.rows.length;
  $("data-summary").textContent = fitted === total
    ? `${fmtInt(fitted)} rows × ${cols} columns`
    : `${fmtInt(fitted)} of ${fmtInt(total)} rows × ${cols} columns (sampled)`;
}

// Full-data view in a modal <dialog>. Rendering is capped so a huge CSV cannot lock up the
// UI building millions of DOM cells; the note says when rows are omitted.
const PREVIEW_ROW_CAP = 2000;
function openPreviewDialog() {
  if (!state.table) return;
  const { columns, rows } = state.table;
  const cap = Math.min(rows.length, PREVIEW_ROW_CAP);
  const table = $("preview-full-table");
  table.querySelector("thead").innerHTML =
    "<tr>" + columns.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr>";
  let html = "";
  for (let i = 0; i < cap; i++) {
    html += "<tr>" + rows[i].map((v) => `<td>${esc(v)}</td>`).join("") + "</tr>";
  }
  table.querySelector("tbody").innerHTML = html;
  // The preview shows the table that is FITTED, so when sampling is on it must say so —
  // otherwise the rows on screen read as the file's first rows.
  const shown = rows.length > cap ? `Showing first ${fmtInt(cap)} of ${fmtInt(rows.length)} rows.`
                                  : `${fmtInt(rows.length)} rows.`;
  $("preview-note").textContent = isSampled()
    ? `${shown} These are the sampled rows the search is fitted on ` +
      `(${fmtInt(state.sourceTable.rows.length)} in the file).`
    : shown;
  $("preview-dialog").showModal();
}

function buildTargetAndFeatures() {
  const { columns } = state.table;
  const tsel = $("target-select");
  tsel.innerHTML = "";
  columns.forEach((c, j) => {
    if (!state.numeric[j]) return;
    const o = document.createElement("option");
    o.value = String(j);
    o.textContent = c;
    tsel.appendChild(o);
  });
  // Default target = last numeric column.
  const numericIdx = columns.map((_, j) => j).filter((j) => state.numeric[j]);
  state.targetIndex = numericIdx.length ? numericIdx[numericIdx.length - 1] : -1;
  tsel.value = String(state.targetIndex);
  renderFeatureList();
}

function renderFeatureList() {
  const { columns } = state.table;
  const wrap = $("feature-list");
  wrap.innerHTML = "";
  columns.forEach((c, j) => {
    if (!state.numeric[j]) return;
    if (j === state.targetIndex) return;
    const l = document.createElement("label");
    l.className = "check";
    l.title = `Use column "${c}" as an input variable. Unchecked columns are ignored entirely.`;
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.value = String(j);
    cb.dataset.feature = "1";
    l.appendChild(cb);
    l.appendChild(document.createTextNode(c));
    wrap.appendChild(l);
  });
}

function currentFeatureIndices() {
  return [...document.querySelectorAll('input[data-feature="1"]:checked')].map((e) => parseInt(e.value, 10));
}

// all / none for the feature pills. A wide table otherwise costs one click per column, and
// the list is rebuilt fully ticked on every target change, so "none then pick two" is the
// common shape. Ticking changes the fitted width and therefore the row ceiling, so this ends
// in the same place the individual pills do.
function setAllFeatures(on) {
  document.querySelectorAll('input[data-feature="1"]').forEach((cb) => { cb.checked = on; });
  syncRowPolicyIfShapeChanged();
  saveSearchSettingsSoon();
}

// The Pareto chart's loss axis. A <select> since it joined the two selects beside it, so the
// truth is a string, not a checkbox's .checked.
function logLossEnabled() {
  return $("logloss").value === "log";
}

// --- Config -----------------------------------------------------------------------
// Read each numeric input, falling back to its DEFAULTS entry if the field is empty or
// non-numeric so a cleared box never sends NaN to the engine (an out-of-range max_nodes/seed
// would degrade the search — e.g. max_nodes=NaN collapses every candidate to a single
// constant).
function readConfig() {
  const field = (id) => {
    const d = DEFAULTS[id];
    const v = d.int ? parseInt($(id).value, 10) : parseFloat($(id).value);
    return Number.isFinite(v) ? v : d.value;
  };
  const macros = readMacros();
  return {
    unary_ops: checkedOps("un"),
    binary_ops: checkedOps("bin"),
    // Two parallel arrays, matching the R/Python bridge signatures (the WASM bridge cannot
    // enumerate the keys of a JS object). Empty = the feature is off.
    macro_names: macros.names,
    macro_bodies: macros.bodies,
    generations: field("generations"),
    n_populations: field("n_populations"),
    population_size: field("population_size"),
    max_nodes: field("max_nodes"),
    max_depth: field("max_depth"),
    seed: field("seed"),
    timeout_seconds: field("timeout_seconds"),
    tournament_size: field("tournament_size"),
    tournament_selection_p: field("tournament_selection_p"),
    parsimony: field("parsimony"),
    adaptive_parsimony_scaling: field("adaptive_parsimony_scaling"),
    optimize_probability: field("optimize_probability"),
    crossover_probability: field("crossover_probability"),
    fraction_replaced_hof: field("fraction_replaced_hof"),
    target_loss: field("target_loss"),
    max_evals: field("max_evals"),
    warmup_maxsize_by: field("warmup_maxsize_by"),
    batch_size: field("batch_size"),
    model_selection: $("model_selection").value,
    batching: $("batching").checked,
    linear_scaling: $("linear_scaling").checked,
    eval_cache: $("eval_cache").checked,
  };
}

// The numeric fields carry their own lower bounds in the markup (index.html `min`), and those
// bounds are load-bearing rather than decorative: `min` on a number input constrains the
// spinner, not what can be typed, and every count below is cast to an unsigned type in the
// engine — so a negative one does not fail, it wraps. `population_size = -1` used to surface as
// the bare C++ message "error: vector", and `generations = -1` started a run of 1.8e19
// generations that nothing short of closing the tab ends. The engine now refuses both at its own
// boundary (rsymbolic2_wasm.cpp, matching the R and Python bridges); this check is what turns
// the refusal into a sentence naming the field, before a run is launched at all.
//
// Driven off the markup rather than a second table of bounds, so a field added later is covered
// by the `min` its input already needs. Fields with no `min` (the parity constants, which are
// meaningful at any magnitude) are skipped.
function settingError(cfg) {
  for (const id of Object.keys(DEFAULTS)) {
    const el = $(id);
    if (!el.min) continue;
    const lo = Number(el.min);
    if (!Number.isFinite(lo) || cfg[id] >= lo) continue;
    return `${fieldLabel(id)} must be at least ${lo} (it is ${cfg[id]}).`;
  }
  return null;
}

// The field's own visible label, so the message names what the user sees in the dialog rather
// than the engine's argument name. The label's first child is its text node;
// annotateSettingsFields() appends the hint AFTER the input, so it cannot be picked up here.
function fieldLabel(id) {
  const label = $(id).closest("label");
  const text = label && label.firstChild ? label.firstChild.textContent.trim() : "";
  return text || id;
}

// --- Settings dialog --------------------------------------------------------------
// The settings themselves live in a modal (index.html #settings-dialog); the rail keeps only
// the summary line, which must say enough that opening the dialog is a choice rather than the
// only way to learn the budget. It therefore carries the two values users actually change plus
// a warning when anything has drifted off the PySR-parity defaults (CLAUDE.md's
// highest-priority configuration rule — the GUI should never let that happen silently).
function fieldModified(id) {
  const d = DEFAULTS[id];
  const v = d.int ? parseInt($(id).value, 10) : parseFloat($(id).value);
  return !Number.isFinite(v) || v !== d.value;
}
function settingsModified() {
  return Object.keys(DEFAULTS).some(fieldModified);
}
// Same warning, per field: the summary says *something* diverged, this says which one, so a
// stray keystroke in a 17-field panel cannot hide.
function markModifiedFields() {
  Object.keys(DEFAULTS).forEach((id) => {
    const label = $(id).closest("label");
    if (label) label.classList.toggle("modified", fieldModified(id));
  });
}
function updateSettingsSummary() {
  // Batching is a checkbox, so it is outside DEFAULTS and invisible to the "modified"
  // marker — but it does change which candidates the search sees, and a change of that
  // size must never be silent in a rail that otherwise reports the budget faithfully.
  // The seed is printed only when it is NOT the shipped value. Every run is seeded and the
  // seed is 1 unless someone changes it, so "seed 1" was a constant string next to the one
  // number that does vary — it said nothing about this run. A seed the user has moved is a
  // fact about this run, and then it appears. Same rule as the "modified" marker.
  const seed = $("seed").value;
  const seedMoved = seed !== "" && Number(seed) !== DEFAULTS.seed.value;
  $("settings-summary").textContent =
    `${$("generations").value} generations` +
    (seedMoved ? ` · seed ${seed}` : "") +
    ($("batching").checked ? " · batching" : "") +
    (settingsModified() ? " · modified" : "");
  markModifiedFields();
}
// Every field the dialog owns, read and written as one unit, so the snapshot/restore and the
// reset button can never cover a subset of them (Batching used to survive both).
function readSettingsFields() {
  const s = {};
  Object.keys(DEFAULTS).forEach((id) => { s[id] = $(id).value; });
  Object.keys(CHECKBOX_DEFAULTS).forEach((id) => { s[id] = $(id).checked; });
  return s;
}
function writeSettingsFields(s) {
  Object.keys(DEFAULTS).forEach((id) => { if (id in s) $(id).value = s[id]; });
  Object.keys(CHECKBOX_DEFAULTS).forEach((id) => { if (id in s) $(id).checked = s[id]; });
  syncBatchingDependants(); // a restored/reset checkbox re-enables what it disabled
  updateSettingsSummary();
}
function resetDefaults() {
  const d = {};
  Object.keys(DEFAULTS).forEach((id) => { d[id] = String(DEFAULTS[id].value); });
  Object.keys(CHECKBOX_DEFAULTS).forEach((id) => { d[id] = CHECKBOX_DEFAULTS[id]; });
  writeSettingsFields(d);
}

// Print each field's shipped default and what it does, straight from DEFAULTS, so the dialog
// can never disagree with the values the search actually falls back to.
function annotateSettingsFields() {
  Object.keys(DEFAULTS).forEach((id) => {
    const label = $(id).closest("label");
    if (!label) return;
    const hint = document.createElement("span");
    hint.className = "field-hint";
    hint.textContent = `${DEFAULTS[id].note} Default ${DEFAULTS[id].value}.`;
    label.appendChild(hint);
  });
}

// Editing is transactional: every field the dialog owns is snapshotted on open so all four
// dismissal paths (Cancel, ×, Esc, backdrop) restore them, and only Apply keeps them. Without
// this a mistyped constant could only be undone by resetting the whole panel.
function openSettings() {
  state.settingsSnapshot = readSettingsFields();
  updateSettingsSummary();
  $("settings-dialog").showModal();
}
function closeSettings(keep) {
  const snapshot = state.settingsSnapshot;
  if (!keep && snapshot) writeSettingsFields(snapshot);
  state.settingsSnapshot = null;
  updateSettingsSummary();
  syncRowPolicyIfShapeChanged(); // Apply and every dismissal path can land on a new value
  refreshDataNotice();           // ... including a new Batching state, which the notice offers
  $("settings-dialog").close();
  // The commit point for the dialog's fields: nothing typed inside it is persisted until it
  // closes, so Apply stores what was kept and every dismissal path stores the restored snapshot.
  saveSearchSettings();
}

// --- Settings persistence ---------------------------------------------------------
// What the user sets up BEFORE a run — operators, macros, the dialog's fields and the two
// opt-ins — survives a reload. Nothing else does, and the exclusions are the point (docs/63):
//   * the data, because it never leaves the browser today and that claim is worth more than the
//     convenience, because a real table does not fit in localStorage (docs/59), and because
//     auto-restoring one into the fixed 128 MB heap would make a reload a broken page;
//   * the results, which describe data that is gone and already have export paths that survive
//     a reload properly (Pareto CSV, and the R/Python snippets that reproduce the run);
//   * the results-view controls, above all model_selection: it decides which equation the page
//     RECOMMENDS, and a fresh visit must recommend the shipped default, not a weeks-old click.
// The theme keeps its own older key; renaming it would silently reset every visitor's theme.
const SETTINGS_KEY = "rsymbolic2.search-settings.v1";
const MAX_STORED_CHARS = 64 * 1024; // a settings blob is well under 2 KB; larger is not ours
const MAX_STORED_MACROS = 20;
const MAX_MACRO_CHARS = 200;

function readSearchSettings() {
  const macros = readMacros();
  return {
    v: 1,
    fields: readSettingsFields(),
    binary_ops: checkedOps("bin"),
    unary_ops: checkedOps("un"),
    macros: macros.names.map((name, i) => ({ name, body: macros.bodies[i] })),
    opt_ins: {
      linear_scaling: $("linear_scaling").checked,
      eval_cache: $("eval_cache").checked,
    },
  };
}

// Persistence is a convenience, never a requirement: private mode and a full quota both throw,
// and the app must not notice. Same guard as toggleTheme().
function saveSearchSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(readSearchSettings()));
  } catch (e) { /* storage unavailable — the session simply is not remembered */ }
}

// One debounced funnel instead of a listener per control, so a control added later is covered
// by construction. Events from things we do not store (the file input, the target select) only
// cost a redundant identical write.
let saveTimer = null;
function saveSearchSettingsSoon() {
  // The dialog is transactional — openSettings() snapshots and all four dismissal paths restore
  // — so what is typed inside it is not committed state; closeSettings() saves the outcome.
  if ($("settings-dialog").open) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSearchSettings, 200);
}

// Storage is outside our control: a stale schema from an older deploy, a truncated write, or a
// hand-edited value can all turn up here. Everything is whitelisted against the tables that
// define the UI and anything unrecognised is dropped, so a bad blob costs the shipped defaults
// rather than a broken panel. A section left out entirely (an older schema) returns null and
// keeps the DOM's own defaults, which is why "invalid" and "absent" must not collapse together.
function loadSearchSettings() {
  let raw = null;
  try { raw = localStorage.getItem(SETTINGS_KEY); } catch (e) { return null; }
  if (!raw || raw.length > MAX_STORED_CHARS) return null;
  let s = null;
  try { s = JSON.parse(raw); } catch (e) { return null; }
  if (!s || typeof s !== "object" || s.v !== 1) return null;
  return {
    fields: sanitizeFields(s.fields),
    binary_ops: sanitizeOps(s.binary_ops, BINARY),
    unary_ops: sanitizeOps(s.unary_ops, UNARY),
    macros: sanitizeMacros(s.macros),
    opt_ins: sanitizeOptIns(s.opt_ins),
  };
}
function sanitizeFields(f) {
  if (!f || typeof f !== "object") return null;
  const out = {};
  Object.keys(DEFAULTS).forEach((id) => {
    const v = f[id];
    // Stored as the string the input holds. A blank field is legal while typing but is not
    // worth restoring — readConfig() would fall back to the default anyway.
    if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) out[id] = v;
  });
  Object.keys(CHECKBOX_DEFAULTS).forEach((id) => {
    if (typeof f[id] === "boolean") out[id] = f[id];
  });
  return out;
}
// Filtering the canonical list (rather than the stored one) drops operators this build no
// longer has and re-emits the rest in the order checkedOps() uses.
function sanitizeOps(list, canonical) {
  if (!Array.isArray(list)) return null;
  return canonical.filter((op) => list.includes(op));
}
function sanitizeMacros(list) {
  if (!Array.isArray(list)) return null;
  return list
    .filter((m) => m && typeof m.name === "string" && typeof m.body === "string")
    .slice(0, MAX_STORED_MACROS)
    .map((m) => ({
      name: m.name.slice(0, MAX_MACRO_CHARS),
      body: m.body.slice(0, MAX_MACRO_CHARS),
    }));
}
function sanitizeOptIns(o) {
  if (!o || typeof o !== "object") return null;
  const out = {};
  ["linear_scaling", "eval_cache"].forEach((id) => {
    if (typeof o[id] === "boolean") out[id] = o[id];
  });
  return out;
}

function applySearchSettings(s) {
  if (s.fields) writeSettingsFields(s.fields);
  if (s.binary_ops) applyOpSelection("bin", s.binary_ops);
  if (s.unary_ops) applyOpSelection("un", s.unary_ops);
  if (s.macros) {
    $("macro-list").textContent = "";
    s.macros.forEach((m) => addMacroRow(m.name, m.body));
    updateMacroSummary(); // addMacroRow does this too, but an empty saved list must clear it
  }
  if (s.opt_ins) {
    Object.keys(s.opt_ins).forEach((id) => { $(id).checked = s.opt_ins[id]; });
  }
  syncBatchingDependants();
}

// Whether anything the user configures differs from what the page ships with. Only then is the
// restore worth announcing — a browser that remembered the defaults has nothing to disclose.
function opsAreDefault(picked, defaults) {
  return picked.length === defaults.size && picked.every((op) => defaults.has(op));
}
function differsFromShippedDefaults() {
  return settingsModified() ||
    $("batching").checked ||
    $("linear_scaling").checked ||
    $("eval_cache").checked ||
    readMacros().names.length > 0 ||
    !opsAreDefault(checkedOps("bin"), BINARY_DEFAULT) ||
    !opsAreDefault(checkedOps("un"), UNARY_DEFAULT);
}

function restoreSearchSettings() {
  const saved = loadSearchSettings();
  if (!saved) { syncOperatorReset(); return; }
  applySearchSettings(saved);
  syncOperatorReset();
  $("settings-restored").hidden = !differsFromShippedDefaults();
}

// Reset for the operator library alone. Scoped deliberately: the operator set is the one
// problem input a user builds up over a session AND the only one the app itself adds to
// (applyExampleOps), so it needs an undo of its own — while macros, the opt-ins and the
// parity constants are all edited in place and visible where they are edited.
function resetOperators() {
  applyOpSelection("bin", [...BINARY_DEFAULT]);
  applyOpSelection("un", [...UNARY_DEFAULT]);
  syncOperatorReset();
  saveSearchSettings();
}

// Present-but-disabled, never hidden (index.html says why). The disabled state is also the
// only place the rail reports "these are the shipped operators", which is worth saying.
function syncOperatorReset() {
  const atDefault =
    opsAreDefault(checkedOps("bin"), BINARY_DEFAULT) &&
    opsAreDefault(checkedOps("un"), UNARY_DEFAULT);
  $("reset-ops").disabled = atDefault;
}

// A wider reset than the dialog's own button, which stays field-only on purpose (see
// CHECKBOX_DEFAULTS): this one lives in the rail, where every operator, macro and opt-in it
// clears is on screen, and it also forgets what was stored.
function useShippedDefaults() {
  clearTimeout(saveTimer); // a write still in flight would put back what this is clearing
  try { localStorage.removeItem(SETTINGS_KEY); } catch (e) { /* nothing to forget */ }
  resetDefaults();
  applyOpSelection("bin", [...BINARY_DEFAULT]);
  applyOpSelection("un", [...UNARY_DEFAULT]);
  syncOperatorReset();
  $("macro-list").textContent = "";
  updateMacroSummary();
  $("linear_scaling").checked = false;
  $("eval_cache").checked = false;
  syncBatchingDependants();
  $("settings-restored").hidden = true;
}

// --- Run --------------------------------------------------------------------------
function run() {
  if (!state.table) return;
  state.targetIndex = parseInt($("target-select").value, 10);
  state.featureIndices = currentFeatureIndices();
  if (state.featureIndices.length === 0) {
    // "Select at least one feature" is only true advice when there IS one to select. With a
    // one-column file, or a table whose other columns all hold a non-numeric cell, the Features
    // list is empty and that sentence asks for something the page cannot do — which is the
    // state a user reaches by pasting a table with one "NA" in it. Say what is actually wrong.
    const selectable = document.querySelectorAll('input[data-feature="1"]').length;
    setStatus(selectable
      ? "Select at least one feature."
      : "No column is available as an input: a feature must be a numeric column other than the " +
        "target. Load a table with at least two fully numeric columns.");
    return;
  }
  // The heap ceiling is re-checked here because it depends on settings the user can change
  // after loading the data (Populations) and on how many feature columns are ticked. The
  // fitted width is featureIndices + the target, not the whole file's column count.
  const width = state.featureIndices.length + 1;
  const limit = maxRowsForBrowser(width, configuredPopulations());
  if (state.table.rows.length > limit) {
    setStatus(
      `${fmtInt(state.table.rows.length)} rows exceed the engine's heap at ` +
      `${configuredPopulations()} populations (about ${fmtInt(limit)} rows fit). ` +
      `Lower the sample size or the population count.`);
    return;
  }
  const { X, y } = toMatrix(state.table, state.targetIndex, state.featureIndices);
  state.X = X;
  state.y = y;
  state.featureNames = state.featureIndices.map((j) => state.table.columns[j]);
  // Snapshot the target name with the matrices: the target <select> can change after the
  // run, but the displayed result must keep the names it was fitted with.
  state.targetName = state.table.columns[state.targetIndex];
  // Snapshotted with the matrices for the same reason: the sampling controls can move after
  // the run, but the exported snippet must describe the data this result was fitted on.
  state.sampling = isSampled()
    ? { fitted: state.table.rows.length, total: state.sourceTable.rows.length, seed: SAMPLE_SEED }
    : null;
  const config = readConfig();
  // Name-level macro problems are caught here rather than costing the user a launched run;
  // a bad macro BODY is reported by the engine itself, through the normal error path.
  const macroProblem = macroError(config.macro_names, config.macro_bodies);
  if (macroProblem) {
    setStatus(macroProblem);
    return;
  }
  // Same rule for an out-of-range setting: the engine would refuse it, but only after the run
  // is launched and in its own vocabulary. See settingError().
  const settingProblem = settingError(config);
  if (settingProblem) {
    setStatus(settingProblem);
    return;
  }
  state.config = config;

  // The restore notice is an arrival disclosure: once a run is launched the user has seen the
  // panel and committed to it. The Settings summary's "modified" marker carries on saying that
  // something is off the shipped values.
  $("settings-restored").hidden = true;

  setRunButton(true);
  document.body.classList.add("running"); // shows the header progress bar
  $("pareto-card").classList.remove("live"); // cleared again on the first snapshot's redraw
  state.lastProgressDraw = 0;
  // Back to the indeterminate sweep until the first snapshot says how long the run is.
  state.epoch = 0;
  state.totalEpochs = 0;
  state.epochMarks = [];
  document.body.classList.remove("determinate");
  $("progress-bar").style.setProperty("--progress", "0%");
  state.runStartedAt = new Date(); // wall-clock start, for the report's run summary
  startTimer();
  setStatus("running…");

  // Fresh worker per run so Stop == terminate is clean.
  if (state.worker) state.worker.terminate();
  state.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  state.worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === "ready") return;
    if (msg.type === "result") onResult(msg.result, msg.elapsed);
    else if (msg.type === "error") onError(msg.message);
    else if (msg.type === "progress") onProgress(msg);
  };
  state.worker.onerror = (e) => onError(e.message || "worker error");

  // Flatten here and TRANSFER the buffers: posting the array-of-row-arrays structure-clones
  // the whole dataset into the worker (a second copy, ~0.3 s of main-thread time at 500k
  // rows) only for the worker to flatten it anyway. state.X keeps the row arrays the charts
  // and predict() need, so detaching these buffers costs us nothing.
  const nrow = X.length;
  const ncol = nrow > 0 ? X[0].length : 0;
  const flat = new Float64Array(nrow * ncol);
  for (let i = 0; i < nrow; i++) {
    const row = X[i];
    for (let j = 0; j < ncol; j++) flat[i * ncol + j] = row[j];
  }
  const yFlat = Float64Array.from(y);
  state.worker.postMessage(
    { type: "run", X: flat, y: yFlat, nrow, ncol, options: config },
    [flat.buffer, yFlat.buffer],
  );
}

function stop() {
  if (state.worker) {
    state.worker.terminate();
    state.worker = null;
  }
  finishRun();
  restoreResultCharts();
  setStatus(state.result ? "stopped — showing the previous result" : "stopped");
}

// An aborted run leaves the Pareto canvas holding its live front (onProgress draws one per
// epoch), while the table, the hero equation and the metrics still describe the last COMPLETED
// run. Once the dimming and the "live" badge come off, that partial front reads as this panel's
// final answer. So both abort paths put the completed run's chart back — or clear the canvas
// when there is no completed run to fall back to.
function restoreResultCharts() {
  if (state.result) drawParetoChart();
  else destroyPlots();
}

function onResult(result, elapsed) {
  finishRun();
  state.result = result;
  // Compact completion chip: elapsed time and the configured generation budget, e.g.
  // "0.57s | generations: 2800". Use config.generations (what the Settings summary shows), not
  // the engine's epoch count — those are different units, and showing the epoch count here
  // would contradict the "2800 generations" line in the sidebar.
  // A run can also end before its budget — target_loss, the timeout and max_evals all stop the
  // loop early — and reporting the configured budget for one of those is simply false (the
  // bundled quadratic example hits target_loss in well under a second). The epoch counter from
  // the progress snapshots is what actually ran, so say that instead when the two disagree.
  const gens = state.config ? state.config.generations : null;
  const early = state.totalEpochs > 0 && state.epoch < state.totalEpochs;
  // Same three facts the chip is about to turn into a sentence, kept as values: the report
  // states them too, and re-deriving them from chip text would be reading the UI back.
  state.run = {
    startedAt: state.runStartedAt || new Date(),
    elapsed,
    stoppedEarly: early,
    epoch: state.epoch,
    totalEpochs: state.totalEpochs,
    source: state.dataSource,
  };
  const el = $("status");
  if (early) {
    setStatus(`${elapsed.toFixed(2)}s | stopped early: epoch ${state.epoch}/${state.totalEpochs}`);
    el.title = `Ended before the configured budget of ${fmtInt(gens)} generations ` +
               "(target loss, timeout or max evals).";
  } else {
    setStatus(gens != null
      ? `${elapsed.toFixed(2)}s | generations: ${fmtInt(gens)}`
      : `done in ${elapsed.toFixed(2)} s`);
    el.title = "Configured search budget.";
  }
  renderResult();
}

function onError(message) {
  // An OOM abort leaves the module dead but the worker alive, holding its 128 MB heap until
  // the next run replaces it. Drop it now; run() creates a fresh worker anyway.
  if (state.worker) { state.worker.terminate(); state.worker = null; }
  finishRun();
  restoreResultCharts(); // same stale-live-front hazard as stop()
  setStatus("error: " + message);
}

function finishRun() {
  stopTimer();
  document.body.classList.remove("running", "determinate");
  $("pareto-card").classList.remove("live"); // Stop/error/result all end the live state
  // The first-run reveal lasts exactly as long as the run: a finished run is revealed by
  // renderResult() (.has-result, synchronously after this, so nothing flashes), while a
  // stopped or failed one goes back to the placeholder — restoreResultCharts() has no
  // completed front to put back, and a partial one must not read as the answer.
  $("results-area").classList.remove("has-live");
  setRunButton(false);
}

// Live Pareto updates (docs/53): one ProgressSnapshot arrives per completed epoch from
// the worker. Redraws are throttled to >= 250 ms apart so a fast run (many small
// epochs) cannot flood the main thread with Chart.js rebuilds. The snapshot has no
// score/r2 (drawPareto degrades gracefully) and no bestIndex/selectedIndex/onSelect —
// nothing in a live front is clickable or highlighted, matching the CSS's
// pointer-events: none while body.running.
function onProgress(msg) {
  $("pareto-card").classList.add("live");
  // Before there is any result, the whole panel is still the placeholder and the Pareto card
  // is display:none — so on the first run of a session these snapshots would be drawn into a
  // hidden canvas and the user would watch an empty column for the entire search. This
  // promotes that one card (style.css) for the duration of the run; the rest stay hidden.
  $("results-area").classList.add("has-live");
  const now = performance.now();
  // Progress accounting is separate from the redraw throttle below: every epoch must be
  // timed even when its chart redraw is skipped, or the ETA's rate is computed from a
  // fraction of the epochs. The status line itself is written by the run timer (one owner,
  // see startTimer) — this only records what it will read.
  if (msg.epoch > state.epoch) {
    state.epoch = msg.epoch;
    state.totalEpochs = msg.total_epochs || state.totalEpochs;
    state.epochMarks.push(now);
    if (state.epochMarks.length > 6) state.epochMarks.shift();
    if (state.totalEpochs > 0) {
      document.body.classList.add("determinate");
      const frac = Math.min(1, state.epoch / state.totalEpochs);
      $("progress-bar").style.setProperty("--progress", `${(frac * 100).toFixed(1)}%`);
    }
  }
  if (now - state.lastProgressDraw < 250) return;
  state.lastProgressDraw = now;
  drawPareto($("pareto-canvas"), { complexity: msg.complexity, loss: msg.loss, score: null }, {
    logLoss: logLossEnabled(),
  });
}

// One morphing header button: "▶ Run" when idle, "■ Stop" (danger red) while a search is
// running. All three end-of-run paths (stop, result, error) restore it via finishRun().
function setRunButton(running) {
  const btn = $("run-btn");
  btn.textContent = running ? "■ Stop" : "▶ Run";
  btn.classList.toggle("danger", running);
  btn.disabled = !running && !state.table;
}

// One writer for the status line while a run is in flight. The epoch/ETA figures live in
// `state` and are read from here rather than written by onProgress, because this interval
// would overwrite anything onProgress printed within 200 ms of it.
function startTimer() {
  state.t0 = performance.now();
  state.timer = setInterval(() => {
    const secs = (performance.now() - state.t0) / 1000;
    let line = `${secs.toFixed(1)}s`;
    if (state.totalEpochs > 0) line += ` | epoch ${state.epoch}/${state.totalEpochs}`;
    const eta = estimateRemaining();
    if (eta != null) line += ` · ≤ ${formatDuration(eta)} left`;
    setStatus(line, false); // ticking text: never announced (see setStatus)
  }, 200);
}
function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

// Seconds still to run, or null when there is not enough evidence yet. Deliberately based
// on the RECENT epoch rate, not the whole run: early epochs carry small trees and are much
// cheaper than later ones, so an average over the whole run reads far too optimistic at the
// start. Nothing is shown before three epochs for the same reason. It stays an upper bound —
// target_loss, the timeout and max_evals can all end the run sooner.
function estimateRemaining() {
  if (state.totalEpochs <= 0 || state.epochMarks.length < 3) return null;
  const marks = state.epochMarks;
  const perEpoch = (marks[marks.length - 1] - marks[0]) / (marks.length - 1);
  const remaining = Math.max(0, state.totalEpochs - state.epoch);
  if (!(perEpoch > 0)) return null;
  return (perEpoch * remaining) / 1000;
}

function formatDuration(seconds) {
  if (seconds < 90) return `${Math.ceil(seconds)} s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

// --- Results rendering ------------------------------------------------------------
// Faithful port of the core select_best() (hall_of_fame.cpp) so the recommended (★) row can be
// re-picked instantly when the user changes model_selection, without re-running the search. The
// front already carries the same per-member `score` (pareto_scores) the core ranks by, so this
// only re-applies the selection rule — the C++ implementation stays authoritative. The front is
// strictly decreasing in loss, so its last (most complex) entry is the lowest-loss one.
function selectBestIndex(front, mode) {
  const n = front.complexity.length;
  if (n <= 1) return 0;
  if (mode === "accuracy") return n - 1; // lowest loss
  // "best" keeps only members within 1.5× of the lowest loss; "score" keeps the whole front.
  const minLoss = front.loss[n - 1];
  const threshold = mode === "best" ? 1.5 * minLoss : Infinity;
  let bestIdx = 0;
  let bestScore = -Infinity;
  for (let i = 1; i < n; i++) {
    if (front.loss[i] > threshold) continue; // outside the accuracy band ("best")
    if (front.score[i] > bestScore) { bestScore = front.score[i]; bestIdx = i; }
  }
  return bestIdx;
}

function renderResult() {
  // Reveal the results + detail panels (hidden behind a placeholder until the first run).
  $("results-area").classList.add("has-result");
  // Defensive: the normal path already clears .live via finishRun() before this runs, but
  // renderResult() is the authoritative "final answer replaces everything" point (docs/53),
  // so it also guarantees no stale live badge/opacity-exemption survives a result render.
  $("pareto-card").classList.remove("live");
  const res = state.result;
  const front = res.pareto_front;
  const hasSst = res.sst && isFinite(res.sst) && res.sst > 0;
  front.r2 = front.loss.map((l) => (hasSst ? 1 - l / res.sst : null));

  renderTable(res, front);
  drawParetoChart();
  selectEquation(res.best_index != null ? res.best_index : front.complexity.length - 1);
  renderEvalAccounting(res);
  $("print-btn").disabled = false; // there is now a run to report on
}

// Re-pick the recommended equation for the current model_selection, in place (no re-run).
function applyModelSelection() {
  if (!state.result) return;
  // Keep the snapshotted config in step: it is what the Python/R snippets are generated from,
  // and model_selection is the one setting the user can still change after the run. It picks a
  // member of an existing front, so nothing about the search is invalidated by moving it.
  if (state.config) state.config.model_selection = $("model_selection").value;
  const front = state.result.pareto_front;
  state.result.best_index = selectBestIndex(front, $("model_selection").value);
  document.querySelectorAll("#pareto-table tbody tr").forEach((tr) => {
    tr.classList.toggle("recommended", parseInt(tr.dataset.index, 10) === state.result.best_index);
  });
  selectEquation(state.result.best_index);
}

function renderTable(res, front) {
  const tbody = $("pareto-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (let i = 0; i < front.complexity.length; i++) {
    const tr = document.createElement("tr");
    tr.dataset.index = String(i);
    if (i === res.best_index) tr.classList.add("recommended");
    tr.innerHTML =
      `<td>${i}</td><td>${fmtComplexity(front, i)}</td><td>${fmt(front.loss[i])}</td>` +
      `<td>${fmt(front.score[i])}</td><td>${front.r2[i] == null ? "—" : fmt(front.r2[i])}</td>` +
      `<td title="${esc(front.expression[i])}">${esc(front.expression_simplified ? front.expression_simplified[i] : front.expression[i])}</td>`;
    tr.addEventListener("click", () => selectEquation(i));
    tbody.appendChild(tr);
  }
}

function drawParetoChart() {
  const front = state.result.pareto_front;
  drawPareto($("pareto-canvas"), front, {
    bestIndex: state.result.best_index,
    logLoss: logLossEnabled(),
    selectedIndex: state.selectedIndex,
    onSelect: (i) => selectEquation(i),
  });
}

function selectEquation(i) {
  const front = state.result.pareto_front;
  if (i == null || i < 0 || i >= front.complexity.length) return;
  state.selectedIndex = i;

  // Highlight table row.
  document.querySelectorAll("#pareto-table tbody tr").forEach((tr) => {
    tr.classList.toggle("selected", parseInt(tr.dataset.index, 10) === i);
  });
  drawParetoChart();

  // Equation detail. Display surfaces prefer the simplified companions (docs/52);
  // everything evaluated (predict below, Copy expression) stays on the raw string,
  // which is the frozen round-trip source (docs/48 D2).
  const latex = front.latex_simplified ? front.latex_simplified[i] : front.latex[i];
  renderInto($("eq-latex"), latex, state.featureNames);
  const eqEl = $("eq-string");
  const shown = front.expression_simplified ? front.expression_simplified[i] : front.expression[i];
  eqEl.textContent = shown;
  // The raw string is the only thing on this card that is NOT the displayed form, so name it
  // rather than showing a bare second string — and only when the simplifier actually changed
  // something, since an identical tooltip repeating the line under it says nothing.
  eqEl.title = shown === front.expression[i] ? ""
    : `As searched, before display simplification (docs/52): ${front.expression[i]}`;

  // Tree of the same string the hero card shows (docs/48 D6). Its node count is the printed
  // form's, not the `complexity` column's — see the caption's title attribute.
  state.treeSvg = renderTree($("tree-box"), shown, state.featureNames);
  const treeNodes = state.treeSvg ? state.treeSvg.querySelectorAll(".tree-node").length : 0;
  $("tree-nodes").textContent =
    treeNodes ? `${treeNodes} node${treeNodes === 1 ? "" : "s"}` : "";
  const r2 = front.r2 ? front.r2[i] : null;
  $("eq-metrics").innerHTML =
    metric("loss", fmt(front.loss[i])) +
    metric("complexity", fmtComplexity(front, i)) +
    metric("score", fmt(front.score[i])) +
    metric("R²", r2 == null ? "—" : fmt(r2));

  // Prediction plot, labelled with the real column names captured at run time. Only a
  // bounded stride subset is evaluated and drawn: Chart.js rebuilds the whole dataset on
  // every selection and theme toggle, and the metrics above come from the engine (loss, and
  // R² from res.sst), so nothing displayed depends on predicting every row here.
  // The fit view keeps one heading for both its modes (curve overlay / predicted-vs-actual);
  // the column identity is not lost, since the single-feature scatter still labels its axes
  // with the real target/feature names (drawPrediction). The residual view renames it.
  const residual = $("fit-view").value === "residual";
  $("fit-title").textContent = residual ? "Residuals" : "Predicted vs actual";
  const idx = strideIndices(state.X.length, DISPLAY_POINT_CAP);
  const Xd = idx ? idx.map((k) => state.X[k]) : state.X;
  const yd = idx ? idx.map((k) => state.y[k]) : state.y;
  $("fit-note").textContent = idx
    ? `Showing ${fmtInt(idx.length)} of ${fmtInt(state.X.length)} points.` : "";
  try {
    const yhat = predict(front.expression[i], Xd);
    if (residual) {
      drawResidual($("pred-canvas"), yd, yhat, { yLabel: state.targetName });
    } else {
      drawPrediction($("pred-canvas"), Xd, yd, yhat, {
        xLabel: state.featureNames[0],
        yLabel: state.targetName,
      });
    }
  } catch (e) {
    // The expression could not be evaluated on these inputs; drop the chart rather than leave
    // the previous equation's fit under this one's heading.
    destroyPrediction();
  }
}

function renderEvalAccounting(res) {
  const c = res.eval_counts || {};
  $("eval-accounting").textContent =
    `evaluations: ${fmtInt(res.n_evals)} (forward ${fmtInt(c.forward)}, LM residual ${fmtInt(c.lm_resid)})` +
    (c.cache_hits ? `, cache hits ${fmtInt(c.cache_hits)}` : "");
}

// --- Export wiring ----------------------------------------------------------------
function wireExport() {
  $("copy-expr").addEventListener("click", () => {
    if (state.selectedIndex != null) {
      const front = state.result.pareto_front;
      // Same rule as LaTeX and SymPy below: copy what is displayed. This button sits beside
      // the string, so it must hand over that string — it used to copy the raw round-trip
      // form instead, which differs whenever the display simplifier fired (x*x -> (x ^ 2)).
      // The raw form stays reachable: it is the element's hover title and the CSV's
      // `expression` column.
      copyText((front.expression_simplified || front.expression)[state.selectedIndex]);
    }
  });
  $("copy-latex").addEventListener("click", () => {
    if (state.selectedIndex != null) {
      const front = state.result.pareto_front;
      // Copy what is displayed: the simplified LaTeX (display-only, no round-trip duty).
      copyText((front.latex_simplified || front.latex)[state.selectedIndex]);
    }
  });
  $("copy-sympy").addEventListener("click", () => {
    if (state.selectedIndex != null) {
      const front = state.result.pareto_front;
      // Same rule as LaTeX above: copy what is displayed, i.e. the simplified form. Falls
      // back to the raw rendering if the engine is an older build without the field.
      copyText((front.sympy_simplified || front.sympy)[state.selectedIndex]);
    }
  });
  $("copy-python").addEventListener("click", () => {
    if (state.config) copyText(pythonCall(state.config, state.sampling));
  });
  $("copy-r").addEventListener("click", () => {
    if (state.config) copyText(rCall(state.config, state.sampling));
  });
  $("download-csv").addEventListener("click", () => {
    if (state.result) downloadText("rsymbolic2_pareto.csv", paretoCsv(state.result.pareto_front), "text/csv");
  });
  $("download-tree").addEventListener("click", () => {
    if (state.treeSvg)
      downloadText("rsymbolic2_tree.svg", treeSvgStandalone(state.treeSvg), "image/svg+xml");
  });
}

// --- Printable report (docs/64) ---------------------------------------------------
// Everything report.js needs, as plain data. Assembled here for the same reason export.js
// takes a config object rather than reading the DOM: the report is then a function of the
// run, not of the app's current layout, and neither module can drift into reading the other's
// element ids.

// Every search setting with the PySR-parity default it is measured against. DEFAULTS covers
// the numeric dialog fields; the four below are the checkboxes and the results-side selector,
// which live outside it. model_selection appears with PySR's "best" as its reference even
// though the GUI ships "score" — that divergence is sanctioned (CLAUDE.md) but it decides
// which equation the report calls the answer, so it is stated, not hidden.
const EXTRA_SETTING_DEFAULTS = {
  batching: false,
  linear_scaling: false,
  eval_cache: false,
  model_selection: "best",
};

function reportSettingRows(cfg) {
  const rows = Object.keys(DEFAULTS).map((id) => (
    { name: id, value: cfg[id], fallback: DEFAULTS[id].value }));
  Object.keys(EXTRA_SETTING_DEFAULTS).forEach((id) => {
    rows.push({ name: id, value: cfg[id], fallback: EXTRA_SETTING_DEFAULTS[id] });
  });
  return rows;
}

function reportContext() {
  const res = state.result;
  const front = res.pareto_front;
  const i = state.selectedIndex;
  const residual = $("fit-view").value === "residual";
  // The same bounded stride subset the on-screen fit chart draws, so the printed figure is
  // the printed copy of what was on screen rather than a second, differently-sampled chart.
  const idx = strideIndices(state.X.length, DISPLAY_POINT_CAP);
  const Xd = idx ? idx.map((k) => state.X[k]) : state.X;
  const yd = idx ? idx.map((k) => state.y[k]) : state.y;
  let fit = null;
  try {
    const yhat = predict(front.expression[i], Xd);
    fit = residual
      ? residualImage(yd, yhat, { yLabel: state.targetName })
      : fitImage(Xd, yd, yhat, { xLabel: state.featureNames[0], yLabel: state.targetName });
  } catch (e) {
    // Same failure the on-screen card handles by dropping its chart: the equation could not
    // be evaluated on these inputs. The report keeps every other section.
  }

  return {
    version: APP_VERSION,
    generatedAt: new Date(),
    source: state.run.source,
    data: {
      rows: state.X.length,
      columns: state.featureNames.length + 1,
      targetName: state.targetName,
      featureNames: state.featureNames,
    },
    sampling: state.sampling,
    front,
    selectedIndex: i,
    bestIndex: res.best_index,
    modelSelection: state.config.model_selection,
    charts: {
      pareto: paretoImage(front, {
        bestIndex: res.best_index,
        logLoss: logLossEnabled(),
        selectedIndex: i,
      }),
      // The on-screen chart is read with the legend printed above it; the figure is not, so
      // the caption has to name the marks. It must also match what the figure actually draws:
      // a point that is both selected and recommended is drawn ONLY in the selected colour
      // (plots.js pointColors), so there is no second mark to point at.
      paretoCaption: i === res.best_index
        ? "Pareto front: loss against complexity. The highlighted point is the equation " +
          "above, which is also the recommended one."
        : "Pareto front: loss against complexity. Red is the equation above; green is the " +
          "recommended equation.",
      fit,
      fitCaption: residual
        ? "Residuals (actual − predicted) against predicted. Structure around the zero line " +
          "means the equation is missing something a high R² can hide."
        : "The selected equation against the data.",
    },
    fitNote: $("fit-note").textContent,
    tree: state.treeSvg ? state.treeSvg.cloneNode(true) : null,
    config: state.config,
    settingRows: reportSettingRows(state.config),
    run: state.run,
    evalCounts: { n_evals: res.n_evals, ...(res.eval_counts || {}) },
    snippets: {
      python: pythonCall(state.config, state.sampling),
      r: rCall(state.config, state.sampling),
    },
  };
}

// Build the report into its container and let the print stylesheet know there is one. The
// body class is what gates the "hide the app, show the report" rules: a print started with no
// result at all then falls back to printing the app, rather than a blank page.
function buildPrintReport() {
  if (!state.result || !state.run || state.selectedIndex == null) return false;
  buildReport($("print-report"), reportContext());
  document.body.classList.add("has-print-report");
  return true;
}

function clearPrintReport() {
  document.body.classList.remove("has-print-report");
  $("print-report").textContent = "";
}

// The supported path. A data URI is not painted the instant its src is set, so printing
// before the decode gives blank chart boxes — hence the await, and hence a button handler
// that is async while beforeprint (which cannot await anything) stays a best-effort net.
async function printReport() {
  if (!buildPrintReport()) return;
  await Promise.all([...$("print-report").querySelectorAll("img")].map(
    (img) => (img.decode ? img.decode().catch(() => {}) : Promise.resolve())));
  window.print();
}

// --- Column splitters -------------------------------------------------------------
// Each `.gutter` sits in a dedicated grid track between two panes. Dragging it writes the
// left pane's pixel width into the container's CSS variable (--col-config on .layout);
// the right pane keeps its `1fr`/minmax track and absorbs the rest. Chart.js is
// responsive, so the plots re-fit to their container automatically.
function initSplitters() {
  const VAR = { main: "--col-config" };
  const MIN_LEFT = 240; // px; keep the left pane usable
  const MIN_RIGHT = 320; // px; keep the right pane usable
  document.querySelectorAll(".gutter").forEach((gutter) => {
    const container = gutter.parentElement;
    const cssVar = VAR[gutter.dataset.split];
    const leftPane = gutter.previousElementSibling;
    if (!cssVar || !leftPane) return;

    let startX = 0;
    let startW = 0;
    const onMove = (e) => {
      const max = container.clientWidth - MIN_RIGHT - gutter.offsetWidth;
      const w = Math.max(MIN_LEFT, Math.min(startW + (e.clientX - startX), max));
      container.style.setProperty(cssVar, `${w}px`);
      // Switch this grid from its static default tracks to the var()-driven tracks. Doing it
      // only on the first drag keeps the never-dragged layout free of any var()-in-grid usage.
      container.classList.add("resized");
    };
    const onUp = (e) => {
      gutter.classList.remove("dragging");
      gutter.releasePointerCapture(e.pointerId);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    gutter.addEventListener("pointerdown", (e) => {
      startX = e.clientX;
      startW = leftPane.getBoundingClientRect().width;
      gutter.classList.add("dragging");
      gutter.setPointerCapture(e.pointerId);
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
      e.preventDefault();
    });
  });
}

// --- Helpers ----------------------------------------------------------------------
// The visible status chip, plus a screen-reader announcement for the DISCRETE messages only.
// The run timer rewrites the chip five times a second (startTimer), which as a live region
// would be unusable, so it opts out; everything a user needs to hear — errors, "data loaded",
// the finished run, a stop — announces by default.
function setStatus(s, announce = true) {
  $("status").textContent = s;
  if (announce) $("status-live").textContent = s;
}
function metric(k, v) { return `<span class="k">${k}</span><span>${v}</span>`; }
// Is a modal <dialog> open? The browser makes everything behind a modal inert for POINTER
// input, so every button on this page already respects it — but the two document-level
// shortcuts below are keyboard-only paths that were reachable from behind an open dialog, and
// both MUTATE state the dialog is meant to own for the duration:
//   * Ctrl+Enter launched a run that read the settings dialog's UNCOMMITTED fields. That panel
//     is transactional (openSettings snapshots, all four dismissal paths restore), so Cancel
//     then discarded the very values the finished run had used — leaving the rail reporting
//     "2800 generations" beside a result the chip said took 7. The run also started BEHIND the
//     modal, where its progress bar and its Stop button cannot be reached.
//   * Ctrl+V replaced the loaded table while the data preview was open, so that dialog went on
//     showing (and its note went on counting) rows that were no longer loaded.
// Ctrl+P is deliberately NOT guarded: it prints rather than mutates, and the print stylesheet
// already hides dialogs (body.has-print-report dialog { display: none }). Blocking it here
// would only hand the print to the browser's own path, which cannot await the chart images.
function modalOpen() {
  return document.querySelector("dialog[open]") !== null;
}
// Escapes for BOTH text and attribute contexts (renderTable puts expressions in a title=""),
// so the quote characters are part of the set even though today's callers cannot produce them.
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- Event wiring -----------------------------------------------------------------
function init() {
  updateThemeToggleIcon();
  $("theme-toggle").addEventListener("click", toggleTheme);
  buildOperatorChecks();
  buildMacroPresets();
  buildExamples();
  initSplitters();
  // After the controls it writes into exist, and before annotateSettingsFields() /
  // updateSettingsSummary() below, so the summary and the per-field "modified" marks are
  // computed from the restored values on first paint rather than from the HTML defaults.
  restoreSearchSettings();

  $("add-macro").addEventListener("click", () => addMacroRow());

  $("run-btn").addEventListener("click", () => {
    if (document.body.classList.contains("running")) stop();
    else run();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      // A dialog owns the keyboard while it is open, exactly as it already owns the pointer
      // (modalOpen). Falling through here started a run on the settings panel's uncommitted
      // values, behind a modal that hides both the progress bar and Stop.
      if (modalOpen()) return;
      e.preventDefault();
      if (state.table && !document.body.classList.contains("running")) run();
      return;
    }
    // Ctrl/Cmd+P is how most people start a print, so it takes the same guaranteed-complete
    // path as the button rather than racing the browser to build the report (see printReport).
    // Only claimed when there is a result to print; otherwise the browser's own print runs.
    if ((e.ctrlKey || e.metaKey) && (e.key === "p" || e.key === "P") && state.result) {
      e.preventDefault();
      printReport();
    }
  });
  $("target-select").addEventListener("change", (e) => {
    state.targetIndex = parseInt(e.target.value, 10);
    renderFeatureList(); // rebuilt with every feature ticked, so the fitted width can move
    syncRowPolicyIfShapeChanged();
  });
  // Delegated: the feature checkboxes are rebuilt on every target change. Ticking one changes
  // the fitted width, hence the row ceiling — see fittedWidth().
  $("feature-list").addEventListener("change", syncRowPolicyIfShapeChanged);
  $("features-all").addEventListener("click", () => setAllFeatures(true));
  $("features-none").addEventListener("click", () => setAllFeatures(false));
  $("reset-ops").addEventListener("click", resetOperators);
  // The reset's enabled state follows the pills, and the pills move through three paths:
  // a click, a restored session, and applyExampleOps(). One delegated listener plus the two
  // explicit calls below (restore, example load) covers all of them.
  document.addEventListener("change", (e) => {
    if (e.target && e.target.dataset && e.target.dataset.kind) syncOperatorReset();
  });
  $("logloss").addEventListener("change", () => { if (state.result) drawParetoChart(); });
  $("model_selection").addEventListener("change", applyModelSelection);
  // Redraw the currently selected equation in the other view; selectEquation() reads the
  // toggle, so this needs no state of its own.
  $("fit-view").addEventListener("change", () => {
    if (state.result) selectEquation(state.selectedIndex);
  });

  $("file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    // Clear the control, not just read it: a file input fires `change` only when the chosen
    // file list DIFFERS from the one it holds, so picking the same path again — the shape of
    // "I fixed the CSV, load it again" — fired nothing and the page silently kept the old
    // table. Same defect the example picker had (docs/75 finding 5), same remedy. Safe to do
    // before the read: `file` is a Blob reference that outlives the input's value.
    e.target.value = "";
    if (file) loadFile(file);
  });
  // Drag & drop onto the data card's drop zone (clicking it opens #file-input via the label).
  const zone = $("drop-zone");
  ["dragenter", "dragover"].forEach((t) => zone.addEventListener(t, (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  }));
  ["dragleave", "drop"].forEach((t) => zone.addEventListener(t, (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
  }));
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadFile(file);
  });
  // Paste a table from the clipboard anywhere on the page. This replaces a dedicated
  // textarea + button: same destination (parseTable), no permanent UI. Typing/pasting into
  // a real field must keep working, so events originating in a form control or editable
  // element are left alone.
  document.addEventListener("paste", (e) => {
    const t = e.target;
    const tag = t && t.tagName ? t.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select" || (t && t.isContentEditable)) return;
    // Same modality rule as Ctrl+Enter above: loading a table is the one thing this shortcut
    // does, and doing it behind the open data preview left that dialog describing data the app
    // no longer holds.
    if (modalOpen()) return;
    const text = e.clipboardData && e.clipboardData.getData("text");
    if (!text || !text.trim()) return;
    e.preventDefault();
    loadTextAsTable(text, { label: "pasted table", kind: "paste" });
  });

  // Empty-state shortcuts. Only one of the two is ever visible (body.has-data, style.css):
  // the example loader before there is data, Run afterwards — so the example button can no
  // longer replace a table the user has just loaded.
  $("placeholder-run-example").addEventListener("click", () => {
    loadTable(EXAMPLES[0].make(), { label: EXAMPLES[0].label, kind: "example" });
    run();
  });
  $("placeholder-run").addEventListener("click", run);

  // The summary line is the preview trigger; "change" re-opens the intake block.
  $("data-summary").addEventListener("click", openPreviewDialog);
  $("change-data").addEventListener("click", () => {
    document.body.classList.toggle("editing-data");
  });
  $("preview-close").addEventListener("click", () => $("preview-dialog").close());
  // Backdrop click closes: only the <dialog> element itself is hit outside the modal box.
  $("preview-dialog").addEventListener("click", (e) => {
    if (e.target === $("preview-dialog")) $("preview-dialog").close();
  });

  // Sampling controls. Both re-derive the fitted table from the parsed source, so the size
  // can be raised and lowered without reloading the file.
  $("sample-rows").addEventListener("change", () => { if (state.sourceTable) reapplyRowPolicy(); });
  $("sample-all").addEventListener("change", () => { if (state.sourceTable) reapplyRowPolicy(); });
  $("sample-size").addEventListener("change", () => { if (state.sourceTable) reapplyRowPolicy(); });

  // Keep the Settings summary in step with the fields it reports (including the "modified"
  // marker), and offer the one-click way back to the shipped PySR-parity defaults.
  Object.keys(DEFAULTS).forEach((id) => $(id).addEventListener("input", updateSettingsSummary));
  $("batching").addEventListener("change", () => {
    updateSettingsSummary();
    syncBatchingDependants();
  });
  syncBatchingDependants();
  // An edit to Populations can turn a table that fitted into one that does not; re-evaluate
  // the policy instead of waiting for Run to refuse.
  $("n_populations").addEventListener("change", syncRowPolicyIfShapeChanged);
  $("reset-defaults").addEventListener("click", resetDefaults);
  $("use-defaults").addEventListener("click", useShippedDefaults);
  // One delegated funnel for everything persisted: typed fields, operator pills, macro rows and
  // the opt-ins all reach it, and a control added later is covered without new wiring. The three
  // paths it cannot see (dialog close, example operators, macro removal) call the saver directly.
  ["input", "change"].forEach((t) => document.addEventListener(t, saveSearchSettingsSoon));
  annotateSettingsFields();
  updateSettingsSummary();

  // "edit" is the only trigger; the summary beside it is plain text (index.html says why).
  // Every dismissal path (Cancel, ×, Esc, backdrop) discards.
  $("open-settings").addEventListener("click", openSettings);
  $("settings-apply").addEventListener("click", () => closeSettings(true));
  $("settings-cancel").addEventListener("click", () => closeSettings(false));
  $("settings-close").addEventListener("click", () => closeSettings(false));
  // Esc: take over the native close so the snapshot is restored on the way out.
  $("settings-dialog").addEventListener("cancel", (e) => {
    e.preventDefault();
    closeSettings(false);
  });
  $("settings-dialog").addEventListener("click", (e) => {
    if (e.target === $("settings-dialog")) closeSettings(false);
  });

  wireExport();

  // Report printing. The button and Ctrl+P go through printReport(); `beforeprint` catches
  // the browser's own menu, where there is nothing to await on — it builds synchronously and
  // may miss the chart images on the first print of a session. `afterprint` drops the report
  // again so the DOM (and the chart data URIs) do not linger.
  $("print-btn").addEventListener("click", printReport);
  window.addEventListener("beforeprint", () => {
    if (!document.body.classList.contains("has-print-report")) buildPrintReport();
  });
  window.addEventListener("afterprint", clearPrintReport);
}

init();
