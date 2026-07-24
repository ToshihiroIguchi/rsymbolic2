// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// UI wiring and application state for the rsymbolic2 web GUI. Ties together data intake
// (data.js), the WASM search run (worker.js), the result plots (plots.js), equation
// rendering (latex.js) and export (export.js). No framework — plain DOM + ES modules.

import {
  parseTable, numericColumns, toMatrix, maxRowsForBrowser, sampleTable, strideIndices,
} from "./data.js";
import { EXAMPLES } from "./examples.js";
import { drawPareto, drawPrediction, destroyPlots } from "./plots.js";
import { renderInto } from "./latex.js";
import { predict } from "./predict.js";
import {
  copyText, pythonCall, rCall, paretoCsv, downloadText,
} from "./export.js";
import { fmt, fmtInt } from "./format.js";

// Canonical unary order sent to the engine. The core picks operators by a random index into
// this list (mutation.cpp: space.unary_ops[op(rng)]), so the list ORDER is part of the search
// trajectory for a fixed seed. The checkboxes are grouped only for readability (UNARY_GROUPS);
// checkedOps("un") re-emits in this canonical order so the visual grouping never changes the
// search. Keep this array's order stable.
const UNARY = ["neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square", "inv"];
const UNARY_GROUPS = [
  { label: "Trigonometric", ops: ["sin", "cos", "tanh"] },
  { label: "Exp / Log", ops: ["exp", "log"] },
  { label: "Power / other", ops: ["sqrt", "square", "inv", "abs", "neg"] },
];
const UNARY_DEFAULT = new Set(["neg", "exp", "log", "sin", "cos"]);
const BINARY = ["add", "sub", "mul", "div", "pow"];
const BINARY_DEFAULT = new Set(["add", "sub", "mul"]);

// --- Data-size policy (docs/59) ---------------------------------------------------
// Three measured facts drive every constant here. (1) A default-budget run costs ~2.83M
// evaluations, each O(rows): 200 rows take 18.5 s, 2,000 rows 154 s, and 100,000 rows
// about two hours — single-threaded, with no way for the user to know that in advance.
// (2) The WASM heap is fixed at 128 MB and aborts on over-allocation, so the row ceiling
// must be predicted before the run (data.js maxRowsForBrowser). (3) Reading and parsing
// happen synchronously on the main thread, so an enormous file freezes the UI before any
// of our checks could run — hence the byte cap, tested before the file is read.
const MAX_INPUT_BYTES = 64 * 1024 * 1024;
const ROW_WARN_THRESHOLD = 5000;   // above this a default-budget run is minutes, not seconds
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
const MACRO_PRESETS = [
  { name: "gauss", body: "exp(-square(x))" },
  { name: "sigmoid", body: "1 / (1 + exp(-x))" },
  { name: "softplus", body: "log(1 + exp(x))" },
  { name: "cube", body: "x^3" },
  { name: "decay", body: "exp(-1.0 * x)" },
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

const state = {
  sourceTable: null, // { columns, rows } exactly as parsed
  table: null, // the table actually fitted — sourceTable, or a sample of it
  rowLimit: 0, // rows the engine can hold for this shape (data.js maxRowsForBrowser)
  policyPopulations: 0, // the n_populations the current rowLimit/sample was derived from
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
  selectedIndex: null,
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
  BINARY.forEach((op) => bwrap.appendChild(opCheck("bin", op, BINARY_DEFAULT.has(op))));
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
    g.ops.forEach((op) => grid.appendChild(opCheck("un", op, UNARY_DEFAULT.has(op))));
    group.appendChild(grid);
    uwrap.appendChild(group);
  });
}
function opCheck(kind, op, checked) {
  const l = document.createElement("label");
  l.className = "check";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = checked;
  cb.dataset.kind = kind;
  cb.value = op;
  l.appendChild(cb);
  l.appendChild(document.createTextNode(op));
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
function addMacroRow(name = "", body = "") {
  const row = document.createElement("div");
  row.className = "macro-row";
  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.placeholder = "name";
  nameInput.value = name;
  nameInput.dataset.macro = "name";
  const bodyInput = document.createElement("input");
  bodyInput.type = "text";
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

function buildMacroPresets() {
  const sel = $("macro-preset");
  MACRO_PRESETS.forEach((m, i) => {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `${m.name} = ${m.body}`;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => {
    const m = MACRO_PRESETS[parseInt(sel.value, 10)];
    if (m) addMacroRow(m.name, m.body);
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
    loadTable(ex.make());
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

// --- Data intake ------------------------------------------------------------------
function loadTable(table) {
  // New data invalidates everything downstream of it, including a search still in flight:
  // its result would render against the table it can no longer be read as describing.
  if (document.body.classList.contains("running")) {
    if (state.worker) { state.worker.terminate(); state.worker = null; }
    finishRun();
  }
  clearResults();
  state.sourceTable = table;
  applyRowPolicy(true);
  // Classify columns on the FULL table: a column holding text in a row that sampling happens
  // to drop is still not a numeric column, and this way the target/feature lists do not
  // reshuffle when the sample size changes.
  state.numeric = numericColumns(state.sourceTable);
  buildTargetAndFeatures();
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
  state.selectedIndex = null;
  $("results-area").classList.remove("has-result");
  $("pareto-card").classList.remove("live");
  $("pareto-table").querySelector("tbody").innerHTML = "";
  $("eq-latex").innerHTML = "";
  $("eq-string").textContent = "";
  $("eq-string").title = "";
  $("eq-metrics").innerHTML = "";
  $("eval-accounting").textContent = "";
  $("fit-note").textContent = "";
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
function loadTextAsTable(text) {
  const problem = tooLarge(text.length);
  if (problem) { setStatus(problem); return; }
  try { loadTable(parseTable(text)); }
  catch (err) { setStatus("parse error: " + err.message); }
}
function loadFile(file) {
  const problem = tooLarge(file.size);
  if (problem) { setStatus(problem); return; }
  const reader = new FileReader();
  reader.onload = () => loadTextAsTable(reader.result);
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
  // the live setting rather than the shipped default: raising Populations lowers it.
  state.rowLimit = maxRowsForBrowser(src.columns.length, configuredPopulations());
  const over = n > state.rowLimit;

  const box = $("sample-rows");
  const sizeInput = $("sample-size");
  $("sample-control").hidden = n <= ROW_WARN_THRESHOLD;
  sizeInput.max = String(state.rowLimit);
  if (fresh) {
    box.checked = over;
    sizeInput.value = String(Math.min(DEFAULT_SAMPLE_ROWS, state.rowLimit));
  }
  box.disabled = over;
  if (over) box.checked = true;

  let k = parseInt(sizeInput.value, 10);
  if (!Number.isFinite(k) || k < 1) k = Math.min(DEFAULT_SAMPLE_ROWS, state.rowLimit);
  k = Math.min(k, state.rowLimit);
  sizeInput.value = String(k);

  state.table = box.checked ? sampleTable(src, k, SAMPLE_SEED) : src;
  state.policyPopulations = configuredPopulations();
  renderDataSummary();
  renderDataNotice(over);
}

// Populations moves the row ceiling, and it can change through paths that fire no `change`
// event — the settings dialog restores every field by assignment on Cancel/Esc. Comparing
// against the value the current policy was computed from covers all of them, and skips the
// needless result-clearing when a field was retyped to the same number.
function syncRowPolicyIfPopulationsChanged() {
  if (state.sourceTable && configuredPopulations() !== state.policyPopulations) {
    reapplyRowPolicy();
  }
}

// A change to the sampling controls (or to Populations) changes which rows are fitted, so
// it invalidates a displayed result and a search in flight exactly like loading a new file.
function reapplyRowPolicy() {
  if (document.body.classList.contains("running")) {
    if (state.worker) { state.worker.terminate(); state.worker = null; }
    finishRun();
    setStatus("stopped — the data changed");
  }
  clearResults();
  applyRowPolicy(false);
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

// Is the displayed result fitted on fewer rows than the file holds?
function isSampled() {
  return !!state.sourceTable && state.table.rows.length < state.sourceTable.rows.length;
}

function renderDataNotice(over) {
  const el = $("data-notice");
  const n = state.sourceTable.rows.length;
  const fitted = state.table.rows.length;
  el.classList.remove("warn", "acted");
  if (over) {
    el.classList.add("acted");
    el.textContent =
      `${fmtInt(n)} rows exceed what the in-browser engine can hold — about ` +
      `${fmtInt(state.rowLimit)} rows for ${state.sourceTable.columns.length} columns at ` +
      `${configuredPopulations()} populations (fixed 128 MB heap). Fitting a ` +
      `${fmtInt(fitted)}-row sample instead; adjust the count below, or use the R or ` +
      `Python package for the full table.`;
    el.hidden = false;
  } else if (n > ROW_WARN_THRESHOLD) {
    el.classList.add("warn");
    el.textContent =
      `${fmtInt(n)} rows is a lot for the browser: the engine is single-threaded here and a ` +
      `default-budget run evaluates every row ~2.8M times (10,000 rows takes on the order of ` +
      `ten minutes). Sample the rows below, enable Batching in Search settings, or set a ` +
      `Timeout — or run the R or Python package.`;
    el.hidden = false;
  } else {
    el.hidden = true;
    el.textContent = "";
  }
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
  $("settings-summary").textContent =
    `${$("generations").value} generations · seed ${$("seed").value}` +
    ($("batching").checked ? " · batching" : "") +
    (settingsModified() ? " · modified" : "");
  markModifiedFields();
}
function resetDefaults() {
  Object.keys(DEFAULTS).forEach((id) => { $(id).value = String(DEFAULTS[id].value); });
  updateSettingsSummary();
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

// Editing is transactional: values are snapshotted on open so every dismissal path restores
// them, and only Apply keeps them. Without this a mistyped constant could only be undone by
// resetting all 17 fields.
function openSettings() {
  const snapshot = {};
  Object.keys(DEFAULTS).forEach((id) => { snapshot[id] = $(id).value; });
  state.settingsSnapshot = snapshot;
  updateSettingsSummary();
  $("settings-dialog").showModal();
}
function closeSettings(keep) {
  const snapshot = state.settingsSnapshot;
  if (!keep && snapshot) {
    Object.keys(snapshot).forEach((id) => { $(id).value = snapshot[id]; });
  }
  state.settingsSnapshot = null;
  updateSettingsSummary();
  syncRowPolicyIfPopulationsChanged(); // Apply and every dismissal path can land on a new value
  $("settings-dialog").close();
}

// --- Run --------------------------------------------------------------------------
function run() {
  if (!state.table) return;
  state.targetIndex = parseInt($("target-select").value, 10);
  state.featureIndices = currentFeatureIndices();
  if (state.featureIndices.length === 0) {
    setStatus("Select at least one feature.");
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
  state.config = config;

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
  setStatus("stopped");
}

function onResult(result, elapsed) {
  finishRun();
  state.result = result;
  setStatus(`done in ${elapsed.toFixed(2)} s`);
  renderResult();
}

function onError(message) {
  // An OOM abort leaves the module dead but the worker alive, holding its 128 MB heap until
  // the next run replaces it. Drop it now; run() creates a fresh worker anyway.
  if (state.worker) { state.worker.terminate(); state.worker = null; }
  finishRun();
  setStatus("error: " + message);
}

function finishRun() {
  stopTimer();
  document.body.classList.remove("running", "determinate");
  $("pareto-card").classList.remove("live"); // Stop/error/result all end the live state
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
    logLoss: $("logloss").checked,
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
    let line = `running… ${secs.toFixed(1)} s`;
    if (state.totalEpochs > 0) line += ` · epoch ${state.epoch}/${state.totalEpochs}`;
    const eta = estimateRemaining();
    if (eta != null) line += ` · ≤ ${formatDuration(eta)} left`;
    setStatus(line);
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
}

// Re-pick the recommended equation for the current model_selection, in place (no re-run).
function applyModelSelection() {
  if (!state.result) return;
  const front = state.result.pareto_front;
  state.result.best_index = selectBestIndex(front, $("model_selection").value);
  document.querySelectorAll("#pareto-table tbody tr").forEach((tr) => {
    tr.classList.toggle("recommended", parseInt(tr.dataset.index, 10) === state.result.best_index);
  });
  selectEquation(state.result.best_index);
}

// The `complexity` column is the node count of the RAW tree the search archived (the hall of
// fame keeps one member per raw complexity), while the equation shown is its display-simplified
// form (docs/52). Two front members can therefore differ in complexity yet print the identical
// expression. Rendering both counts ("10 → 7") explains that instead of leaving what looks like
// a contradiction between the two columns.
function fmtComplexity(front, i) {
  const raw = front.complexity[i];
  const simplified = front.complexity_simplified ? front.complexity_simplified[i] : null;
  return simplified == null || simplified === raw ? String(raw) : `${raw} → ${simplified}`;
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
    logLoss: $("logloss").checked,
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
  eqEl.textContent = front.expression_simplified ? front.expression_simplified[i] : front.expression[i];
  eqEl.title = front.expression[i];
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
  const oneVar = state.X[0].length === 1;
  $("fit-title").textContent = oneVar
    ? `Fit: ${state.targetName} vs ${state.featureNames[0]}`
    : "Predicted vs actual";
  const idx = strideIndices(state.X.length, DISPLAY_POINT_CAP);
  const Xd = idx ? idx.map((k) => state.X[k]) : state.X;
  const yd = idx ? idx.map((k) => state.y[k]) : state.y;
  $("fit-note").textContent = idx
    ? `Showing ${fmtInt(idx.length)} of ${fmtInt(state.X.length)} points.` : "";
  try {
    const yhat = predict(front.expression[i], Xd);
    drawPrediction($("pred-canvas"), Xd, yd, yhat, {
      xLabel: state.featureNames[0],
      yLabel: state.targetName,
    });
  } catch (e) {
    destroyPlotsPred();
  }
}

function renderEvalAccounting(res) {
  const c = res.eval_counts || {};
  $("eval-accounting").textContent =
    `evaluations: ${fmtInt(res.n_evals)} (forward ${fmtInt(c.forward)}, LM residual ${fmtInt(c.lm_resid)})` +
    (c.cache_hits ? `, cache hits ${fmtInt(c.cache_hits)}` : "");
}

function destroyPlotsPred() {
  // Only the prediction chart needs clearing on eval failure; keep the Pareto chart.
  const c = $("pred-canvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
}

// --- Export wiring ----------------------------------------------------------------
function wireExport() {
  $("copy-expr").addEventListener("click", () => {
    if (state.selectedIndex != null)
      copyText(state.result.pareto_front.expression[state.selectedIndex]);
  });
  $("copy-latex").addEventListener("click", () => {
    if (state.selectedIndex != null) {
      const front = state.result.pareto_front;
      // Copy what is displayed: the simplified LaTeX (display-only, no round-trip duty).
      copyText((front.latex_simplified || front.latex)[state.selectedIndex]);
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
function setStatus(s) { $("status").textContent = s; }
function metric(k, v) { return `<span class="k">${k}</span><span>${v}</span>`; }
function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

// --- Event wiring -----------------------------------------------------------------
function init() {
  updateThemeToggleIcon();
  $("theme-toggle").addEventListener("click", toggleTheme);
  buildOperatorChecks();
  buildMacroPresets();
  buildExamples();
  initSplitters();

  $("add-macro").addEventListener("click", () => addMacroRow());

  $("run-btn").addEventListener("click", () => {
    if (document.body.classList.contains("running")) stop();
    else run();
  });
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (state.table && !document.body.classList.contains("running")) run();
    }
  });
  $("target-select").addEventListener("change", (e) => {
    state.targetIndex = parseInt(e.target.value, 10);
    renderFeatureList();
  });
  $("logloss").addEventListener("change", () => { if (state.result) drawParetoChart(); });
  $("model_selection").addEventListener("change", applyModelSelection);

  $("file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
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
    const text = e.clipboardData && e.clipboardData.getData("text");
    if (!text || !text.trim()) return;
    e.preventDefault();
    loadTextAsTable(text);
  });

  // Empty-state shortcut: one click loads the first example dataset and runs it.
  $("placeholder-run-example").addEventListener("click", () => {
    loadTable(EXAMPLES[0].make());
    run();
  });

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
  $("n_populations").addEventListener("change", syncRowPolicyIfPopulationsChanged);
  $("reset-defaults").addEventListener("click", resetDefaults);
  annotateSettingsFields();
  updateSettingsSummary();

  // The summary line is the settings trigger, matching the data summary above it; "edit" is
  // the same action spelled out. Every dismissal path (Cancel, ×, Esc, backdrop) discards.
  $("settings-summary").addEventListener("click", openSettings);
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
}

init();
