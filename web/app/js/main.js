// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// UI wiring and application state for the rsymbolic2 web GUI. Ties together data intake
// (data.js), the WASM search run (worker.js), the result plots (plots.js), equation
// rendering (latex.js) and export (export.js). No framework — plain DOM + ES modules.

import { parseTable, numericColumns, toMatrix } from "./data.js";
import { EXAMPLES } from "./examples.js";
import { drawPareto, drawPrediction, destroyPlots } from "./plots.js";
import { renderInto } from "./latex.js";
import { predict } from "./predict.js";
import {
  copyText, pythonCall, rCall, paretoCsv, downloadText,
} from "./export.js";

// Canonical unary order sent to the engine. The core picks operators by a random index into
// this list (mutation.cpp: space.unary_ops[op(rng)]), so the list ORDER is part of the search
// trajectory for a fixed seed. The checkboxes are grouped only for readability (UNARY_GROUPS);
// checkedOps("un") re-emits in this canonical order so the visual grouping never changes the
// search. Keep this array's order stable.
const UNARY = ["neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"];
const UNARY_GROUPS = [
  { label: "Trigonometric", ops: ["sin", "cos", "tanh"] },
  { label: "Exp / Log", ops: ["exp", "log"] },
  { label: "Power / other", ops: ["sqrt", "square", "abs", "neg"] },
];
const UNARY_DEFAULT = new Set(["neg", "exp", "log", "sin", "cos"]);
const BINARY = ["add", "sub", "mul", "div", "pow"];
const BINARY_DEFAULT = new Set(["add", "sub", "mul"]);

const PRESETS = {
  quick: { generations: 200, n_populations: 6, population_size: 27 },
  balanced: { generations: 600, n_populations: 15, population_size: 27 },
  full: { generations: 2800, n_populations: 31, population_size: 27 },
};

const $ = (id) => document.getElementById(id);

const state = {
  table: null, // { columns, rows }
  numeric: [], // bool[] per column
  targetIndex: -1,
  featureIndices: [],
  result: null, // last WASM result
  config: null, // last config used
  X: null,
  y: null,
  featureNames: null,
  selectedIndex: null,
  worker: null,
  timer: null,
  t0: 0,
};

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

function buildExamples() {
  const sel = $("example-select");
  EXAMPLES.forEach((ex) => {
    const o = document.createElement("option");
    o.value = ex.id;
    o.textContent = ex.label;
    sel.appendChild(o);
  });
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  $("generations").value = p.generations;
  $("n_populations").value = p.n_populations;
  $("population_size").value = p.population_size;
}

// --- Data intake ------------------------------------------------------------------
function loadTable(table) {
  state.table = table;
  state.numeric = numericColumns(table);
  renderDataSummary();
  renderPreview();
  buildTargetAndFeatures();
  $("run-btn").disabled = false;
}

function renderDataSummary() {
  const { columns, rows } = state.table;
  $("data-summary").textContent = `${rows.length} rows × ${columns.length} columns`;
}

function renderPreview() {
  const { columns, rows } = state.table;
  const max = Math.min(rows.length, 6);
  let html = "<table><thead><tr>" + columns.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr></thead><tbody>";
  for (let i = 0; i < max; i++) {
    html += "<tr>" + rows[i].map((v) => `<td>${esc(v)}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";
  $("data-preview").innerHTML = html;
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
// Read a numeric input, falling back to `dflt` if the field is empty or non-numeric so a
// cleared box never sends NaN to the engine (an out-of-range max_nodes/seed would degrade
// the search — e.g. max_nodes=NaN collapses every candidate to a single constant).
function readConfig() {
  const numById = (id, dflt) => {
    const v = parseFloat($(id).value);
    return Number.isFinite(v) ? v : dflt;
  };
  const intById = (id, dflt) => {
    const v = parseInt($(id).value, 10);
    return Number.isFinite(v) ? v : dflt;
  };
  return {
    unary_ops: checkedOps("un"),
    binary_ops: checkedOps("bin"),
    generations: intById("generations", 600),
    n_populations: intById("n_populations", 15),
    population_size: intById("population_size", 27),
    max_nodes: intById("max_nodes", 30),
    max_depth: intById("max_depth", 30),
    seed: intById("seed", 1),
    timeout_seconds: numById("timeout_seconds", 0),
    tournament_size: intById("tournament_size", 15),
    tournament_selection_p: numById("tournament_selection_p", 0.982),
    parsimony: numById("parsimony", 0),
    adaptive_parsimony_scaling: numById("adaptive_parsimony_scaling", 1040),
    optimize_probability: numById("optimize_probability", 0.14),
    crossover_probability: numById("crossover_probability", 0.0259),
    fraction_replaced_hof: numById("fraction_replaced_hof", 0.0614),
    target_loss: numById("target_loss", 1e-10),
    max_evals: numById("max_evals", 0),
    warmup_maxsize_by: numById("warmup_maxsize_by", 0),
    model_selection: $("model_selection").value,
    linear_scaling: $("linear_scaling").checked,
    eval_cache: $("eval_cache").checked,
  };
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
  const { X, y } = toMatrix(state.table, state.targetIndex, state.featureIndices);
  state.X = X;
  state.y = y;
  state.featureNames = state.featureIndices.map((j) => state.table.columns[j]);
  const config = readConfig();
  state.config = config;

  $("run-btn").disabled = true;
  $("stop-btn").disabled = false;
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
  };
  state.worker.onerror = (e) => onError(e.message || "worker error");
  state.worker.postMessage({ type: "run", X, y, options: config });
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
  finishRun();
  setStatus("error: " + message);
}

function finishRun() {
  stopTimer();
  $("run-btn").disabled = false;
  $("stop-btn").disabled = true;
}

function startTimer() {
  state.t0 = performance.now();
  state.timer = setInterval(() => {
    setStatus(`running… ${((performance.now() - state.t0) / 1000).toFixed(1)} s`);
  }, 200);
}
function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
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

function renderTable(res, front) {
  const tbody = $("pareto-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (let i = 0; i < front.complexity.length; i++) {
    const tr = document.createElement("tr");
    tr.dataset.index = String(i);
    if (i === res.best_index) tr.classList.add("recommended");
    tr.innerHTML =
      `<td>${i}</td><td>${front.complexity[i]}</td><td>${fmt(front.loss[i])}</td>` +
      `<td>${fmt(front.score[i])}</td><td>${front.r2[i] == null ? "—" : fmt(front.r2[i])}</td>` +
      `<td>${esc(front.expression[i])}</td>`;
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

  // Equation detail.
  renderInto($("eq-latex"), front.latex[i], state.featureNames);
  $("eq-string").textContent = front.expression[i];
  const r2 = front.r2 ? front.r2[i] : null;
  $("eq-metrics").innerHTML =
    metric("loss", fmt(front.loss[i])) +
    metric("complexity", front.complexity[i]) +
    metric("score", fmt(front.score[i])) +
    metric("R²", r2 == null ? "—" : fmt(r2));

  // Prediction plot.
  try {
    const yhat = predict(front.expression[i], state.X);
    drawPrediction($("pred-canvas"), state.X, state.y, yhat);
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
    if (state.selectedIndex != null)
      copyText(state.result.pareto_front.latex[state.selectedIndex]);
  });
  $("copy-python").addEventListener("click", () => {
    if (state.config) copyText(pythonCall(state.config));
  });
  $("copy-r").addEventListener("click", () => {
    if (state.config) copyText(rCall(state.config));
  });
  $("download-csv").addEventListener("click", () => {
    if (state.result) downloadText("rsymbolic2_pareto.csv", paretoCsv(state.result.pareto_front), "text/csv");
  });
}

// --- Helpers ----------------------------------------------------------------------
function setStatus(s) { $("status").textContent = s; }
function metric(k, v) { return `<span class="k">${k}</span><span>${v}</span>`; }
function esc(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}
function fmt(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
  return String(Math.round(v * 1e6) / 1e6);
}
function fmtInt(v) { return v == null ? "0" : Math.round(v).toLocaleString(); }

// --- Event wiring -----------------------------------------------------------------
function init() {
  buildOperatorChecks();
  buildExamples();
  applyPreset("balanced");

  $("preset-select").addEventListener("change", (e) => applyPreset(e.target.value));
  $("run-btn").addEventListener("click", run);
  $("stop-btn").addEventListener("click", stop);
  $("target-select").addEventListener("change", (e) => {
    state.targetIndex = parseInt(e.target.value, 10);
    renderFeatureList();
  });
  $("logloss").addEventListener("change", () => { if (state.result) drawParetoChart(); });
  $("model_selection").addEventListener("change", applyModelSelection);

  $("file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try { loadTable(parseTable(reader.result)); }
      catch (err) { setStatus("parse error: " + err.message); }
    };
    reader.readAsText(file);
  });
  $("example-select").addEventListener("change", (e) => {
    const ex = EXAMPLES.find((x) => x.id === e.target.value);
    if (ex) loadTable(ex.make());
  });
  $("paste-load").addEventListener("click", () => {
    const text = $("paste-area").value;
    if (!text.trim()) return;
    try { loadTable(parseTable(text)); }
    catch (err) { setStatus("parse error: " + err.message); }
  });

  wireExport();
}

init();
