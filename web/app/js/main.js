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
import { fmt, fmtInt } from "./format.js";

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

const $ = (id) => document.getElementById(id);

// The shipped search defaults, PySR-identical (docs/28). Single source for three consumers:
// readConfig()'s fallback when a field is blank, resetDefaults(), and the "modified" marker
// in the Settings summary — so those can never drift apart. Keys are the input element ids;
// `int` marks the fields parsed with parseInt. model_selection is deliberately absent: it is
// a results-side display control (score, the web GUI's sanctioned divergence — CLAUDE.md
// "the web GUI is exempt"), not a search setting, and it lives outside this panel.
const DEFAULTS = {
  generations: { value: 2800, int: true },
  n_populations: { value: 31, int: true },
  population_size: { value: 27, int: true },
  max_nodes: { value: 30, int: true },
  seed: { value: 1, int: true },
  timeout_seconds: { value: 0 },
  tournament_size: { value: 15, int: true },
  tournament_selection_p: { value: 0.982 },
  parsimony: { value: 0 },
  adaptive_parsimony_scaling: { value: 1040 },
  optimize_probability: { value: 0.14 },
  crossover_probability: { value: 0.0259 },
  fraction_replaced_hof: { value: 0.0614 },
  max_depth: { value: 30, int: true },
  target_loss: { value: 1e-10 },
  max_evals: { value: 0 },
  warmup_maxsize_by: { value: 0 },
};

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
  targetName: null,
  selectedIndex: null,
  worker: null,
  timer: null,
  t0: 0,
  lastProgressDraw: 0, // performance.now() of the last live-Pareto redraw (throttling)
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
  state.table = table;
  state.numeric = numericColumns(table);
  renderDataSummary();
  buildTargetAndFeatures();
  // Progressive disclosure: the target/feature controls stay hidden until there is data to
  // model, and loading data collapses the intake block (drop zone + example picker) — the
  // "change" button re-opens it.
  document.body.classList.add("has-data");
  document.body.classList.remove("editing-data");
  $("run-btn").disabled = false;
  $("data-summary").disabled = false;
}

// Shared by all three intake paths (file input, drag & drop, clipboard paste).
function loadTextAsTable(text) {
  try { loadTable(parseTable(text)); }
  catch (err) { setStatus("parse error: " + err.message); }
}
function loadFile(file) {
  const reader = new FileReader();
  reader.onload = () => loadTextAsTable(reader.result);
  reader.readAsText(file);
}

function renderDataSummary() {
  const { columns, rows } = state.table;
  $("data-summary").textContent = `${rows.length} rows × ${columns.length} columns`;
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
  $("preview-note").textContent =
    rows.length > cap ? `Showing first ${cap} of ${rows.length} rows.` : `${rows.length} rows.`;
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
  return {
    unary_ops: checkedOps("un"),
    binary_ops: checkedOps("bin"),
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
    model_selection: $("model_selection").value,
    linear_scaling: $("linear_scaling").checked,
    eval_cache: $("eval_cache").checked,
  };
}

// A collapsed disclosure that says nothing forces a click just to learn the budget, so the
// summary carries the two values users actually change plus a warning when anything has
// drifted off the PySR-parity defaults (CLAUDE.md's highest-priority configuration rule —
// the GUI should never let that happen silently).
function settingsModified() {
  return Object.keys(DEFAULTS).some((id) => {
    const d = DEFAULTS[id];
    const v = d.int ? parseInt($(id).value, 10) : parseFloat($(id).value);
    return !Number.isFinite(v) || v !== d.value;
  });
}
function updateSettingsSummary() {
  $("settings-summary").textContent =
    `— ${$("generations").value} generations · seed ${$("seed").value}` +
    (settingsModified() ? " · modified" : "");
}
function resetDefaults() {
  Object.keys(DEFAULTS).forEach((id) => { $(id).value = String(DEFAULTS[id].value); });
  updateSettingsSummary();
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
  // Snapshot the target name with the matrices: the target <select> can change after the
  // run, but the displayed result must keep the names it was fitted with.
  state.targetName = state.table.columns[state.targetIndex];
  const config = readConfig();
  state.config = config;

  setRunButton(true);
  document.body.classList.add("running"); // shows the header progress bar
  $("pareto-card").classList.remove("live"); // cleared again on the first snapshot's redraw
  state.lastProgressDraw = 0;
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
  document.body.classList.remove("running");
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

  // Prediction plot, labelled with the real column names captured at run time.
  const oneVar = state.X[0].length === 1;
  $("fit-title").textContent = oneVar
    ? `Fit: ${state.targetName} vs ${state.featureNames[0]}`
    : "Predicted vs actual";
  try {
    const yhat = predict(front.expression[i], state.X);
    drawPrediction($("pred-canvas"), state.X, state.y, yhat, {
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
    if (state.config) copyText(pythonCall(state.config));
  });
  $("copy-r").addEventListener("click", () => {
    if (state.config) copyText(rCall(state.config));
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
  buildExamples();
  initSplitters();

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

  // Keep the Settings summary in step with the fields it reports (including the "modified"
  // marker), and offer the one-click way back to the shipped PySR-parity defaults.
  Object.keys(DEFAULTS).forEach((id) => $(id).addEventListener("input", updateSettingsSummary));
  $("reset-defaults").addEventListener("click", resetDefaults);
  updateSettingsSummary();

  wireExport();
}

init();
