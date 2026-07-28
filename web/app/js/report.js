// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// The printable report (docs/64): the whole run as one document — the answer, the evidence
// for it, and the settings that produced it. Rendered into a hidden container that the print
// stylesheet swaps in for the app, so "Save as PDF" in the browser's own print dialog is the
// only PDF writer involved.
//
// This module takes a plain context object and touches no application state and no element
// id of its own (main.js assembles the context, exactly as it does for export.js). The
// report is therefore reorderable without going near the app, and the app's layout — a
// scroll-capped table, an ellipsised equation column, a canvas sized to the results column —
// cannot leak onto paper.
//
// Everything user-supplied (column names, expressions, macro bodies) goes in through
// textContent, never innerHTML.

import { fmt, fmtInt, fmtComplexity } from "./format.js";
import { renderInto } from "./latex.js";

// Minimal element builder: tag, class, text. Children are appended by the caller.
function h(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text != null) el.textContent = text;
  return el;
}

function section(parent, className, heading) {
  const el = h("section", `report-block${className ? ` ${className}` : ""}`);
  if (heading) el.appendChild(h("h2", null, heading));
  parent.appendChild(el);
  return el;
}

// One "key value" pair on a wrapping line — the same reading as the hero card's metric chips.
function metrics(pairs) {
  const row = h("div", "report-metrics");
  pairs.forEach(([k, v]) => {
    const item = h("span", "report-metric");
    item.appendChild(h("span", "k", k));
    item.appendChild(h("span", "v", v));
    row.appendChild(item);
  });
  return row;
}

function figure(src, caption) {
  const fig = h("figure", "report-figure");
  if (src) {
    const img = document.createElement("img");
    img.src = src;
    img.alt = caption;
    fig.appendChild(img);
  } else {
    // chartImage() returned null: say which figure is missing rather than leave a gap the
    // reader has to interpret.
    fig.appendChild(h("div", "report-figure-missing", "(chart could not be rendered)"));
  }
  fig.appendChild(h("figcaption", null, caption));
  return fig;
}

function definitions(pairs) {
  const dl = h("dl", "report-defs");
  pairs.forEach(([k, v]) => {
    dl.appendChild(h("dt", null, k));
    dl.appendChild(h("dd", null, v));
  });
  return dl;
}

function codeBlock(text) {
  const pre = h("pre", "report-code");
  pre.appendChild(h("code", null, text));
  return pre;
}

// --- Page 1: the answer -----------------------------------------------------------

function head(root, ctx) {
  const el = h("header", "report-head");
  el.appendChild(h("h1", null, "Symbolic regression report"));
  el.appendChild(h("div", "report-meta",
    `rsymbolic2 ${ctx.version} · web GUI (WebAssembly) · generated ${formatStamp(ctx.generatedAt)}`));
  root.appendChild(el);
}

// Local time with the offset spelled out: a report is read somewhere else, later.
function formatStamp(d) {
  const pad = (n) => String(n).padStart(2, "0");
  const offsetMin = -d.getTimezoneOffset();
  const sign = offsetMin < 0 ? "-" : "+";
  const abs = Math.abs(offsetMin);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
         `${pad(d.getHours())}:${pad(d.getMinutes())} UTC${sign}${pad(Math.floor(abs / 60))}:${pad(abs % 60)}`;
}

function dataBlock(root, ctx) {
  const el = section(root, "report-data");
  const { targetName, featureNames, rows, columns } = ctx.data;
  el.appendChild(h("div", "report-dataline",
    `${targetName} = f(${featureNames.join(", ")})`));
  const parts = [`${fmtInt(rows)} rows × ${fmtInt(columns)} columns`];
  if (ctx.source) parts.push(sourceText(ctx.source));
  el.appendChild(h("div", "report-subline", parts.join(" · ")));
  if (ctx.sampling) {
    el.appendChild(h("div", "report-flag",
      `Fitted on a ${fmtInt(ctx.sampling.fitted)}-row sample of ${fmtInt(ctx.sampling.total)} rows ` +
      `(deterministic, seed ${ctx.sampling.seed}).`));
  }
}

function sourceText(source) {
  if (source.kind === "example") return `example dataset "${source.label}"`;
  if (source.kind === "paste") return "pasted table";
  return source.label;
}

// The app's hero card is titled "Best formula" and shows whichever equation is SELECTED —
// on screen that is unambiguous, because clicking a row is what changed it. On paper it is
// not: an equation headed "best" that the table below does not mark ★ reads as a
// contradiction the reader cannot resolve. So the heading follows what was actually printed,
// and a line says where it sits in the front whenever the two differ.
function answerBlock(root, ctx) {
  const { front, selectedIndex: i } = ctx;
  const isRecommended = i === ctx.bestIndex;
  const el = section(root, "report-answer", isRecommended ? "Best formula" : "Selected formula");
  const eq = h("div", "report-eq");
  renderInto(eq, front.latex_simplified ? front.latex_simplified[i] : front.latex[i],
             ctx.data.featureNames);
  el.appendChild(eq);
  el.appendChild(codeBlock(
    front.expression_simplified ? front.expression_simplified[i] : front.expression[i]));
  const r2 = front.r2 ? front.r2[i] : null;
  el.appendChild(metrics([
    ["loss", fmt(front.loss[i])],
    ["complexity", fmtComplexity(front, i)],
    ["score", fmt(front.score[i])],
    ["R²", r2 == null ? "—" : fmt(r2)],
  ]));
  if (!isRecommended) {
    el.appendChild(h("p", "report-note",
      `Row ${i} of the Pareto front, chosen by hand. The recommended equation is row ` +
      `${ctx.bestIndex}, marked ★ under All equations.`));
  }
}

function chartsBlock(root, ctx) {
  const el = section(root, "report-charts");
  el.appendChild(figure(ctx.charts.pareto, ctx.charts.paretoCaption));
  el.appendChild(figure(ctx.charts.fit, ctx.charts.fitCaption));
}

// --- Page 2: the evidence ---------------------------------------------------------

// Every front member, no scroll cap and no ellipsis — this is the table the on-screen one
// can only show part of. `report-flow` lets it break across pages; the header repeats
// (thead as a table-header-group) and no row is split.
function tableBlock(root, ctx) {
  const el = section(root, "report-flow report-newpage", "All equations");
  const { front } = ctx;
  const table = h("table", "report-table");
  const thead = h("thead");
  const hr = h("tr");
  ["#", "complexity", "loss", "score", "R²", "equation"].forEach((t) => hr.appendChild(h("th", null, t)));
  thead.appendChild(hr);
  table.appendChild(thead);

  const tbody = h("tbody");
  for (let i = 0; i < front.complexity.length; i++) {
    const tr = h("tr", i === ctx.bestIndex ? "recommended" : null);
    const r2 = front.r2 ? front.r2[i] : null;
    const cells = [
      i === ctx.bestIndex ? `★ ${i}` : String(i),
      fmtComplexity(front, i),
      fmt(front.loss[i]),
      fmt(front.score[i]),
      r2 == null ? "—" : fmt(r2),
    ];
    cells.forEach((c) => tr.appendChild(h("td", null, c)));
    tr.appendChild(h("td", "report-expr",
      front.expression_simplified ? front.expression_simplified[i] : front.expression[i]));
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  el.appendChild(table);
  el.appendChild(h("p", "report-note",
    "Complexity counts the raw searched tree; \"10 → 7\" means the equation printed here is " +
    "the display-simplified form and has fewer nodes."));
}

function treeBlock(root, ctx) {
  if (!ctx.tree) return;
  const el = section(root, null, "Equation tree");
  const box = h("div", "report-tree");
  box.appendChild(ctx.tree);
  el.appendChild(box);
  el.appendChild(h("p", "report-note",
    "Structure of the equation above, as printed (display-simplified)."));
}

// --- Page 3: the reproducibility appendix -----------------------------------------

function operatorBlock(parent, ctx) {
  const cfg = ctx.config;
  parent.appendChild(h("h3", null, "Operator library"));
  parent.appendChild(definitions([
    ["binary", cfg.binary_ops.join(", ") || "(none)"],
    ["unary", cfg.unary_ops.join(", ") || "(none)"],
  ]));
  if (cfg.macro_names && cfg.macro_names.length) {
    const dl = h("dl", "report-defs");
    cfg.macro_names.forEach((name, i) => {
      dl.appendChild(h("dt", null, name));
      dl.appendChild(h("dd", "report-mono", cfg.macro_bodies[i]));
    });
    parent.appendChild(h("div", "report-sublabel", "macros"));
    parent.appendChild(dl);
  }
}

// Every search setting with its PySR-parity default beside it, and any field moved off that
// default marked. Same discipline the settings dialog applies on screen: PySR default parity
// is the project's highest-priority configuration rule, so a divergence must never be
// silent — least of all in the document someone keeps.
function settingsBlock(parent, ctx) {
  parent.appendChild(h("h3", null, "Search settings"));
  const table = h("table", "report-table report-settings");
  const thead = h("thead");
  const hr = h("tr");
  ["setting", "value", "PySR default"].forEach((t) => hr.appendChild(h("th", null, t)));
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = h("tbody");

  let changed = 0;
  ctx.settingRows.forEach(({ name, value, fallback }) => {
    const differs = String(value) !== String(fallback);
    if (differs) changed++;
    const tr = h("tr", differs ? "changed" : null);
    tr.appendChild(h("td", "report-mono", name));
    tr.appendChild(h("td", null, `${value}${differs ? " ✱" : ""}`));
    tr.appendChild(h("td", "report-muted", String(fallback)));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  parent.appendChild(table);
  parent.appendChild(h("p", "report-note", changed
    ? `✱ marks the ${changed} setting${changed === 1 ? "" : "s"} moved off the PySR default.`
    : "Every setting is at its PySR default."));
}

function snippetBlock(parent, ctx) {
  parent.appendChild(h("h3", null, "Reproducing this run"));
  parent.appendChild(h("p", "report-note",
    "The same code the app's copy buttons produce. Both packages run the same C++ engine."));
  parent.appendChild(h("div", "report-sublabel", "Python"));
  parent.appendChild(codeBlock(ctx.snippets.python));
  parent.appendChild(h("div", "report-sublabel", "R"));
  parent.appendChild(codeBlock(ctx.snippets.r));
}

function runBlock(parent, ctx) {
  const run = ctx.run;
  parent.appendChild(h("h3", null, "This run"));
  const counts = ctx.evalCounts || {};
  const evalText = counts.n_evals == null ? "—"
    : `${fmtInt(counts.n_evals)} (forward ${fmtInt(counts.forward || 0)}, ` +
      `LM residual ${fmtInt(counts.lm_resid || 0)}` +
      (counts.cache_hits ? `, cache hits ${fmtInt(counts.cache_hits)}` : "") + ")";
  parent.appendChild(definitions([
    ["started", formatStamp(run.startedAt)],
    ["elapsed", `${run.elapsed.toFixed(2)} s`],
    ["ended", run.stoppedEarly
      ? `early, at epoch ${fmtInt(run.epoch)} of ${fmtInt(run.totalEpochs)} ` +
        "(target loss, timeout or max evals)"
      : `completed the configured budget of ${fmtInt(ctx.config.generations)} generations`],
    ["evaluations", evalText],
  ]));
}

// A number a reader cannot reproduce is worse than no number. The first two apply to every
// report; the rest state a fact about this one.
function notesBlock(parent, ctx) {
  const notes = [
    "This is the browser (WebAssembly) build. It uses the identical search and the same " +
    "PySR-parity defaults as the R and Python packages, but it is not bit-identical to them: " +
    "Emscripten's libm differs from the native one in the last bits, and an evolutionary " +
    "search is sensitive to that. Re-running the code above can return a different but " +
    "equally valid expression.",
    "Equations are printed in their display-simplified form. The complexity column counts " +
    "the raw tree the search archived, and the raw expression is what the code above " +
    "round-trips.",
  ];
  if (ctx.sampling) {
    notes.push(
      `The search was fitted on ${fmtInt(ctx.sampling.fitted)} of the file's ` +
      `${fmtInt(ctx.sampling.total)} rows (deterministic sample, seed ${ctx.sampling.seed}). ` +
      "Losses and R² describe those rows.");
  }
  if (ctx.modelSelection && ctx.modelSelection !== "best") {
    notes.push(
      `The recommended equation was picked with model_selection = "${ctx.modelSelection}". ` +
      "The web GUI defaults to \"score\" (the parsimony elbow) where PySR and the R/Python " +
      "packages default to \"best\"; this chooses which member of the front above is called " +
      "the answer and changes nothing about the search.");
  }
  if (ctx.fitNote) notes.push(`Fit chart: ${ctx.fitNote}`);

  parent.appendChild(h("h3", null, "Notes"));
  const ol = h("ol", "report-notes");
  notes.forEach((t) => ol.appendChild(h("li", null, t)));
  parent.appendChild(ol);
}

function appendixBlock(root, ctx) {
  const el = section(root, "report-flow report-newpage", "Appendix: how this result was produced");
  operatorBlock(el, ctx);
  settingsBlock(el, ctx);
  snippetBlock(el, ctx);
  runBlock(el, ctx);
  notesBlock(el, ctx);
  el.appendChild(h("p", "report-footer",
    `rsymbolic2 ${ctx.version} · Apache-2.0 · https://github.com/ToshihiroIguchi/rsymbolic2`));
}

// --- Entry point ------------------------------------------------------------------

export function buildReport(container, ctx) {
  container.textContent = "";
  const root = h("div", "report");
  head(root, ctx);
  dataBlock(root, ctx);
  answerBlock(root, ctx);
  chartsBlock(root, ctx);
  tableBlock(root, ctx);
  treeBlock(root, ctx);
  appendixBlock(root, ctx);
  container.appendChild(root);
}
