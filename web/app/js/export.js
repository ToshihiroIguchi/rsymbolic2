// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Export helpers: copy an equation as LaTeX / an equivalent Python or R call, and
// download the Pareto front as CSV. The Python/R snippets reproduce the exact call that
// would fit the same model in those packages (same shared engine, PySR-parity defaults).

export function copyText(text) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    return navigator.clipboard.writeText(text);
  }
  // Fallback for non-secure contexts.
  const ta = document.createElement("textarea");
  ta.value = text;
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  document.body.removeChild(ta);
  return Promise.resolve();
}

function opsLiteral(list) {
  return "[" + list.map((s) => `"${s}"`).join(", ") + "]";
}

const q = (s) => `"${String(s).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;

// Macro operators (docs/57) are opt-in and change the search, so a snippet that omitted them
// would not reproduce the run. Emitted only when the user defined some; `entry` renders one
// line in the host language's mapping literal.
function macroLines(cfg, open, entry, close) {
  const names = cfg.macro_names || [];
  if (!names.length) return [];
  const items = names.map((n, i) => entry(n, cfg.macro_bodies[i])).join(", ");
  return [`${open}${items}${close}`];
}

// Opt-in flags that change the search (or, for eval_cache, the work done to reach the same
// answer) and are OFF by default. A snippet that dropped them would not reproduce the run,
// so each is emitted only when it differs from the shipped default. The argument names are
// identical in R, Python and the WASM bridge, so one table serves both snippets.
function optInLines(cfg, fmtArg) {
  const out = [];
  if (cfg.batching) {
    out.push(fmtArg("batching", true));
    out.push(fmtArg("batch_size", cfg.batch_size));
  }
  if (cfg.linear_scaling) out.push(fmtArg("linear_scaling", true));
  if (cfg.eval_cache && !cfg.batching) out.push(fmtArg("eval_cache", true));
  return out;
}

// A note when the GUI fitted fewer rows than the file holds — without it the snippet reads
// as "run this on your data" while the equation above it came from a sample.
// `sampling` = { fitted, total, seed } or null.
function samplingNote(sampling, comment) {
  if (!sampling) return [];
  return [`${comment} fitted on a ${sampling.fitted}-row sample of ${sampling.total} rows ` +
          `(deterministic, seed ${sampling.seed}); X and y below are the full data.`];
}

// A Python snippet reproducing the run (unary/binary ops + the non-default numeric knobs
// most users touch; the rest stay at PySR-parity defaults).
export function pythonCall(cfg, sampling = null) {
  return [
    ...samplingNote(sampling, "#"),
    "from rsymbolic2 import symbolic_regression",
    "res = symbolic_regression(",
    "    X, y,",
    `    unary_ops=${opsLiteral(cfg.unary_ops)},`,
    `    binary_ops=${opsLiteral(cfg.binary_ops)},`,
    ...macroLines(cfg, "    macro_ops={", (n, b) => `${q(n)}: ${q(b)}`, "},"),
    `    population_size=${cfg.population_size}, n_populations=${cfg.n_populations},`,
    `    generations=${cfg.generations}, max_nodes=${cfg.max_nodes}, seed=${cfg.seed},`,
    ...optInLines(cfg, (k, v) => `    ${k}=${v === true ? "True" : v},`),
    ")",
    "print(res)",
  ].join("\n");
}

// An R snippet reproducing the run.
export function rCall(cfg, sampling = null) {
  const rvec = (list) => "c(" + list.map((s) => `"${s}"`).join(", ") + ")";
  // The last argument line must not carry a trailing comma, so the opt-ins (when present)
  // take over that duty from the seed line.
  const optIns = optInLines(cfg, (k, v) => `  ${k} = ${v === true ? "TRUE" : v}`);
  const fixed = `  generations = ${cfg.generations}, max_nodes = ${cfg.max_nodes}, ` +
                `seed = ${cfg.seed}`;
  return [
    ...samplingNote(sampling, "#"),
    "library(rsymbolic2)",
    "res <- symbolic_regression(",
    "  X, y,",
    `  unary_ops = ${rvec(cfg.unary_ops)},`,
    `  binary_ops = ${rvec(cfg.binary_ops)},`,
    ...macroLines(cfg, "  macro_ops = c(", (n, b) => `${n} = ${q(b)}`, "),"),
    `  population_size = ${cfg.population_size}, n_populations = ${cfg.n_populations},`,
    optIns.length ? `${fixed},` : fixed,
    ...optIns.map((line, i) => (i === optIns.length - 1 ? line : `${line},`)),
    ")",
    "print(res)",
  ].join("\n");
}

// CSV of the Pareto front. `expression` is the evaluatable round-trip string;
// `expression_simplified` is the display-only companion (docs/52), and
// `complexity_simplified` is its node count — it can be smaller than `complexity`, which
// counts the raw searched tree.
export function paretoCsv(front) {
  const rows = ["complexity,complexity_simplified,loss,score,expression,expression_simplified"];
  for (let i = 0; i < front.complexity.length; i++) {
    const expr = String(front.expression[i]).replace(/"/g, '""');
    const simplified = String(
      front.expression_simplified ? front.expression_simplified[i] : front.expression[i]
    ).replace(/"/g, '""');
    const cxSimplified = front.complexity_simplified
      ? front.complexity_simplified[i] : front.complexity[i];
    rows.push(`${front.complexity[i]},${cxSimplified},${front.loss[i]},` +
              `${front.score[i]},"${expr}","${simplified}"`);
  }
  return rows.join("\n");
}

export function downloadText(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
