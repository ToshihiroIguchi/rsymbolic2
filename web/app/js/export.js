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

// A Python snippet reproducing the run (unary/binary ops + the non-default numeric knobs
// most users touch; the rest stay at PySR-parity defaults).
export function pythonCall(cfg) {
  return [
    "from rsymbolic2 import symbolic_regression",
    "res = symbolic_regression(",
    "    X, y,",
    `    unary_ops=${opsLiteral(cfg.unary_ops)},`,
    `    binary_ops=${opsLiteral(cfg.binary_ops)},`,
    `    population_size=${cfg.population_size}, n_populations=${cfg.n_populations},`,
    `    generations=${cfg.generations}, max_nodes=${cfg.max_nodes}, seed=${cfg.seed},`,
    ")",
    "print(res)",
  ].join("\n");
}

// An R snippet reproducing the run.
export function rCall(cfg) {
  const rvec = (list) => "c(" + list.map((s) => `"${s}"`).join(", ") + ")";
  return [
    "library(rsymbolic2)",
    "res <- symbolic_regression(",
    "  X, y,",
    `  unary_ops = ${rvec(cfg.unary_ops)},`,
    `  binary_ops = ${rvec(cfg.binary_ops)},`,
    `  population_size = ${cfg.population_size}, n_populations = ${cfg.n_populations},`,
    `  generations = ${cfg.generations}, max_nodes = ${cfg.max_nodes}, seed = ${cfg.seed}`,
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
