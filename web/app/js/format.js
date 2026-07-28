// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// The single number formatter shared by the metrics chips, the Pareto table and the chart
// tooltips, so the same value never displays with two different precisions in the UI.

export function fmt(v) {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
  return String(Math.round(v * 1e6) / 1e6);
}

export function fmtInt(v) {
  return v == null ? "0" : Math.round(v).toLocaleString();
}

// Compact axis-tick formatter: switches to exponential notation before a label would
// need a long run of decimal places (tiny losses on the Pareto y-axis), with a short
// mantissa ("2e-5", "1.5e-5") since ticks only need order-of-magnitude readability.
export function fmtTick(v) {
  if (typeof v !== "number" || Number.isNaN(v)) return "";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(1).replace(".0e", "e");
  return String(Math.round(v * 1e4) / 1e4);
}

// The `complexity` column is the node count of the RAW tree the search archived (the hall of
// fame keeps one member per raw complexity), while the equation shown is its
// display-simplified form (docs/52). Two front members can therefore differ in complexity yet
// print the identical expression. Rendering both counts ("10 → 7") explains that instead of
// leaving what looks like a contradiction between the two columns.
//
// It lives here, beside the number formatters, because the on-screen table and the printed
// report (report.js) must not be able to count the same equation two different ways.
export function fmtComplexity(front, i) {
  const raw = front.complexity[i];
  const simplified = front.complexity_simplified ? front.complexity_simplified[i] : null;
  return simplified == null || simplified === raw ? String(raw) : `${raw} → ${simplified}`;
}
