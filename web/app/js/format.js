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
