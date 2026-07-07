// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// A couple of small, quickly-recoverable example datasets, generated inline so the app
// needs no extra fetch (and works when opened without a server). Each returns a parsed
// table shape { columns, rows } compatible with data.js.

function makeTable(columns, rowArrays) {
  return { columns, rows: rowArrays.map((r) => r.map((v) => String(v))) };
}

// y = 2.5 x^2 - 1.3   (single feature; recoverable with add/sub/mul)
function quadratic() {
  const rows = [];
  for (let k = 0; k < 40; k++) {
    const x = -3 + (6 * k) / 39;
    const y = 2.5 * x * x - 1.3;
    rows.push([round(x), round(y)]);
  }
  return makeTable(["x", "y"], rows);
}

// y = x0 * x1 + x0   (two features; recoverable with add/mul)
function product2d() {
  const rows = [];
  let seed = 12345;
  const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let k = 0; k < 60; k++) {
    const x0 = round(-2 + 4 * rnd());
    const x1 = round(-2 + 4 * rnd());
    const y = round(x0 * x1 + x0);
    rows.push([x0, x1, y]);
  }
  return makeTable(["x0", "x1", "y"], rows);
}

function round(v) {
  return Math.round(v * 1e6) / 1e6;
}

export const EXAMPLES = [
  { id: "quadratic", label: "Quadratic  y = 2.5x² − 1.3", make: quadratic },
  { id: "product2d", label: "Two features  y = x0·x1 + x0", make: product2d },
];
