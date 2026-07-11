// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Small example datasets, generated inline so the app needs no extra fetch (and works when
// opened without a server). Each `make()` returns a parsed table shape { columns, rows }
// compatible with data.js. Generation is deterministic (fixed-seed LCG) so an example always
// yields the same table. An optional `ops` field lists non-default operators the target
// formula needs; main.js checks those operator boxes when the example is loaded, so every
// example is searchable and plottable out of the box.

function makeTable(columns, rowArrays) {
  return { columns, rows: rowArrays.map((r) => r.map((v) => String(v))) };
}

// Deterministic uniform-[0,1) generator (same LCG the original product2d example used).
function makeLcg(seed) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

// Standard normal via Box-Muller; consumes two draws from `rnd`.
function gaussian(rnd) {
  const u1 = Math.max(rnd(), 1e-12);
  const u2 = rnd();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function round(v) {
  return Math.round(v * 1e6) / 1e6;
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

// y = exp(-x/2) * sin(3x)   (single feature; needs exp/sin/mul, all default-checked)
function dampedOscillation() {
  const rows = [];
  for (let k = 0; k < 50; k++) {
    const x = (10 * k) / 49;
    const y = Math.exp(-x / 2) * Math.sin(3 * x);
    rows.push([round(x), round(y)]);
  }
  return makeTable(["x", "y"], rows);
}

// y = (x^2 + 1) / (x + 3)   (single feature; x in [-2, 5] keeps clear of the pole at -3)
function rational() {
  const rows = [];
  for (let k = 0; k < 45; k++) {
    const x = -2 + (7 * k) / 44;
    const y = (x * x + 1) / (x + 3);
    rows.push([round(x), round(y)]);
  }
  return makeTable(["x", "y"], rows);
}

// y = x0 * x1 + x0   (two features; recoverable with add/mul)
function product2d() {
  const rows = [];
  const rnd = makeLcg(12345);
  for (let k = 0; k < 60; k++) {
    const x0 = round(-2 + 4 * rnd());
    const x1 = round(-2 + 4 * rnd());
    const y = round(x0 * x1 + x0);
    rows.push([x0, x1, y]);
  }
  return makeTable(["x0", "x1", "y"], rows);
}

// y = sin(x0) + x1^2   (two features; sin/mul default-checked)
function trig2d() {
  const rows = [];
  const rnd = makeLcg(24680);
  for (let k = 0; k < 60; k++) {
    const x0 = round(-3 + 6 * rnd());
    const x1 = round(-3 + 6 * rnd());
    const y = round(Math.sin(x0) + x1 * x1);
    rows.push([x0, x1, y]);
  }
  return makeTable(["x0", "x1", "y"], rows);
}

// y = 3 x0 - 2 x1 + noise   (two features; Gaussian noise, sigma ~2% of the y range)
function noisyLinear() {
  const rows = [];
  const rnd = makeLcg(42);
  const sigma = 0.4; // y spans roughly [-20, 20] over x0,x1 in [-4,4]
  for (let k = 0; k < 90; k++) {
    const x0 = round(-4 + 8 * rnd());
    const x1 = round(-4 + 8 * rnd());
    const y = round(3 * x0 - 2 * x1 + sigma * gaussian(rnd));
    rows.push([x0, x1, y]);
  }
  return makeTable(["x0", "x1", "y"], rows);
}

// y = x0^2 - x1^2 + x0*x1   (two features, 200 rows; a larger table for the preview modal)
function largeQuadratic2d() {
  const rows = [];
  const rnd = makeLcg(13579);
  for (let k = 0; k < 200; k++) {
    const x0 = round(-4 + 8 * rnd());
    const x1 = round(-4 + 8 * rnd());
    const y = round(x0 * x0 - x1 * x1 + x0 * x1);
    rows.push([x0, x1, y]);
  }
  return makeTable(["x0", "x1", "y"], rows);
}

// y = x0 * x1 / x2^2   (three features; x2 in [1, 5] keeps clear of division by zero)
function gravity3d() {
  const rows = [];
  const rnd = makeLcg(97531);
  for (let k = 0; k < 80; k++) {
    const x0 = round(1 + 9 * rnd());
    const x1 = round(1 + 9 * rnd());
    const x2 = round(1 + 4 * rnd());
    const y = round((x0 * x1) / (x2 * x2));
    rows.push([x0, x1, x2, y]);
  }
  return makeTable(["x0", "x1", "x2", "y"], rows);
}

export const EXAMPLES = [
  { id: "quadratic", label: "Quadratic  y = 2.5x² − 1.3  (1 var)", make: quadratic },
  {
    id: "damped_oscillation",
    label: "Damped oscillation  y = e^(−x/2)·sin(3x)  (1 var)",
    make: dampedOscillation,
  },
  {
    id: "rational",
    label: "Rational  y = (x² + 1)/(x + 3)  (1 var)",
    make: rational,
    ops: { binary: ["div"] },
  },
  { id: "product2d", label: "Product  y = x0·x1 + x0  (2 vars)", make: product2d },
  { id: "trig2d", label: "Trig  y = sin(x0) + x1²  (2 vars)", make: trig2d },
  {
    id: "noisy_linear",
    label: "Noisy linear  y = 3x0 − 2x1 + ε  (2 vars)",
    make: noisyLinear,
  },
  {
    id: "large_quadratic2d",
    label: "Large  y = x0² − x1² + x0·x1  (2 vars, 200 rows)",
    make: largeQuadratic2d,
  },
  {
    id: "gravity3d",
    label: "Gravity-like  y = x0·x1/x2²  (3 vars)",
    make: gravity3d,
    ops: { binary: ["div"] },
  },
];
