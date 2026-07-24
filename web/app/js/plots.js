// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Chart.js wrappers for the three result plots: the interactive complexity-vs-loss
// Pareto scatter (click a point to select an equation), and the per-equation
// prediction-vs-data / residual views. Chart.js is vendored same-origin (no CDN) and
// exposed as the global `Chart` by vendor/chart.umd.js.

/* global Chart */

import { fmt, fmtTick } from "./format.js";

let paretoChart = null;
let predChart = null;

// Options shared by every chart. Charts are destroyed and rebuilt on each redraw (theme
// toggle, equation click, and once per throttled progress snapshot while a search runs), so
// Chart.js's default grow-in animation would replay in full every time — a live Pareto front
// that should converge gradually instead re-inflates from the axis several times a second.
// Static draws are the correct behaviour here.
const BASE_OPTIONS = { responsive: true, maintainAspectRatio: false, animation: false };

// Chart colors come from the active theme's CSS variables, read fresh on every draw. Both
// charts are destroyed and recreated by their draw functions, so a theme toggle only needs
// to trigger a redraw (main.js redrawCharts) — Chart.defaults is never mutated.
function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    grid: v("--chart-grid"),
    text: v("--chart-text"),
    selected: v("--chart-selected"),
    recommended: v("--chart-recommended"),
    data: v("--chart-data"),
    line: v("--chart-line"),
    refline: v("--chart-refline"),
  };
}

// Shared scale styling so axis titles/ticks/gridlines stay legible on both themes
// (Chart.js defaults assume a light background).
function themedScale(theme, title, extra = {}) {
  return {
    ...extra,
    title: { display: true, text: title, color: theme.text },
    ticks: { ...(extra.ticks || {}), color: theme.text },
    grid: { color: theme.grid },
  };
}

// Draw / redraw the Pareto front. `front` = {complexity[], loss[], score[]}. `score`
// (and `r2`) may be null/absent — e.g. a live in-progress snapshot only has
// complexity/loss — in which case the tooltip degrades to those two fields.
// `bestIndex` marks the recommended member. `logLoss` toggles a log y-axis.
// `onSelect(i)` fires when a point is clicked; both `onSelect` and `bestIndex` /
// `selectedIndex` are optional (a live snapshot passes neither: nothing is
// clickable or highlighted yet).
export function drawPareto(canvas, front, { bestIndex, logLoss, onSelect, selectedIndex } = {}) {
  const theme = themeColors();
  const points = front.complexity.map((c, i) => ({ x: c, y: front.loss[i], i }));
  const pointColors = points.map((_, i) =>
    i === selectedIndex ? theme.selected : i === bestIndex ? theme.recommended : theme.data
  );
  const pointRadius = points.map((_, i) =>
    i === selectedIndex ? 7 : i === bestIndex ? 6 : 4
  );

  const useLog = logLoss && front.loss.every((l) => l > 0);

  if (paretoChart) paretoChart.destroy();
  paretoChart = new Chart(canvas.getContext("2d"), {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Pareto front",
          data: points,
          showLine: true,
          borderColor: theme.line,
          backgroundColor: pointColors,
          pointBackgroundColor: pointColors,
          pointRadius,
          pointHoverRadius: 8,
          order: 2,
        },
      ],
    },
    options: {
      ...BASE_OPTIONS,
      onClick: (_evt, elements) => {
        if (elements && elements.length && onSelect) {
          const idx = points[elements[0].index].i;
          // Defer past Chart.js's event pipeline: onSelect redraws (destroys) this very
          // chart, and destroying it synchronously inside its own onClick throws inside
          // chart.umd.js (handleEvent on undefined).
          setTimeout(() => onSelect(idx), 0);
        }
      },
      scales: {
        x: themedScale(theme, "Complexity (nodes)", { ticks: { precision: 0 } }),
        y: themedScale(theme, useLog ? "Loss (SSE, log)" : "Loss (SSE)", {
          type: useLog ? "logarithmic" : "linear",
          // Tiny losses otherwise render as long decimal strings; exponential ticks
          // keep the axis readable (fmtTick handles both the log and linear scale).
          ticks: { callback: fmtTick },
        }),
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const i = points[ctx.dataIndex].i;
              const base = `complexity ${front.complexity[i]}, loss ${fmt(front.loss[i])}`;
              return front.score ? `${base}, score ${fmt(front.score[i])}` : base;
            },
          },
        },
      },
    },
  });
}

// Span of two series, ignoring non-finite entries. Written as a loop on purpose: the
// obvious `Math.min(...y, ...yhat)` spreads every row onto the call stack and throws
// RangeError once the dataset passes ~100k rows (measured), which used to make the fit
// chart silently disappear for large multi-feature data. A prediction can also be
// NaN/Inf (log of a negative, a division by zero on the plotted inputs), which would
// otherwise poison the axis range for every other point.
function finiteRange(a, b) {
  let lo = Infinity;
  let hi = -Infinity;
  const scan = (arr) => {
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i];
      if (!Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  };
  scan(a);
  scan(b);
  if (lo > hi) return { lo: 0, hi: 1 }; // nothing finite to plot
  return { lo, hi };
}

// Draw the selected equation's fit. For a single feature (ncol === 1) overlay the fitted
// curve on the data scatter (sorted by x). For multiple features, show predicted-vs-actual.
// `X` = array of row arrays, `y` = array, `yhat` = Float64Array/array of predictions.
// `xLabel`/`yLabel` are the real dataset column names for the axis titles.
// The caller passes an already down-sampled view (main.js: DISPLAY_POINT_CAP): Chart.js
// allocates one object per point and redraws on every equation click and theme toggle, so
// handing it a full 100k-row dataset would stall the UI each time.
export function drawPrediction(canvas, X, y, yhat, { xLabel = "x0", yLabel = "y" } = {}) {
  if (predChart) predChart.destroy();
  const theme = themeColors();
  // Legend keys mirror the marks they stand for: a dot for the scattered points, a short
  // line (dashed where the dataset is dashed) for the curves. Chart.js's default legend box
  // is a filled 40px rectangle, which misrepresents both; `usePointStyle` swaps in each
  // dataset's own pointStyle instead. `boxHeight` is the only size knob usable here — it
  // sets the dot radius and the line length together. Do NOT add `pointStyleWidth` to
  // lengthen the line keys: a set width makes Chart.js draw the dot as an ellipse.
  const legend = {
    display: true,
    labels: { color: theme.text, usePointStyle: true, boxHeight: 9 },
  };
  const ncol = X.length ? X[0].length : 0;

  if (ncol === 1) {
    const scatter = X.map((row, i) => ({ x: row[0], y: y[i] }));
    const curve = X.map((row, i) => ({ x: row[0], y: yhat[i] })).sort((a, b) => a.x - b.x);
    predChart = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "data", data: scatter, backgroundColor: theme.data, pointRadius: 3 },
          {
            label: "model",
            data: curve,
            showLine: true,
            pointStyle: "line",
            borderColor: theme.selected,
            backgroundColor: theme.selected,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        ...BASE_OPTIONS,
        scales: {
          x: themedScale(theme, xLabel),
          y: themedScale(theme, yLabel),
        },
        plugins: { legend },
      },
    });
  } else {
    const pts = y.map((yi, i) => ({ x: yi, y: yhat[i] }));
    const { lo, hi } = finiteRange(y, yhat);
    predChart = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          // The legend names the two marks, not the chart: the card heading and the axis
          // titles already say "predicted vs actual", and "y = x" would read as an equation
          // between the dataset's own columns when the target column is itself named y.
          { label: "predictions", data: pts, backgroundColor: theme.data, pointRadius: 3 },
          {
            label: "ideal (predicted = actual)",
            data: [
              { x: lo, y: lo },
              { x: hi, y: hi },
            ],
            showLine: true,
            pointStyle: "line",
            borderColor: theme.refline,
            borderDash: [6, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        ...BASE_OPTIONS,
        scales: {
          x: themedScale(theme, `actual ${yLabel}`),
          y: themedScale(theme, `predicted ${yLabel}`),
        },
        plugins: { legend },
      },
    });
  }
}

export function destroyPlots() {
  if (paretoChart) {
    paretoChart.destroy();
    paretoChart = null;
  }
  if (predChart) {
    predChart.destroy();
    predChart = null;
  }
}
