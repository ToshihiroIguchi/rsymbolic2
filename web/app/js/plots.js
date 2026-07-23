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

// Draw the selected equation's fit. For a single feature (ncol === 1) overlay the fitted
// curve on the data scatter (sorted by x). For multiple features, show predicted-vs-actual.
// `X` = array of row arrays, `y` = array, `yhat` = Float64Array/array of predictions.
// `xLabel`/`yLabel` are the real dataset column names for the axis titles.
export function drawPrediction(canvas, X, y, yhat, { xLabel = "x0", yLabel = "y" } = {}) {
  if (predChart) predChart.destroy();
  const theme = themeColors();
  const legend = { display: true, labels: { color: theme.text } };
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
    const lo = Math.min(...y, ...yhat);
    const hi = Math.max(...y, ...yhat);
    predChart = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "predicted vs actual", data: pts, backgroundColor: theme.data, pointRadius: 3 },
          {
            label: "y = x",
            data: [
              { x: lo, y: lo },
              { x: hi, y: hi },
            ],
            showLine: true,
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
