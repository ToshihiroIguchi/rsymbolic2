// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Chart.js wrappers for the three result plots: the interactive complexity-vs-loss
// Pareto scatter (click a point to select an equation), and the per-equation
// prediction-vs-data / residual views. Chart.js is vendored same-origin (no CDN) and
// exposed as the global `Chart` by vendor/chart.umd.js.

/* global Chart */

import { fmt } from "./format.js";

let paretoChart = null;
let predChart = null;

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

// Draw / redraw the Pareto front. `front` = {complexity[], loss[], score[]}.
// `bestIndex` marks the recommended member. `logLoss` toggles a log y-axis.
// `onSelect(i)` fires when a point is clicked. `selectedIndex` is highlighted.
export function drawPareto(canvas, front, { bestIndex, logLoss, onSelect, selectedIndex }) {
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
      responsive: true,
      maintainAspectRatio: false,
      onClick: (_evt, elements) => {
        if (elements && elements.length && onSelect) {
          onSelect(points[elements[0].index].i);
        }
      },
      scales: {
        x: themedScale(theme, "Complexity (nodes)", { ticks: { precision: 0 } }),
        y: themedScale(theme, useLog ? "Loss (SSE, log)" : "Loss (SSE)", {
          type: useLog ? "logarithmic" : "linear",
        }),
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const i = points[ctx.dataIndex].i;
              return `complexity ${front.complexity[i]}, loss ${fmt(front.loss[i])}, score ${fmt(
                front.score[i]
              )}`;
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
export function drawPrediction(canvas, X, y, yhat) {
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
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: themedScale(theme, "x0"),
          y: themedScale(theme, "y"),
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
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: themedScale(theme, "actual y"),
          y: themedScale(theme, "predicted y"),
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
