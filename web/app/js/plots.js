// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Chart.js wrappers for the three result plots: the interactive complexity-vs-loss
// Pareto scatter (click a point to select an equation), and the per-equation
// prediction-vs-data / residual views. Chart.js is vendored same-origin (no CDN) and
// exposed as the global `Chart` by vendor/chart.umd.js.

/* global Chart */

let paretoChart = null;
let predChart = null;

// Draw / redraw the Pareto front. `front` = {complexity[], loss[], score[]}.
// `bestIndex` marks the recommended member. `logLoss` toggles a log y-axis.
// `onSelect(i)` fires when a point is clicked. `selectedIndex` is highlighted.
export function drawPareto(canvas, front, { bestIndex, logLoss, onSelect, selectedIndex }) {
  const points = front.complexity.map((c, i) => ({ x: c, y: front.loss[i], i }));
  const pointColors = points.map((_, i) =>
    i === selectedIndex ? "#d62728" : i === bestIndex ? "#2ca02c" : "#1f77b4"
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
          borderColor: "rgba(120,120,120,0.6)",
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
        x: { title: { display: true, text: "Complexity (nodes)" }, ticks: { precision: 0 } },
        y: {
          title: { display: true, text: useLog ? "Loss (SSE, log)" : "Loss (SSE)" },
          type: useLog ? "logarithmic" : "linear",
        },
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
  const ncol = X.length ? X[0].length : 0;

  if (ncol === 1) {
    const scatter = X.map((row, i) => ({ x: row[0], y: y[i] }));
    const curve = X.map((row, i) => ({ x: row[0], y: yhat[i] })).sort((a, b) => a.x - b.x);
    predChart = new Chart(canvas.getContext("2d"), {
      type: "scatter",
      data: {
        datasets: [
          { label: "data", data: scatter, backgroundColor: "#1f77b4", pointRadius: 3 },
          {
            label: "model",
            data: curve,
            showLine: true,
            borderColor: "#d62728",
            backgroundColor: "#d62728",
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "x0" } },
          y: { title: { display: true, text: "y" } },
        },
        plugins: { legend: { display: true } },
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
          { label: "predicted vs actual", data: pts, backgroundColor: "#1f77b4", pointRadius: 3 },
          {
            label: "y = x",
            data: [
              { x: lo, y: lo },
              { x: hi, y: hi },
            ],
            showLine: true,
            borderColor: "#888",
            borderDash: [6, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "actual y" } },
          y: { title: { display: true, text: "predicted y" } },
        },
        plugins: { legend: { display: true } },
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

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(3);
  return v.toPrecision(4).replace(/\.?0+$/, "");
}
