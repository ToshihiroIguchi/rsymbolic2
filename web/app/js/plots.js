// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Chart.js wrappers for the three result plots: the interactive complexity-vs-loss
// Pareto scatter (click a point to select an equation), and the per-equation
// prediction-vs-data / residual views. Chart.js is vendored same-origin (no CDN) and
// exposed as the global `Chart` by vendor/chart.umd.js.
//
// The fit and residual views share one canvas (and one `predChart` slot): they are two
// readings of the same selected equation and the card shows one at a time.
//
// Every chart is described exactly once, by a *Config() function that takes a resolved
// palette and returns a Chart.js config. Two consumers call those: the draw*() functions
// below, which own the on-screen canvases, and the *Image() functions, which re-render the
// same chart off-screen for the printable report (docs/64). Keeping one description means a
// chart cannot look like one thing on screen and another on paper.

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
//
// `root` is the element the variables are resolved against, normally <html> (the theme the
// user is looking at). The print path passes its own off-screen host instead, which carries
// .print-palette: a report is going onto white paper whatever the screen theme is, and
// re-declaring the variables on one container is a local change, where swapping
// <html data-theme> would be a global one made to produce a local effect.
function themeColors(root = document.documentElement) {
  const cs = getComputedStyle(root);
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

// Config for the Pareto front. `front` = {complexity[], loss[], score[]}. `score`
// (and `r2`) may be null/absent — e.g. a live in-progress snapshot only has
// complexity/loss — in which case the tooltip degrades to those two fields.
// `bestIndex` marks the recommended member. `logLoss` toggles a log y-axis.
// `onSelect(i)` fires when a point is clicked; both `onSelect` and `bestIndex` /
// `selectedIndex` are optional (a live snapshot passes neither: nothing is
// clickable or highlighted yet, and the printed copy has nothing to click).
function paretoConfig(front, theme, { bestIndex, logLoss, onSelect, selectedIndex } = {}) {
  const points = front.complexity.map((c, i) => ({ x: c, y: front.loss[i], i }));
  const pointColors = points.map((_, i) =>
    i === selectedIndex ? theme.selected : i === bestIndex ? theme.recommended : theme.data
  );
  const pointRadius = points.map((_, i) =>
    i === selectedIndex ? 7 : i === bestIndex ? 6 : 4
  );

  const useLog = logLoss && front.loss.every((l) => l > 0);

  return {
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
  };
}

// Draw / redraw the Pareto front on its on-screen canvas.
export function drawPareto(canvas, front, opts = {}) {
  if (paretoChart) paretoChart.destroy();
  paretoChart = new Chart(canvas.getContext("2d"), paretoConfig(front, themeColors(), opts));
}

// Span of two series, ignoring non-finite entries. Written as a loop on purpose: the
// obvious `Math.min(...y, ...yhat)` spreads every row onto the call stack and throws
// RangeError once the dataset passes ~100k rows (measured), which used to make the fit
// chart silently disappear for large multi-feature data. A prediction can also be
// NaN/Inf (log of a negative, a division by zero on the plotted inputs), which would
// otherwise poison the axis range for every other point.
function finiteRange(a, b = []) {
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

// Legend keys mirror the marks they stand for: a dot for the scattered points, a short
// line (dashed where the dataset is dashed) for the curves. Chart.js's default legend box
// is a filled 40px rectangle, which misrepresents both; `usePointStyle` swaps in each
// dataset's own pointStyle instead. `boxHeight` is the only size knob usable here — it
// sets the dot radius and the line length together. Do NOT add `pointStyleWidth` to
// lengthen the line keys: a set width makes Chart.js draw the dot as an ellipse.
function themedLegend(theme) {
  return { display: true, labels: { color: theme.text, usePointStyle: true, boxHeight: 9 } };
}

// Config for the selected equation's fit. For a single feature (ncol === 1) overlay the
// fitted curve on the data scatter (sorted by x). For multiple features, show
// predicted-vs-actual. `X` = array of row arrays, `y` = array, `yhat` =
// Float64Array/array of predictions. `xLabel`/`yLabel` are the real dataset column names
// for the axis titles.
// The caller passes an already down-sampled view (main.js: DISPLAY_POINT_CAP): Chart.js
// allocates one object per point and redraws on every equation click and theme toggle, so
// handing it a full 100k-row dataset would stall the UI each time.
function predictionConfig(X, y, yhat, theme, { xLabel = "x0", yLabel = "y" } = {}) {
  const legend = themedLegend(theme);
  const ncol = X.length ? X[0].length : 0;

  if (ncol === 1) {
    const scatter = X.map((row, i) => ({ x: row[0], y: y[i] }));
    const curve = X.map((row, i) => ({ x: row[0], y: yhat[i] })).sort((a, b) => a.x - b.x);
    return {
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
    };
  } else {
    const pts = y.map((yi, i) => ({ x: yi, y: yhat[i] }));
    const { lo, hi } = finiteRange(y, yhat);
    return {
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
    };
  }
}

// Draw the selected equation's fit on the shared prediction canvas.
export function drawPrediction(canvas, X, y, yhat, opts = {}) {
  if (predChart) predChart.destroy();
  predChart = new Chart(canvas.getContext("2d"),
                        predictionConfig(X, y, yhat, themeColors(), opts));
}

// Config for the selected equation's residuals (actual − predicted) against the predicted
// value. This is what the fit view cannot show: once the points are dense, a curve overlay
// and a high R² both hide systematic error. Residuals scattered evenly about the dashed zero
// line mean the equation captured the structure; a visible shape (a bend, a fan, a step)
// means it did not, however good the loss looks. Shares the canvas and down-sampling rules
// of drawPrediction — same caller, same selected equation.
function residualConfig(y, yhat, theme, { yLabel = "y" } = {}) {
  const pts = [];
  for (let i = 0; i < y.length; i++) pts.push({ x: yhat[i], y: y[i] - yhat[i] });
  const { lo, hi } = finiteRange(yhat);

  return {
    type: "scatter",
    data: {
      datasets: [
        { label: "residuals", data: pts, backgroundColor: theme.data, pointRadius: 3 },
        {
          label: "zero (perfect fit)",
          data: [
            { x: lo, y: 0 },
            { x: hi, y: 0 },
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
        x: themedScale(theme, `predicted ${yLabel}`),
        // Residuals of a good fit are tiny next to the data, so the axis routinely lands in
        // the 1e-9 range where default ticks render as long decimals — same fmtTick the
        // Pareto loss axis uses.
        y: themedScale(theme, "residual (actual − predicted)", { ticks: { callback: fmtTick } }),
      },
      plugins: { legend: themedLegend(theme) },
    },
  };
}

// Draw the selected equation's residuals on the shared prediction canvas.
export function drawResidual(canvas, y, yhat, opts = {}) {
  if (predChart) predChart.destroy();
  predChart = new Chart(canvas.getContext("2d"), residualConfig(y, yhat, themeColors(), opts));
}

// --- Print-resolution snapshots (docs/64) -----------------------------------------
// The report cannot reuse the on-screen canvases: they are drawn at the screen's device
// pixel ratio (1 on an ordinary display) and sized to the results column, so they print
// blurry. Each chart is re-rendered off-screen instead and embedded as a PNG data URI.
//
// The two knobs do different jobs and must not be confused. The LOGICAL size (CSS px)
// fixes the proportion of text to plot: rendered 760 px wide and printed 88 mm wide, a 12 px
// axis label lands at about 4 pt on paper. So the logical size is deliberately small — close
// to the physical size — and the font is enlarged to match. devicePixelRatio fixes SHARPNESS
// only: Chart.js honours options.devicePixelRatio when it scales the backing store, leaving
// every dimension in logical px, which puts the printed chart near 480 dpi.
const PRINT_CHART = { width: 440, height: 300, scale: 3, fontSize: 15 };

// Render one config off-screen and return its PNG data URI.
//
// The host is positioned off-viewport rather than `display: none`: Chart.js's responsive
// sizing reads the container's computed size, and a `display: none` parent reports zero.
// It carries .print-palette, so themeColors(host) resolves the light chart colors whatever
// the screen theme is.
//
// Chart.defaults.font.size is global, and Chart.js resolves it lazily at draw time — so it
// is raised only for the duration of this synchronous render and restored in `finally`. No
// on-screen chart updates inside that window, and none can: nothing here yields.
function chartImage(makeConfig) {
  const host = document.createElement("div");
  host.className = "print-chart-host print-palette";
  host.style.width = `${PRINT_CHART.width}px`;
  host.style.height = `${PRINT_CHART.height}px`;
  const canvas = document.createElement("canvas");
  host.appendChild(canvas);
  document.body.appendChild(host);

  const previousFontSize = Chart.defaults.font.size;
  Chart.defaults.font.size = PRINT_CHART.fontSize;
  let chart = null;
  try {
    const config = makeConfig(themeColors(host));
    config.options = { ...config.options, devicePixelRatio: PRINT_CHART.scale };
    chart = new Chart(canvas.getContext("2d"), config);
    return chart.toBase64Image();
  } catch (e) {
    // A missing chart costs the report one figure; it must never cost the whole report.
    return null;
  } finally {
    if (chart) chart.destroy();
    Chart.defaults.font.size = previousFontSize;
    host.remove();
  }
}

// The printed copies of the three plots. Nothing in a printed chart is clickable, so
// `onSelect` is never passed on.
export function paretoImage(front, { bestIndex, logLoss, selectedIndex } = {}) {
  return chartImage((theme) => paretoConfig(front, theme, { bestIndex, logLoss, selectedIndex }));
}
export function fitImage(X, y, yhat, opts = {}) {
  return chartImage((theme) => predictionConfig(X, y, yhat, theme, opts));
}
export function residualImage(y, yhat, opts = {}) {
  return chartImage((theme) => residualConfig(y, yhat, theme, opts));
}

export function destroyPlots() {
  if (paretoChart) {
    paretoChart.destroy();
    paretoChart = null;
  }
  destroyPrediction();
}

// Drop the prediction chart alone (the Pareto chart stays). Callers that only want to blank
// the canvas must still come through here: Chart.js keeps its own copy of the data and is
// `responsive`, so a chart merely cleared with clearRect() repaints itself — with the stale
// series — on the next container resize.
export function destroyPrediction() {
  if (predChart) {
    predChart.destroy();
    predChart = null;
  }
}
