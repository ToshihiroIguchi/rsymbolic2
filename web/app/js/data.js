// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Minimal CSV / whitespace table parsing and numeric-column detection for the data
// intake panel. Kept dependency-free (a small hand-rolled parser) per the project's
// Simplicity priority; if quoted-field edge cases ever matter, swap in PapaParse.

// Parse a delimited-text table. Auto-detects the delimiter (comma, tab, semicolon, or
// runs of whitespace) from the first non-empty line. Returns { columns, rows } where
// columns is an array of header names and rows is an array of string arrays.
export function parseTable(text) {
  const lines = text
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .filter((l) => l.trim().length > 0);
  if (lines.length === 0) throw new Error("No data found.");

  const first = lines[0];
  let splitter;
  if (first.includes(",")) splitter = (l) => splitCsvLine(l, ",");
  else if (first.includes("\t")) splitter = (l) => splitCsvLine(l, "\t");
  else if (first.includes(";")) splitter = (l) => splitCsvLine(l, ";");
  else splitter = (l) => l.trim().split(/\s+/);

  const table = lines.map(splitter);
  const ncol = table[0].length;

  // Header row if the first row is non-numeric in at least one cell; else synthesize.
  const firstRowNumeric = table[0].every((c) => isNumeric(c));
  let columns;
  let dataRows;
  if (firstRowNumeric) {
    columns = Array.from({ length: ncol }, (_, j) => `col${j + 1}`);
    dataRows = table;
  } else {
    columns = table[0].map((c, j) => (c.trim() || `col${j + 1}`));
    dataRows = table.slice(1);
  }
  // Keep only rows with the expected width.
  dataRows = dataRows.filter((r) => r.length === ncol);
  if (dataRows.length === 0) throw new Error("No data rows found.");
  return { columns, rows: dataRows };
}

// A very small CSV line splitter supporting double-quoted fields with embedded
// delimiters and escaped quotes ("").
function splitCsvLine(line, delim) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else inQuotes = false;
      } else cur += c;
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === delim) {
      out.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

function isNumeric(s) {
  if (s === undefined || s === null) return false;
  const t = String(s).trim();
  if (t === "") return false;
  return Number.isFinite(Number(t));
}

// Determine which columns are fully numeric (candidate features/target).
export function numericColumns({ columns, rows }) {
  return columns.map((_, j) => rows.every((r) => isNumeric(r[j])));
}

// Extract X (2-D array of numbers, feature columns in the given order) and y (1-D) from
// a parsed table given a target column index and a list of feature column indices.
export function toMatrix({ rows }, targetIndex, featureIndices) {
  const X = rows.map((r) => featureIndices.map((j) => Number(r[j])));
  const y = rows.map((r) => Number(r[targetIndex]));
  return { X, y };
}

// --- Row-count limits and sampling (docs/59) ---------------------------------------
// The in-browser engine runs on a FIXED 128 MB WASM heap (web/wasm/CMakeLists.txt:
// ALLOW_MEMORY_GROWTH is off), and Emscripten's default ABORTING_MALLOC turns an
// over-allocation into a module-wide abort ("Aborted(OOM)") rather than a recoverable
// error. So the row limit has to be predicted BEFORE the run, not discovered during it.
//
// Peak heap is three O(n) copies of the data plus per-island optimiser scratch:
//   Xflat (n*p*8) + X as vector<vector> (n*p*8 + ~40 B/row, rsymbolic2_wasm.cpp)
//   + the Dataset copy run_evolution() makes (evolutionary_search.cpp: make_shared<Dataset>)
//   + self-LM scratch held by EACH island, which is why the wall moves with n_populations:
//
//       peak bytes ~= n * (24*p + 80 + 16*n_populations)
//
// Measured against the 128 MB heap (docs/59); the island term dominates at the default
// 31 populations:
//   OK   120,000x1 | 100,000x5 | 100,000x10  (31 populations)
//        300,000x1 at 4 and 8 populations
//   OOM  300,000x1 | 200,000x5 | 200,000x10  (31 populations)
//        300,000x1 at 16 populations
// BUDGET_BYTES stays well inside the smallest measured OOM point: an abort costs the user
// the whole run, while a slightly conservative cap only samples a few more rows away.
const BUDGET_BYTES = 64 * 1024 * 1024;

export function maxRowsForBrowser(ncol, nPopulations) {
  const p = Math.max(1, ncol);
  const islands = Math.max(1, nPopulations);
  const perRow = 24 * p + 80 + 16 * islands;
  return Math.max(1000, Math.floor(BUDGET_BYTES / perRow));
}

// xorshift32: a deterministic RNG so the same table always yields the same sample (a
// re-run must fit the same rows). Kept local — the engine's RNG lives in C++ and is not
// reachable from here, and this only picks which rows to keep.
function makeRng(seed) {
  let s = (seed >>> 0) || 0x9e3779b9;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;  s >>>= 0;
    return s / 4294967296;
  };
}

// Knuth's selection sampling (TAOCP 3.4.2 Algorithm S): one pass, exactly `k` of `n`
// indices, every subset equally likely, and the result is already in ascending order so
// the sampled table keeps the file's row order.
export function sampleRowIndices(n, k, seed = 1) {
  if (k >= n) return null; // nothing to do
  const rnd = makeRng(seed);
  const out = new Array(k);
  let picked = 0;
  for (let i = 0; i < n && picked < k; i++) {
    if (rnd() * (n - i) < k - picked) out[picked++] = i;
  }
  return out;
}

// Deterministic stride subset, for DISPLAY only (charts): even coverage, no RNG, and the
// same points on every redraw so a theme toggle or re-select cannot make the scatter
// shimmer. Returns null when everything fits.
export function strideIndices(n, cap) {
  if (n <= cap) return null;
  const out = new Array(cap);
  for (let i = 0; i < cap; i++) out[i] = Math.floor((i * n) / cap);
  return out;
}

// Keep only `k` rows of a parsed table (returns a new {columns, rows}); the columns and
// the row order are unchanged.
export function sampleTable(table, k, seed = 1) {
  const idx = sampleRowIndices(table.rows.length, k, seed);
  if (!idx) return table;
  return { columns: table.columns, rows: idx.map((i) => table.rows[i]) };
}
