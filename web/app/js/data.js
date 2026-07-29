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
// The formula below inverts the model fitted to ten OK/OOM points in docs/59:
//
//       peak bytes ~= n * (24*p + 80 + 16*n_populations)
//
//   OK   120,000x1 | 100,000x5 | 100,000x10  (31 populations)
//        300,000x1 at 4 and 8 populations
//   OOM  300,000x1 | 200,000x5 | 200,000x10  (31 populations)
//        300,000x1 at 16 populations
//
// *** THIS FORMULA NO LONGER DESCRIBES THE ENGINE. It is kept deliberately. ***
//
// docs/65 changed the shape of the memory usage, not just its size, so do not read the
// terms above as an account of what the engine does today:
//   - `24p + 80` described three live O(n) copies held as row vectors. The bridge now
//     builds ONE column-major copy and moves it into the search (rsymbolic2_wasm.cpp),
//     so the per-row allocation overhead (`+80`) is gone entirely.
//   - `16 * n_populations` described self-LM scratch held per ISLAND. It is now held per
//     OpenMP WORKER, and this build is single-threaded — so the real term is `16 * 1` and
//     the ceiling has no physical dependence on n_populations at all.
//   - The term that actually dominates now is not in the model: the intake transpose
//     holds `Xflat` and the column copy simultaneously (~16*p*n) before swapping Xflat
//     away. That transient peaks BEFORE the search starts.
// Modelled that way the true capacity is roughly 5-19x what this returns (~80 B/row at
// p=5 against the 696 B/row below). That is an estimate from reading the code, NOT a
// measurement — the measured points above are all pre-docs/65.
//
// It is not raised, and raising it is not a pending task (docs/66 §6). The binding
// constraint on browser row count is TIME, not memory, and this ceiling already sits at
// that wall: a default-budget run at the current p=5 limit (~96,000 rows) takes ~5.7 min
// with batching on and ~2 h with it off (docs/59 §1-2). Lifting the ceiling to the
// modelled ~839,000 rows would only let a user start a ~48 min run in a browser tab, so
// the ceiling is doing double duty as a guard against unusable run lengths. Correcting it
// would also cost a fresh WASM OOM sweep, because over-estimating aborts the module rather
// than degrading (docs/59 §3) — real risk, no user benefit.
//
// Known cosmetic consequence, accepted: because the spurious n_populations term is still
// here, raising the population count above the default shrinks the ceiling and can force
// sampling that the engine does not actually require. That costs the user rows in the
// sample, never a failed run, and fixing it means raising a ceiling — which needs the
// sweep above to be safe.
//
// If you do change this: the row limit has to be PREDICTED, never discovered, and the
// budget must stay well inside the smallest measured OOM point. An abort costs the user
// the whole run; a conservative cap only samples a few more rows away.
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
