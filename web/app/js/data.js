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
