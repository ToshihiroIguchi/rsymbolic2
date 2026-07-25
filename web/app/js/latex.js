// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// KaTeX render helpers. The C++ core (latex.hpp::to_latex) produces the LaTeX with
// `x_{i}` tokens; here we optionally substitute feature names and render into an
// element. KaTeX is vendored same-origin and exposed as the global `katex`.

/* global katex */

// Column names are user data: a CSV header may contain any of LaTeX's special characters, and
// pasting one unescaped produces a broken fragment (KaTeX renders it as red error text) or, for
// braces, a name that swallows the rest of the expression. Escaping every special is also what
// keeps the substitution loop order-independent — a column literally named "x_{2}" can no
// longer be re-substituted by a later iteration.
//
// The names go into \text{} rather than \mathrm{}: the escapes for \, ^ and ~ are text-mode
// macros, and text mode additionally preserves spaces, so "flow rate" keeps its space (math
// mode dropped it). Both render upright, so nothing else about the output changes.
const LATEX_ESCAPES = {
  "\\": "\\textbackslash{}",
  "^": "\\textasciicircum{}",
  "~": "\\textasciitilde{}",
  "{": "\\{",
  "}": "\\}",
  $: "\\$",
  "&": "\\&",
  "#": "\\#",
  "%": "\\%",
  _: "\\_",
};
function escapeLatexText(s) {
  return String(s).replace(/[\\^~{}$&#%_]/g, (c) => LATEX_ESCAPES[c]);
}

// Replace x_{i} tokens with display feature names (LaTeX specials escaped),
// mirroring the R/Python `latex(variable_names=...)` behaviour.
export function substituteNames(latexStr, featureNames) {
  if (!featureNames || !featureNames.length) return latexStr;
  let out = latexStr;
  featureNames.forEach((name, i) => {
    out = out.split(`x_{${i}}`).join(`\\text{${escapeLatexText(name)}}`);
  });
  return out;
}

// Render a LaTeX string into `el`. Falls back to showing the raw string on error so a
// malformed fragment never blanks the panel.
export function renderInto(el, latexStr, featureNames) {
  const src = substituteNames(latexStr, featureNames);
  try {
    katex.render(src, el, { throwOnError: false, displayMode: true });
  } catch (e) {
    el.textContent = latexStr;
  }
}
