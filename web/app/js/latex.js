// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// KaTeX render helpers. The C++ core (latex.hpp::to_latex) produces the LaTeX with
// `x_{i}` tokens; here we optionally substitute feature names and render into an
// element. KaTeX is vendored same-origin and exposed as the global `katex`.

/* global katex */

// Replace x_{i} tokens with display feature names (underscores escaped for LaTeX),
// mirroring the R/Python `latex(variable_names=...)` behaviour.
export function substituteNames(latexStr, featureNames) {
  if (!featureNames || !featureNames.length) return latexStr;
  let out = latexStr;
  featureNames.forEach((name, i) => {
    const safe = String(name).replace(/_/g, "\\_");
    out = out.split(`x_{${i}}`).join(`\\mathrm{${safe}}`);
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
