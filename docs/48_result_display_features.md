# 48. Result-display features (score, LaTeX, plot, R²)

Status: implemented (2026-07-04).

This document records the design decisions behind the result-display layer added
across the C++ core and both bindings. Display is **not** bound by the PySR
default-parity rule (docs/28), which governs search defaults and behaviour; PySR's
display API (`equations_` score column, `model_selection`, `latex()`) served as the
reference model, with deliberate divergences noted below.

## What was added

| Surface | Addition |
|---|---|
| C++ core | `pareto_scores(front)` (hall_of_fame), `to_latex(tree, precision)` (expression/latex.hpp) |
| Both bridges | `score` and `latex` columns in `pareto_front`; top-level `n_obs`, `sst` |
| R | `summary()`/`as.data.frame()` read the C++ `score`; new `to_latex()` generic + method; `summary()` per-member `r_squared` column + headline `R-squared (recommended)` line |
| Python | `score`/`r_squared`/`latex` in `pareto_front` member dicts and `to_pandas()` (score); `__repr__` score column, `>` marker on the recommended row, >20-row elision (mirrors R `format_pareto_lines`); `get_best(index=)`; `latex(index=, variable_names=)`; `plot()` (matplotlib, the previously declared-but-unused `plot` extra); R² line in `__repr__` |

## D1. Score: computed once in C++, consistent with selection by construction

`pareto_scores()` returns the per-member selection score PySR displays in
`equations_`: `scores[0] = 0.0`, `scores[i] = (log(loss[i-1]) - log(loss[i])) / dc`
with losses floored at `1e-300`, `NaN` for a malformed (non-ascending) step.
`select_best()` was refactored to rank by exactly this vector, so the displayed
score column and the engine's `recommended` pick cannot drift apart. The former
R-side duplicate (`utils.R::pareto_score`) was deleted.

Deliberate divergences from PySR's displayed score:

- **`loss == 0` gives a large finite score (~690/dc), not `inf`.** PySR displays
  `inf` for an exact fit but rsymbolic2's `select_best` has always ranked on the
  floored value; exposing a different number than the engine ranks by would let
  display and selection disagree. The front is strictly loss-decreasing, so at most
  one zero-loss member exists and both conventions pick the same member outside
  sub-1e-300 pathologies.
- **First member scores `0.0`** (PySR convention). R previously displayed `NA`
  there; both languages now agree.

## D2. Round-trip rule: `expression` strings are frozen

Both `predict()` implementations evaluate the stored infix `expression` strings
(R: sandboxed `eval` parse; Python: `eval` with `^`→`**`). Therefore `to_string()`
output and every stored expression remain **byte-for-byte unchanged**; all
readability work went into the parallel, display-only `latex` column. A
precedence-aware minimal-parenthesis *infix* renderer was considered and rejected:
any precedence bug would become a silent wrong prediction (correctness outranks
readability), and LaTeX covers the readability need.

## D3. LaTeX serializer (no sympy)

`rsymbolic/expression/latex.hpp` — header-only, ~150 lines, C++/STL only (the
dependency policy rules out sympy/symengine). Postfix walk with a
`(fragment, precedence)` stack, `Prec ∈ {Add, Mul, Pow, Atom}`:

- `/` → `\frac{}{}` (operands never parenthesized), `*` → `\cdot`,
  `^`/`square` → `base^{e}` with non-atomic bases parenthesized
  (`(e^{x})^{y}`, `(x+1)^{2}`), `sqrt` → `\sqrt{}`, `abs` → `\left|\right|`,
  `exp` → `e^{}`, `log/sin/cos/tanh` → `\log\left( \right)` etc., `neg` → prefix
  `-` (parenthesizing Add-level children only).
- Constants: `%.6g` by default (`precision` parameter); scientific notation is
  rewritten `m \cdot 10^{k}`; `inf`/`nan` render `\infty`/`\mathrm{NaN}` (degenerate
  fits must never crash rendering). A leading `-` binds like an Add-level fragment,
  and `a + -2` is normalised to `a + \left( -2 \right)`.
- Variables render `x_{i}`; **feature-name substitution happens in the bindings**
  (names never cross the bridge): fixed-token replacement of `x_{i}`, with `_` in
  names escaped to `\_`. Explicit `variable_names` overrides, empty forces `x_{i}`.

The `latex` column is generated at fit time in both bridges because the `Tree`
does not survive the bridge; ~`maxsize` short strings per fit is negligible.
`to_pandas()` / `as.data.frame()` deliberately do **not** include the column
(both frames stay lean and symmetric; members/dicts carry it).

## D4. R² from `n_obs`/`sst`, no stored data

The bridges compute `sst = Σ w_i (y_i − ȳ_w)²` (unit weights when unweighted) at
fit time and expose it with `n_obs`. Per-member training R² is then
`1 − loss / sst`, consistent with the (weighted) SSE `loss` — no training data is
stored and nothing is re-evaluated. `sst == 0` (constant target) → `NA`/`None`;
negative R² is valid (fit worse than the mean); with `weights` this is weighted R².

## Compatibility

Pre-1.0, no shims: result objects fitted before these columns existed must be
re-fitted to use `summary()`/`as.data.frame()`/`to_latex()` (R errors with a clear
message in `to_latex()`; Python raw dicts always come from the current bridge).
Search behaviour, defaults, and the PySR default-parity comparison are unchanged —
every addition is display-only.

## Verification

- C++: `standalone/tests/test_hall_of_fame.cpp` (score values, edge cases,
  `select_best == argmax(pareto_scores)` regression), new
  `standalone/tests/test_to_latex.cpp` (32 checks: precedence, constants,
  degenerate values).
- R: testthat 138 PASS on Windows (score column & argmax consistency, `to_latex`
  fixtures incl. `x_{10}` substitution and error paths, R² exact/constant-target,
  predict round-trip unchanged).
- Python: pytest 28 PASS on Windows (score, repr marker/elision, `get_best`,
  `latex` substitution, Agg-backend `plot`, R² consistency).
- Ubuntu (WSL): standalone ctest + R testthat at the Phase-1 milestone and again
  at completion.
