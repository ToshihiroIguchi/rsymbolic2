# 48. Result-display features (score, LaTeX, plot, R²)

Status: implemented (2026-07-04); extended with the fit and residual views
(2026-07-25, D5) and the equation tree diagram (2026-07-25, D6).

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

## D5. Fit and residual views (2026-07-25)

The three surfaces had drifted apart: the web GUI drew the equation against the
data, while `plot()` in R and Python only ever drew the Pareto front — the front
answers "how does accuracy trade against complexity", never "does this equation
actually follow the data". The gap is closed from both ends.

| Surface | Addition |
|---|---|
| R | `plot(type = "fit", newdata =, y =, expression =)`; internal `design_matrix()` shared with `predict()`; `type = "pareto"` unchanged and still the default |
| Python | `plot(kind="fit", X=, y=, expression=)`, same two views, same defaults |
| Web GUI | `view` dropdown on the fit card: `fit` (unchanged) or `residual` |

- **The data is passed back in, not stored.** D4's rule stands: a result object
  keeps `n_obs`/`sst` and no training data, so the fit view takes `newdata`/`y`
  (`X`/`y` in Python) explicitly. That also makes held-out data a first-class
  argument rather than a special case.
- **`type`/`kind` defaults to the fit view when the data is given** (nothing else
  uses those arguments) and to `"pareto"` otherwise; an explicit value always
  wins. Every existing `plot(res)` / `res.plot()` call is unaffected.
- **One feature gets the curve overlay, several get predicted-vs-observed.** The
  overlay is the more direct reading but needs a single x-axis to exist; the
  scatter against a dashed `y = x` is the general fallback. Same rule in all three
  surfaces (web: `drawPrediction`).
- **Residuals stay a web-GUI view for now.** `actual − predicted` vs `predicted`
  with a dashed zero line, sharing the fit card's canvas and its
  `DISPLAY_POINT_CAP` down-sampling. It is the one reading that survives a dense
  scatter and a near-1 R², both of which hide systematic error. It was added where
  the audience is least likely to run a residual plot by hand; R and Python users
  have `predict()` and their own plotting stack, so no third view was pushed into
  those APIs without a request for it.
- **Presentation stays inline, not modal.** The charts are primary evidence for the
  headline equation and the Pareto→equation→fit loop needs them beside the table;
  `<dialog>` remains reserved for the secondary surfaces (data preview, numeric
  settings). Recorded here because the alternative was considered.

Display-only, as with everything else in this document: no search behaviour, no
default, and no stored expression changes.

## D6. Equation tree diagram (2026-07-25)

The formula and the expression string both say *what* the equation is; neither shows
*where* things are nested, which is the reading that explains a complexity number. D6
draws the selected equation as a syntax tree — operators as inner nodes, data columns
and fitted constants as leaves — on all three surfaces.

| Surface | Addition |
|---|---|
| Web GUI | `Equation tree` card under the hero, inline SVG (`js/tree.js`), follows every equation selection, downloads as a standalone `.svg` |
| R | `plot(type = "tree", expression =, variable_names =)` (`R/tree_plot.R`) |
| Python | `plot(kind="tree", expression=, variable_names=)` |

- **The tree is not exported from the C++ core.** Every surface already parses the
  printed infix string: `parseExpression()` in `predict.js`, `str2lang()` in R (which
  `predict.rsymbolic2()` already relies on), and stdlib `ast` in Python. Exporting a
  `Tree` would have touched all four bindings and would have had to export *two* trees
  (raw and display-simplified) to match what the UI shows, for no gain over 5–20 lines
  of language-native parsing. The triplication is the accepted pattern already used by
  the three `predict` evaluators (`docs/51`).
- **The tree draws the printed (display-simplified) expression by default** — the
  string the hero card, the Pareto table and `print()` already show. An explicit
  `expression=` argument draws any other member, including the raw searched form.
  Consequently **the drawn node count is not `complexity`**, which counts the raw
  archived tree; the web caption says so in its tooltip, and both packages say so in
  their help text. Same relationship the table already renders as `10 → 7`.
- **Nodes are capsules, not fixed circles.** There is no formal standard for
  expression-tree node shapes (ISO 5807 is flowcharts; UML is class diagrams; PySR
  ships no tree view). The de-facto convention is Graphviz, whose default is
  `shape=ellipse`, *not* `circle` — the convention itself widens with the label. A
  fully-rounded capsule is a circle for the 1–3 character labels that dominate and
  stretches for `sqrt` or `-1.23457e-05`. `ggforce` (an ellipse geom), Graphviz and
  DiagrammeR were all rejected on the Dependency Policy; the capsule is what all three
  stacks draw correctly with what they already have (`<rect rx="h/2">`, `geom_label`,
  matplotlib `boxstyle="round"`).
  - Two geometry traps, recorded because both produced broken pictures first: grid does
    **not** clamp `label.r`, so a radius wider than half the box makes the corner arcs
    overlap and the text spills out; and `label.padding` applies to all four sides, so a
    one-character label is taller than it is wide and the correct radius degenerates.
    One space on each side of the label buys the width. The layout table keeps the bare
    labels, so the tests are unaffected.
- **Node fill encodes the kind** (operator / variable / constant), identically on all
  three surfaces. Uniform shape, no legend. The reference pictures in the literature do
  not carry this, but which leaves are data columns and which are fitted constants is
  exactly what a symbolic-regression reader wants to know.
- **One layout, three renderers.** Leaves take consecutive integer columns, an inner
  node sits at the mean of its children, `y` is the depth. Sibling subtrees own disjoint
  leaf ranges, so same-depth nodes cannot collide and a unary node lands directly above
  its child. Reingold–Tilford was rejected: at the default `max_nodes = 30` it draws the
  same picture for a fraction of the code. The column pitch comes from the widest label
  in the tree (measured with `getComputedTextLength()` in SVG), or a long constant
  overlaps its neighbour.
- **Three normalisations keep the surfaces identical.** R's parser *keeps* parentheses
  as `(` call nodes (Python's `ast` and the browser parser do not) — they are stripped,
  or the R tree would be visibly larger for the same equation. A negated literal folds
  into one constant (`%.6g` prints `-1.3`, every parser reads unary minus over `1.3`),
  as `parse_expression.hpp` already does for macro bodies. `/` and `*` are relabelled
  `÷` and `×`: `/` reads as part of a fraction that is not there, and `*` is a raised
  asterisk that sits off-centre inside a node. The R sources spell both with
  `intToUtf8()` so the package code stays ASCII.
- **Colours come from CSS variables in the web GUI**, not from JavaScript, so the theme
  toggle recolours the tree with no redraw — unlike the Chart.js plots, which must be
  rebuilt. The SVG download re-resolves them into presentation attributes, since CSS
  variables do not exist outside the document.

Display-only, like the rest of this document: no search behaviour, no default, and no
change to the frozen `expression` strings. Nothing under `src/rsymbolic/` and none of the
four bindings were touched.

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

D5 (2026-07-25): R testthat 228 PASS on Windows (new `test-plot.R`: default view
unchanged, curve overlay against `predict()` in x order, multi-feature fallback,
type inference, formula column names, error paths — all forcing `ggplot_build`,
since a ggplot object built lazily proves nothing); Python pytest 54 PASS
(single/multi-feature axes and layer counts, `kind` inference, error paths, Agg
backend); web GUI checked in-browser on the quadratic example — both views draw,
the toggle survives an equation change, console clean. Ubuntu (WSL): R testthat
228 PASS (identical), Python pytest 53 PASS / 1 skipped (the skip is the optional
pandas `to_pandas` test, unrelated). `R CMD check --as-cran` on the built tarball:
1 NOTE (new submission) — the `aes()` column names are declared in the package's
existing `utils::globalVariables()`, and the fit frame's column is `predicted`,
not `fitted`, to avoid shadowing the `stats` generic.

D6 (2026-07-25): the reference expression `(2.2 - (x0 / 11)) + (7 * cos(x1))` — the
worked example of a tree diagram — is asserted to lay out as **10 nodes, depth 3,
root `+`** with identical labels, kinds and x-positions in all three surfaces, which
is what keeps the implementations from drifting. R testthat 249 PASS / 0 FAIL / 27
SKIP on Windows (layout of the reference expression, `(`-stripping — `(((x0)) +
((1)))` is 3 nodes, not 7 — negated-literal folding, `inf`/`nan` as constants,
`variable_names` substitution and its feature-count check, error paths, and the built
plot's layer/fill counts through `ggplot_build`); Python pytest 58 PASS on Windows
(the same layout assertions plus the Agg-backend artist counts, `kind` validation and
`ax.axison`). Web GUI driven in-browser: the tree renders, follows a Pareto/table
selection, recolours on the theme toggle with no redraw, serialises to a standalone
SVG with inlined colours, and the console stays clean; `buildTree` was checked to
return byte-identical labels to the R and Python layouts for the same six expressions.
Ubuntu (WSL): R testthat 249 PASS / 0 FAIL / 27 SKIP (identical), Python pytest 57
passed / 1 skipped (the skip is the optional pandas `to_pandas` test, unrelated).
`R CMD check --as-cran` on the built tarball: **Status OK** (R code, dependencies, S3
consistency and non-ASCII checks all clean; the `intToUtf8()` spelling of the `×`/`÷`
labels is what keeps the R sources ASCII). No file under `src/rsymbolic/` and no
binding was touched, so the WASM parity gate is unaffected.
