# rsymbolic2 0.1.0.9000 (development)

New operators (docs/62):

- `unary_ops` accepts `"erf"`, `"sinh"` and `"cosh"`. All three are **opt-in**: the
  default operator set is unchanged, so PySR default parity is untouched. They cover
  physical-science motifs that no combination of the existing operators (or of a macro
  operator, which may use its argument only once) can build. Each has the same name and
  meaning in SymbolicRegression.jl, so a comparison can still hand both tools the
  identical operator set. `sinh`/`cosh` are unguarded like `exp` — a large argument
  overflows and the loss guard rejects the candidate; `erf` is bounded, so no guard
  arises. `predict()` gains an `erf` shim (`2 * pnorm(x * sqrt(2)) - 1`), which is why
  the package now declares `Imports: stats`.

Interface fixes (docs/61). None of these touch the C++ core, so the search itself,
its defaults, and PySR parity are unchanged:

- **Behaviour change.** A model fitted through the formula interface now *requires* a
  `data.frame` as `newdata` in `predict()` and `plot(type = "fit")`. Its predictor
  columns are matched by name, and a matrix carries no names to match — it was
  silently taken positionally, so a matrix whose columns were in a different order
  than the fitted terms returned wrong numbers with no error. Matrix-fitted models
  are unaffected and stay positional.
- `predict()` now returns one value per row of `newdata` for every expression. An
  expression that uses no data column — a bare constant, which the simplest
  Pareto-front member usually is — evaluated to a single value regardless of
  `nrow(newdata)`, contradicting the documented return contract; it is now recycled.
  This also fixes `plot(type = "fit", expression = <a constant member>)`, which
  failed with a misleading "newdata has 1 row(s)".
- `predict()` now evaluates the `inf` and `nan` constant tokens that the core's
  `"%.6g"` rendering can emit. R parses them as names, so they previously raised
  `object 'inf' not found` on a string the package's own `plot(type = "tree")`
  drew without complaint.
- `predict()` reports a non-numeric `newdata` column directly instead of letting it
  surface from inside `eval()` as "non-numeric argument to binary operator".
- `summary()`, `as.data.frame()` and `print()` no longer propagate `NA` when a fit
  carries no recommendation (`best_index = NA`); they mark no row instead, where
  `summary()` previously failed with "missing value where TRUE/FALSE needed".

Result-display additions (docs/48):

- `pareto_front` gains a `score` column computed by the C++ core — the log-loss
  drop per unit of added complexity that `model_selection` ranks by. The first
  member's score is `0` (previously `summary()` showed `NA`); `summary()` and
  `as.data.frame()` now read this column instead of recomputing it in R, so the
  displayed score and the engine's recommendation agree by construction. Objects
  fitted with earlier versions lack the column and must be re-fitted for
  `summary()`/`as.data.frame()`/`to_latex()`.
- New `to_latex()` generic and method: display-only LaTeX rendering of
  Pareto-front members (minimal parentheses, `\frac`, `\cdot`, `\sqrt`),
  serialized by the C++ core with no new dependency. `variable_names` (default:
  the fit's `feature_names`) substitutes names for the `x_{i}` tokens. The plain
  `expression` strings are unchanged and remain what `predict()` evaluates.
- The result gains `n_obs` and `sst` (total sum of squares about the weighted
  mean); `summary()` reports a per-member training `r_squared` column and a
  headline `R-squared (recommended)` line (`NA` when the target is constant;
  weighted R-squared when `weights` were used).
- `plot()` gains a second view: `plot(res, type = "fit", newdata = X, y = y)`
  draws a fitted expression against the data — the fitted curve over the observed
  scatter for a single feature, predicted vs. observed with a dashed reference
  line otherwise. Result objects store no training data, so the data is passed
  back in (held-out data works the same way); `expression` selects any Pareto
  member, as in `predict()`. Supplying `newdata` and `y` without naming `type`
  selects the fit view. The default is still `type = "pareto"`, unchanged.
- `plot()` gains a third view: `plot(res, type = "tree")` draws one expression as a
  syntax tree — operators as inner nodes, data columns and fitted constants as
  leaves, distinguished by fill. It needs no data. `expression` selects any Pareto
  member (the default is the recommended one, in its display-simplified form) and
  `variable_names` (default: the fit's `feature_names`) labels the leaves. The node
  count is that of the expression as printed, which can be smaller than the
  `complexity` column — that counts the raw tree the search archived.

# rsymbolic2 0.1.0

Initial release.

- Symbolic regression via steady-state genetic programming, implemented in C++ and
  exposed through `symbolic_regression()`.
- Levenberg-Marquardt constant optimization (self-contained C++/STL solver, no
  third-party linear-algebra dependency) with random-restart fallback.
- OpenMP island model with inter-island migration; serial fallback when OpenMP is
  absent.
- Operators: `+`, `-`, `*`, `/`, `^`, `neg`, `exp`, `log`, `sin`, `cos`, `sqrt`,
  `tanh`, `abs`, `square`.
- S3 methods for fitted models: `predict()`, `print()`, `summary()`,
  `as.data.frame()`, and `plot()`. `print()` shows a compact view; `summary()`
  gives the full Pareto front with a per-member score; `as.data.frame()` returns
  the front as a tidy data frame (the R counterpart of Python's `.to_pandas()`).
- `predict()` now evaluates the recommended (Pareto "best") expression by
  default, matching PySR and the Python interface. A new `expression` argument
  selects any other expression to evaluate (e.g. `expression = res$expression`
  for the lowest-loss model, or any row of `res$pareto_front$expression`).
