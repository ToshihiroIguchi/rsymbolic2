# rsymbolic2 0.1.0.9000 (development)

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
