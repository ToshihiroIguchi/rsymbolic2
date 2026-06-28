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
