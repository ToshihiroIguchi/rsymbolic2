# rsymbolic2 0.1.0.9000 (development)

Invalid and degenerate data (docs/80). None of these changes the search; each fires
before `run_evolution()` is reached, and ordinary data is unaffected.

- **Refused, naming the argument** — datasets with no defensible reading, which the
  engine previously ran on and returned a plausible-looking result for:
  - an `X` with no columns (it silently returned the constant expression `"1"`);
  - an all-zero `weights` vector (the weighted SSE is then identically `0`, so every
    candidate tied at a perfect loss and the "best" expression was whichever one the
    tournament happened to hold — reported as `loss = 0`).
  Both are also guarded in the C++ bridge, so the refusal holds for any caller.
- **Warned about, but still run** — legal datasets the search cannot say anything
  about, each of which made a reported number meaningless with nothing on screen to
  say so: a constant `y` (zero variance, which is exactly why `summary()` prints `NA`
  for `r_squared`), a constant feature column, and a `y` whose total sum of squares
  overflows to non-finite. See `?symbolic_regression`, section "Degenerate data".
- **Accepted** — a logical `X` is now read as `0`/`1` instead of being rejected by the
  `is.numeric()` guard. The rejection only ever fired on a *purely* logical matrix,
  since `as.matrix()` promotes a logical column sitting next to a numeric one, and
  Python has always accepted it.

Guarded operator semantics (docs/69, docs/77):

- **Search behaviour change.** The domain-guarded operators now return `NaN` outside
  their domain rather than a substituted finite value, matching
  `SymbolicRegression.jl`'s `safe_sqrt` / `safe_log` / `safe_pow` and so PySR's search.
  The `NaN` is the mechanism: it makes the candidate's loss non-finite, which is how
  the search **rejects** that candidate. Returning `0` instead let expressions survive
  that PySR discards, and made `predict()` answer a plausible finite number outside the
  model's domain.
  - `sqrt(x)` and `pow(x, y)` (docs/69).
  - `log(x)` is now guarded too (docs/77): `log(x)` for `x > 0`, `NaN` otherwise. Only
    the `x == 0` case changes — IEEE `log` already gave `NaN` for `x < 0` — but `-Inf`
    is not a fixed point of the operator set, so `exp(log(0))` used to score as a
    finite `0` and survive, where SR.jl rejects it. `log` and `exp` are both default
    operators, so this was reachable on the default path.
- The documented guarantee in earlier manuals ("returns 0 for undefined inputs") was
  stale text describing the pre-docs/69 behaviour; it has been corrected everywhere.
- `predict()` evaluates with R's own operators, which follow IEEE rather than these
  guards at a few edges — `log(0)` is `-Inf` and `0 ^ -1` is `Inf` in R, `NaN` in the
  engine. It matters only if the prediction inputs reach them.

Expression rendering (docs/71):

- **Display change.** `square`, `inv` and `neg` are now printed as the operators they
  are: `expression` and `expression_simplified` read `(x0 ^ 2)`, `(1 / x0)` and
  `(-x0)` where they used to read `square(x0)`, `inv(x0)` and `neg(x0)`. Those are the
  engine's internal names, not notation, and `^`, `/` and unary minus are already in
  the grammar — `to_latex()` has always rendered the same nodes as `x_{0}^{2}` and
  `-x_{0}`. The rewrites are exact, not approximate: with an integer exponent
  `safe_pow`'s only guard is `y < 0 && x == 0`, so `pow(x, 2)` is `x*x` for every
  double including ±Inf and NaN, and `inv` is literally `1/a` with the same unguarded
  division. Nothing about the search changes.
- Side effect: an `expression` string now contains no name SymPy is unaware of, so
  `sympify()` parses it correctly on its own. `to_sympy()` is still the right thing to
  paste anywhere else — it rewrites `^`, which Python reads as xor.
- `plot(type = "tree")` labels a negation `-` instead of `neg`, matching the equation.
- `square(...)`, `inv(...)` and `neg(...)` are still **accepted** everywhere they were:
  macro bodies still spell them (`c(gauss = "exp(neg(square(x)))")`), and `predict()`
  keeps its bindings, so an expression string saved by an earlier version still
  evaluates.

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
