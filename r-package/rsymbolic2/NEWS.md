# rsymbolic2 0.1.0

Initial release. rsymbolic2 discovers a closed-form expression that fits data, using
steady-state genetic programming with Levenberg-Marquardt constant optimization. The
search engine is C++ (shared with the project's Python interface) and is reached through
`symbolic_regression()`. Its default configuration and search behaviour are matched to
the documented defaults of PySR / SymbolicRegression.jl; only the implementation differs.
The `docs/` directory of the source repository records the evidence behind each design
decision named below.

## The search

- Steady-state genetic programming over expression trees, with tournament selection,
  subtree crossover, a weighted mutation set, and frequency-adaptive parsimony.
- Levenberg-Marquardt constant optimization in a self-contained C++/STL solver
  (forward-mode automatic differentiation for the Jacobian, random-restart fallback);
  no third-party linear-algebra dependency.
- OpenMP island model with inter-island and hall-of-fame migration, and a correct serial
  fallback when OpenMP is absent. `n_threads` bounds the worker team; the search is
  bit-deterministic across thread counts for a fixed seed, so it changes speed only.
- Off-by-default options that go beyond PySR's defaults: `batching`, `warmup_maxsize_by`,
  `eval_cache`, `linear_scaling`, `strong_simplify`, dimensional analysis
  (`X_units` / `y_units`), and `macro_ops` user-defined operator templates. With every one
  left at its default the search reproduces PySR's exactly.

## Operators

- Default set `add`, `sub`, `mul` and `neg`, `exp`, `log`, `sin`, `cos`; also available
  are `div`, `pow`, `sqrt`, `tanh`, `abs`, `square`, `inv`, `erf`, `sinh`, `cosh`. Every
  name exists in SymbolicRegression.jl with the same meaning, so a comparison can hand
  both tools the identical set.
- `sqrt`, `log` and `pow` are domain-guarded exactly as SymbolicRegression.jl's
  `safe_sqrt` / `safe_log` / `safe_pow`: outside their domain they yield `NaN`, never a
  substituted finite value. That is the mechanism, not a hazard — the candidate's loss
  becomes non-finite, which is how the search rejects it. The others are unguarded:
  `div`/`inv` on zero and `exp`/`sinh`/`cosh` on a large argument overflow, and the loss
  guard rejects those candidates the same way. `erf` is bounded, so no guard arises; it
  is why the package declares `Imports: stats` (`predict()` evaluates it through
  `pnorm()`).
- `square`, `inv` and `neg` print as the notation they are — `(x0 ^ 2)`, `(1 / x0)`,
  `(-x0)` — while the `square(...)` / `inv(...)` / `neg(...)` spellings stay accepted
  everywhere they are written, including macro bodies and `predict()`.

## The fitted object and its methods

- `symbolic_regression()` accepts a matrix and vector, or the formula interface
  (`y ~ x1 + x2`, `y ~ .`) with `data`. Only bare variables are allowed on the
  right-hand side: discovering transformations is the search's job.
- S3 methods: `predict()`, `fitted()`, `residuals()`, `print()`, `summary()`,
  `as.data.frame()` and `plot()`, plus the `to_latex()` and `to_sympy()` generics.
  `predict()` evaluates the recommended (Pareto "best") expression by default, matching
  PySR and the Python interface; `expression =` selects any other Pareto member. Called
  with no `newdata` it returns the fitted values.
- `keep_data` (default `TRUE`) stores the training `X`/`y` on the result, which is what
  `fitted()`, `residuals()`, `predict(object)` and `plot(type = "fit")` read — as `lm()`
  keeps its model frame. `keep_data = FALSE` gives a data-free object, and those four
  entry points then say so rather than guessing.
- A formula-fitted model requires a `data.frame` as `newdata`: its columns are matched by
  name, and a matrix carries no names to match. Matrix-fitted models stay positional.
- `plot()` draws three views: `"pareto"` (default) the accuracy/complexity front,
  `"fit"` the expression against the data (training data when none is passed, held-out
  data when it is), and `"tree"` the expression's structure. Requires ggplot2.
- `pareto_front` carries a per-member `score` computed by the C++ core — the log-loss
  drop per unit of added complexity that `model_selection` ranks by — so the displayed
  score and the engine's recommendation agree by construction. `summary()` adds a
  training `r_squared` column from the result's `n_obs`/`sst` (weighted when `weights`
  were used, `NA` for a constant target).
- `to_latex()` and `to_sympy()` render Pareto members for display only, serialized by
  the C++ core with no new dependency: LaTeX for typesetting, and Python that SymPy's
  `sympify()` parses (`^` means power in an expression string and xor in Python). The
  plain `expression` strings remain what `predict()` evaluates.

## Data the search cannot use

- Refused, naming the argument: an `X` with no columns, an all-zero `weights` vector,
  non-finite `X`/`y`, and non-numeric columns (named individually). There is deliberately
  no `na.action` and no automatic dummy coding — dropping rows and encoding factors are
  decisions worth making visibly, and a column the user did not choose only enlarges the
  search space.
- Warned about but still fitted, because each makes one reported number meaningless: a
  constant `y` (zero variance, so `r_squared` is `NA`), a constant feature column, and a
  `y` whose total sum of squares overflows. A logical `X` is accepted as `0`/`1`.
