# 81 — Aligning the R and Python inputs with each ecosystem's regression conventions

Status: implemented (2026-08-03). Scope: the two host-language layers only —
`python/rsymbolic2/__init__.py` and `r-package/rsymbolic2/R/*.R`, plus their tests and
docs. **No file under `r-package/rsymbolic2/src/rsymbolic/`, `python/src/` or `web/` is
touched**, so the search trajectory, every default, and PySR parity (CLAUDE.md, docs/28)
are unchanged by construction. This continues the interface work of docs/61 and docs/74.

## Why

Measured against what `lm`/`glm` establish in R and what scikit-learn establishes in
Python, rsymbolic2's inputs are mostly right — the formula method deliberately avoids
`model.matrix()` (docs/61), the 1-D `X` rule is stated and shared, and docs/80 already
classifies invalid data into refuse / warn / accept. Five gaps remain. One of them is a
correctness defect, reproduced before this plan was written:

```python
r = symbolic_regression(df, y)        # df columns: a, b
r.predict(df)                         # RMSE 2.8e-16
r.predict(df[["b", "a"]])             # RMSE 5.69 — no error, no warning
```

The fit captured `feature_names = ['a', 'b']`, and `predict` ignored them. R closed
exactly this hole for formula fits in docs/61 C2; Python's has been open since.

## The governing question, and why the answer is per-language

The tempting framing is "make the two interfaces symmetric". It is the wrong one. R and
Python disagree about what a fitted regression object *is*, and copying one convention
into the other language produces something that is idiomatic in neither:

| | R (`lm`) | Python (scikit-learn) |
|---|---|---|
| How the fit binds columns | by **name** (`terms`, model frame) | by **position** (`X` is a matrix) |
| `predict` on renamed/reordered columns | reorders by name | **raises**; never reorders |
| Training data retained on the object | yes (`model = TRUE` by default) | no (only `n_features_in_`) |
| Non-finite values at predict time | propagated as `NA` | rejected (`check_array`) |

So each row below is decided by the *host* convention, and the divergences are
deliberate. They are listed as such in "Deliberate divergences" so a future reader does
not "fix" them into symmetry.

## Changes

### P1 — Python `predict` refuses a feature-name mismatch (correctness)

`SymbolicRegressionResult.predict` gains a check: when the fit captured
`feature_names` **and** `newdata` carries column names, the two lists must be equal.
Otherwise `ValueError`, naming the first difference.

**It refuses; it does not reorder.** Reordering is R's rule and follows from R's fit
binding columns by name. Python's fit binds by position — `feature_names` is metadata
captured from a DataFrame, not the thing the model was built on — so silently permuting
the caller's columns would invent a guarantee the fit never made. scikit-learn refuses
for the same reason.

The asymmetric cases (fit named / predict unnamed, or the reverse) stay **silent**, which
is a deliberate deviation from scikit-learn, which warns. `predict(X_test.values)` and
`predict(X_test.to_numpy())` are ordinary idioms; warning on them is noise on a path where
nothing can be checked either way. Recorded here because it is a knowing divergence.

`_plot_fit` currently converts `X` with `np.asarray` before calling `predict`, which would
bypass the check. It is changed to pass the caller's original object through.

### P2 — R keeps the training data, and gains `fitted()` / `residuals()`

Today the result object stores `n_obs` and `sst` but not the data
(`plot_rsymbolic2.R:27` states this deliberately). The consequences are that
`predict(fit)` is an argument-missing error, `fitted()` and `residuals()` do not exist,
and `plot(type = "fit")` requires the caller to hand the data back.

For R this is the wrong default. `lm(model = TRUE)` retains the model frame precisely so
that the generic vocabulary — `fitted`, `residuals`, `predict` with no `newdata`,
`plot` — works on the returned object. That vocabulary *is* R's regression contract.

- `symbolic_regression.default()` gains `keep_data = TRUE`, storing the coerced numeric
  matrix as `$X` and the response as `$y`. `keep_data = FALSE` restores today's
  behaviour for large data.
- The argument is **not** spelled `model =` (as `lm` does) because what is stored is not
  an `lm`-style model frame; promising that name and delivering `$X`/`$y` would be worse
  than a different name.
- `fitted.rsymbolic2(object, expression = NULL, ...)` evaluates any expression (default
  `recommended`) on the stored `$X`.
- `residuals.rsymbolic2(object, expression = NULL, ...)` returns `$y - fitted(...)`.
- `predict.rsymbolic2` makes `newdata` optional; omitted, it returns `fitted(object)`.
- `fit_plot()` falls back to the stored data when `newdata`/`y` are omitted.

One implementation note that is easy to get wrong: `fitted()` must **not** route through
`design_matrix()`. That helper requires a `data.frame` for formula-fitted models (docs/61
C2), and the stored `$X` is a matrix. The expression evaluation is therefore factored out
of `predict.rsymbolic2` into a helper that takes an already-built numeric matrix, and both
`predict` and `fitted` call it.

Objects fitted by an older version carry no `$X`; every new entry point checks and raises
a message that says to re-fit or to pass `newdata`.

### P3 — Error messages carry the remedy

Refusals are correct (docs/80) but currently state only what is wrong.

- Non-finite `X`/`y`: add the fix — `na.omit()` in R, `dropna()` in Python. rsymbolic2
  will not grow an `na.action`: silently dropping rows is a decision the caller should
  make visibly, and `na.exclude`'s padded-prediction semantics are not worth the
  maintenance.
- Non-numeric input: name the offending column(s) instead of the blanket
  `X must be numeric`, and point at explicit encoding (`model.matrix(~ f - 1, df)` /
  `pandas.get_dummies`). The R default method must inspect a data frame **before**
  `as.matrix()` collapses it to a character matrix, which is why the check moves earlier.
  Automatic dummy coding stays out: it would enlarge the search space with columns the
  user never chose.

### P4 — Python `variable_names=` at fit time, and a finiteness check in `predict`

- `symbolic_regression(..., variable_names=None)`: an explicit sequence of length
  `n_features`, overriding names taken from a DataFrame. PySR has this; today an ndarray
  caller can only supply names later, per display call.
- `predict` rejects non-finite `newdata`, matching scikit-learn's `check_array` default
  and the finiteness rule the training path already enforces.

R deliberately gets neither: `colnames(X)` and the formula already name columns, and
`predict.lm` propagates `NA` rather than refusing.

### P5 — A thin `SymbolicRegressor` class in Python

Provided for **one** reason: callers migrating from `PySRRegressor(...).fit(X, y)` should
not have to restructure their code. It is not a claim of scikit-learn ecosystem
integration, and the documentation will say so:

- A `Pipeline` with a scaler in front returns an expression **in standardised
  coordinates**, destroying the interpretability that is symbolic regression's entire
  product.
- `GridSearchCV` over these hyperparameters contradicts this project's highest-priority
  rule (CLAUDE.md: defaults are PySR's and are not to be replaced by self-tuned values),
  and one default fit is 2800 generations across 31 populations.

`train_test_split`, `cross_val_score` and `clone` do work, because the class implements
`fit`/`predict`/`score`/`get_params`/`set_params` by duck typing. **scikit-learn is not
imported and does not become a dependency** (Dependency Policy: the default answer is no).

One correction found while verifying that claim, worth recording because it is not
guessable: **duck typing alone is no longer enough for `cross_val_score`.** On
scikit-learn 1.8, `clone()` succeeded but `cross_val_score()` raised

```
AttributeError: 'SymbolicRegressor' object has no attribute '__sklearn_tags__'
```

Since 1.6 every estimator is routed through `get_tags()`, which reads that method and
does not fall back. So `__sklearn_tags__` is implemented — the one scikit-learn-internal
protocol this class speaks — with the import *inside* the method, so it runs only when
scikit-learn is already driving and the package still has no scikit-learn dependency. It
declares `non_deterministic=True`, which is simply true: the search is stochastic unless
`seed` is fixed. If a future version changes the `Tags` dataclass, the fallback is the
pre-1.6 situation (`clone` works, `cross_val_score` does not) rather than a broken
estimator for callers who never touch scikit-learn.

To avoid maintaining a second copy of ~40 hyperparameters, `__init__(**params)` validates
its keys against `inspect.signature(symbolic_regression)` and stores them verbatim;
`get_params()` returns that dict, which is what `clone()` needs. Learned attributes use
the trailing-underscore convention (`result_`, `n_features_in_`, `feature_names_in_`).

## Deliberate divergences (do not "fix" these into symmetry)

1. Python `predict` **refuses** a name mismatch; R **reorders** by name. Fit-by-position
   vs fit-by-name.
2. R stores the training data; Python does not. `lm` vs scikit-learn estimators.
3. Python `predict` refuses non-finite `newdata`; R propagates `NA`. `check_array` vs
   `predict.lm`.
4. Python gains `variable_names=`; R does not need it.
5. Feature-name checking is silent when only one side carries names (scikit-learn warns).

## Explicit non-goals

`na.action` / `subset` in R; automatic dummy coding in either language; renaming `seed`
to `random_state` (it would diverge from PySR and break existing callers); a scikit-learn
dependency; and any change to defaults, search behaviour, or files under `src/`.

## Verification (results)

| Check | Windows 11 | Ubuntu 24.04 (WSL) |
|---|---|---|
| R `testthat` (`NOT_CRAN=true`) | `FAIL 0 \| WARN 0 \| SKIP 0 \| PASS 409` | identical: `PASS 409` |
| `pytest python/tests` | `116 passed` | `109 passed, 7 skipped` (no matplotlib) |
| `R CMD check --no-manual` | `Status: OK` | — |

New test files: `python/tests/test_input_conventions.py` (20 cases) and
`r-package/rsymbolic2/tests/testthat/test-model-interface.R` (27 cases).

**Engine untouched.** No file under `r-package/rsymbolic2/src/`, `python/src/`,
`standalone/` or `web/` is modified, and `git diff` over both binding layers shows no
change to the `symbolic_regression_cpp(...)` argument lists — the search receives exactly
the arguments it did before. This is a stronger statement than a digest comparison, which
would only sample it.

### Three existing tests asserted the old contract and were updated

Recorded so it is clear they were not bent to fit a defect:

- `test-output.R` pinned the exact names of the result list; `X` and `y` are now among
  them (P2).
- `test-plot.R` asserted that `plot(type = "fit")` with no data errors; it now draws
  against the stored training data. The half-supplied case still errors, with the new
  message.

### One gap the first Ubuntu run exposed

The new Python test file opened with a module-level `pytest.importorskip("pandas")`, and
the Ubuntu verification venv has no pandas — so **all 20 new tests silently skipped
there**, and the run still looked green. Two-platform verification means the new code
runs on both, not that both runs exit zero. The skip is now per-test: the 6 cases that
need no pandas run anywhere, and pandas/scikit-learn were added to the Ubuntu venv so the
rest run too. Worth remembering: a module-level `importorskip` turns a missing optional
extra into invisible loss of coverage.
