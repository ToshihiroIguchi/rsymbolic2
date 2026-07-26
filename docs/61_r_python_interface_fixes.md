# 61. R / Python interface fixes

Status: implemented (2026-07-26). Scope: the two host-language layers only —
`python/rsymbolic2/__init__.py` and `r-package/rsymbolic2/R/*.R`. **No file under
`r-package/rsymbolic2/src/rsymbolic/` or `python/src/` was touched**, so the search
trajectory, the defaults, and PySR parity (CLAUDE.md, docs/28) are unchanged.

An audit of the R and Python interfaces found places where the shipped code contradicted
its own documentation, returned a wrong-length result, or diverged between the two
languages for the same operation. Each was reproduced at runtime before being fixed.

## Two contract changes worth recording

Everything else below was a straightforward bug. These two changed a documented (or
de-facto) contract, so they are recorded here rather than only in `NEWS.md`.

### C1. A 1-D `X` is one *column*, in Python as it already was in R

`np.atleast_2d` promotes `(n,)` to `(1, n)`, so Python's `symbolic_regression()` and
`predict()` read a 1-D array as a single *row* of n features. The guard that followed
(`if X.ndim == 1: X.reshape(-1, 1)`) could never fire. Three places already promised the
opposite reading — `README.md` ("A 1-D input is treated as a single column"), both
docstrings, and `_plot_fit()`, which does the correct reshape with a comment explaining
why — and R has always agreed, because `as.matrix(<vector>)` yields a column.

The effect was that the most natural call in symbolic regression simply did not work:

```python
symbolic_regression(np.linspace(-3, 3, 20), y)   # ValueError: X.shape[0] must equal len(y)
res.predict(np.array([0., 1., 2.]))              # ValueError: newdata has 3 column(s)
```

Both now route through one helper, `_as_design_matrix()`: 1-D becomes a column, 2-D
passes through, anything else raises.

**Consequence.** `predict(np.array([1.0, 2.0]))` on a *two-feature* model used to return
one prediction, reading the array as a single sample. It now raises "1 column(s) but the
model was fitted on 2 feature(s)". This is deliberate. One shape cannot mean both "n
samples of one feature" and "one sample of n features"; picking per-model would make the
same array mean different things on different fits, which is how a silently wrong
prediction gets made. scikit-learn refuses the same guess. A caller with one
multi-feature sample passes an explicit `(1, n_features)` array, and the error message
says so.

### C2. A formula-fitted R model requires a `data.frame` for `newdata`

`design_matrix()` name-matched columns only when `is.data.frame(newdata)`; a matrix fell
through to the positional branch with nothing but an `ncol` check. The documentation
promises the opposite for formula fits — "selected by name in the fitted order, so column
order in `newdata` does not matter" — and a user who fitted `y ~ a + b` has no reason to
expect their column order to matter. It did:

```r
fit <- symbolic_regression(y ~ a + b, data = df)     # fitted order: a, b
predict(fit, as.matrix(df[, c("b", "a")]))           # numbers, no error, wrong
```

A formula fit now rejects a non-data.frame `newdata` naming the columns it wants. The
name-matching guarantee is only available from a named object, so requiring one is the
only way to keep the documented promise; accepting a matrix with matching `colnames` was
considered and rejected as a second, near-invisible rule. Matrix-fitted models are
unaffected and stay positional. `plot(type = "fit")` routes through the same helper and
inherits the rule.

## The remaining fixes

| # | Where | Defect |
|---|-------|--------|
| 2 | `predict_rsymbolic2.R` | A data-free expression (a bare constant — usually the simplest Pareto member) returned length 1 for any `nrow(newdata)`, against the documented "one per row". Python was already correct via `np.broadcast_to`. Now recycled; any other length is an error. |
| 4 | `predict_rsymbolic2.R` | `inf`/`nan` — the tokens `"%.6g"` emits for a non-finite constant — parse as names in R and raised `object 'inf' not found`. Every other surface already recognised them: the R package's own `tree_plot.R:32,64`, Python's `_eval_namespace()`, and the web GUI's `predict.js:72-73`. R's `predict()` was the sole holdout, so `plot(type = "tree", expression = "(x0 * inf)")` drew a string that `predict()` refused. Now bound in the evaluation environment. |
| 6 | `predict_rsymbolic2.R` | A non-numeric `newdata` column surfaced from inside `eval()` as "non-numeric argument to binary operator". Now checked in `design_matrix()`. |
| 5 | `__init__.py` | `_UNARY_OPS` was defined twice at module scope — a set for operator validation and a tuple for tree drawing — and the second silently won, so validation read the drawing constant. Members coincided, so nothing was broken yet; adding an operator to one would have broken the other silently. One definition now. |
| 7 | `summary_`/`as_data_frame_`/`print_rsymbolic2.R`, `__init__.py` | A fit with no recommendation (`best_index` `NA`/`None`) propagated the missing value: `summary()` died with "missing value where TRUE/FALSE needed", `as.data.frame()` returned an all-`NA` column, `get_best()` raised a `TypeError`. Not reachable through the current bridge (`NA_INTEGER` is only produced for an empty front, which is guarded upstream), but each guard is one line. |
| 8 | `__init__.py` | `to_pandas()` lacked the `recommended` column that its R counterpart `as.data.frame()` has. Both now return `complexity, loss, score, recommended, expression`. |

## Deliberately not fixed

`symbolic_regression.formula()`'s first parameter is `formula`, not the generic's `X`, so
`symbolic_regression(X = y ~ a, data = df)` fails with `argument "formula" is missing`.
`tools::checkS3methods("rsymbolic2")` reports nothing, positional calls (the only
realistic usage) work, and R's own `aggregate.formula` has the same shape. Renaming the
parameter to `X` would make both the signature and its roxygen block read worse for no
practical gain.

## Verification

Windows 11 (R 4.6.0 / Rtools45) and Ubuntu 24.04 (WSL, R 4.3.3) agree exactly:

| | Windows | Ubuntu |
|---|---|---|
| `pytest python/tests` | 63 passed | 61 passed, 2 skipped (no pandas in the WSL venv) |
| R testthat | FAIL 0 / SKIP 27 / PASS 265 | FAIL 0 / SKIP 27 / PASS 265 |

Python was 58 before, so +5 tests cover the new contract; the R suite gained 6 test blocks.

Two harness traps, both pre-existing and worth stating once:

- **Run pytest from the repo root, not `python/`.** From `python/`, the source directory
  shadows the installed package and has no compiled `_core`, so collection fails outright.
- **The R suite needs the package namespace**, because `test-plot.R` calls the unexported
  `expr_tree_layout()` and `TREE_FILL` bare. `library(rsymbolic2)` + `test_dir()` reports
  5 spurious errors; pass `env = new.env(parent = asNamespace("rsymbolic2"))` (or use
  `testthat::test_local()`, which additionally sets `NOT_CRAN` and so runs the 27
  `skip_on_cran` tests, reaching 315 passes).

`R CMD check --no-manual` on the built tarball reports **`Status: OK`** — no ERROR, no
WARNING, no NOTE. In particular `checking S3 generic/method consistency ... OK`, which is
the check that would have flagged the deferred item above.

Note that check must be run on a **built tarball**, not the source directory: `DESCRIPTION`
carries only `Authors@R`, and the derived `Author`/`Maintainer` fields that check demands
are generated by `R CMD build`. Checking the directory fails with a misleading
"Required fields missing or empty".
