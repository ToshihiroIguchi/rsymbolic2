# 74 — R and Python interface audit

Audit of the two shipped bindings — the R package (`R/*.R`, `src/rsymbolic2_r.cpp`) and
the Python package (`python/rsymbolic2/__init__.py`, `python/src/rsymbolic2_py.cpp`) —
following the C++ core audit in docs/73.

The theme is uniform: **the bindings validated their options carefully and their data
barely at all.** `weights`, `batch_size`, `n_threads`, `warmup_maxsize_by` and
`dimensional_constraint_penalty` each had a finiteness or range check. `X`, `y`, and every
count-like integer had none. Every finding below is a consequence of that asymmetry, and
every one was reproduced before being fixed.

The search itself is untouched: `diag_search_digest` is byte-identical before and after on
both platforms.

## Fixed

### 1. A negative count aborted the host process (most severe)

```r
symbolic_regression(X, y, population_size = -1L)
# terminate called after throwing an instance of 'std::length_error'
#   what():  vector::reserve
```

The R session died outright — not an error, a process abort. Python behaved identically,
taking the interpreter with it.

Mechanism, in three steps:

1. `population_size` is a `std::size_t` in `SearchOptions`. The binding cast the signed
   `int` straight across, so `-1` became `1.8e19`.
2. `initialize_island` calls `population.reserve(opts.population_size)`, which throws
   `std::length_error` on a request that large.
3. `initialize_island` runs inside `#pragma omp parallel for`. **An exception may not
   propagate out of an OpenMP structured block**, so it reached `std::terminate`.

Step 3 is what turns a bad argument into a lost session, and it is why the fix is layered
rather than placed in one spot:

- **The bindings validate while the value is still a signed `int`** — the only place the
  sign still exists. `rsymbolic2_r.cpp` and `rsymbolic2_py.cpp` both reject
  `population_size`, `generations`, `tournament_size`, `max_nodes`, `max_depth` and
  `n_populations` below 1. This is the load-bearing guard: it holds for any caller,
  including the WASM binding.
- **The R and Python layers validate too**, so the message names the argument instead of
  surfacing a C++ exception.
- **`run_evolution` keeps a `>= 1` check** ahead of the parallel region, for the zero case
  and for any future binding that forgets.

A guard inside `run_evolution` alone would *not* have been enough, which is worth
recording: by the time the value reaches `SearchOptions` it is already an unsigned
`1.8e19`, indistinguishable from a deliberate huge request.

`population_size = 0` was less dramatic but also wrong — it produced an empty population
and surfaced much later as `HallOfFame::best() called on an empty hall of fame`, a message
that says nothing about the actual mistake.

### 2. NaN/Inf in `X` or `y` produced a plausible-looking, meaningless result

Neither binding checked the data for finiteness, though both checked `weights`. The core
maps a non-finite prediction to an infinite loss *per candidate*, so a single bad point
does not stop the run — it silently starves it.

```r
Xa <- X; Xa[4, 1] <- NA
symbolic_regression(Xa, y)
# no error; expression "0.999758", loss 265.2632
```

The dangerous part is the loss: **265 looks like an ordinary number.** Nothing in the
result says the fit is meaningless. (A NaN in `y` was slightly better-behaved, reporting
`loss = Inf`, but still returned a result rather than an error.)

Both layers now reject non-finite `X` or `y` up front, matching the check `weights` has
always had.

### 3. A factor `y` trained on level codes (R only)

```r
symbolic_regression(X, factor(c(100, 20, 3, ...)))
# no error; expression "2.05"
```

`as.numeric()` on a factor returns its level **codes**, not the values it prints — so this
fitted `c(1, 2, 3)` and dutifully found their mean, 2.05. A wrong answer with nothing
anywhere to suggest it.

`X` was already protected by accident: `as.matrix()` on a data frame containing a factor
yields a *character* matrix, which the existing `is.numeric(X)` check catches. The response
had no equivalent guard. Both `symbolic_regression.default` and the formula method now
reject a factor with a message pointing at `as.numeric(as.character(y))`.

### 4. `cd python && pytest` tested nothing (packaging)

`python/rsymbolic2/` holds only `__init__.py`; the compiled `_core` extension is produced
by CMake into site-packages at install time. With `python/` on `sys.path` ahead of
site-packages, `import rsymbolic2` found the source package and failed at
`from ._core import ...`:

```
ModuleNotFoundError: No module named 'rsymbolic2._core'
```

which reads like a broken build rather than a shadowed import. (A stray
`python/.pytest_cache/` in the tree shows this had been hit before; it was hit again during
this audit.)

Two invocations, two causes, so two fixes:

- `pyproject.toml` sets `--import-mode=importlib`, stopping pytest from prepending the
  rootdir. This covers plain `pytest`.
- `python/conftest.py` removes the source directory from `sys.path`. This covers
  `python -m pytest`, where **CPython itself** — not pytest — puts the working directory at
  `sys.path[0]`, so the pytest setting cannot help. It also converts a still-missing
  install into a message that says `pip install ./python`.

## Checked and found correct

- **Column-major transposition** in the Python binding: `py::array_t<double, c_style |
  forcecast>` normalises C-order, Fortran-order and non-contiguous slices to a contiguous
  buffer before the transpose, so no layout silently trains on transposed data.
- **`predict()` in both languages** reproduces the engine expression. The precedence
  concern from docs/73 does not apply here: `to_string()` fully parenthesises every node
  (`neg` → `(-a)`, `square` → `(a ^ 2)`, `inv` → `(1 / a)`), so neither R's parser nor
  Python's `^`→`**` substitution can misgroup. Correct by construction, not by luck.
- **Variable indexing** is 0-based (`x0`, `x1`, …) consistently across engine, R and Python.
- **GIL** is released around `run_evolution` (`py::gil_scoped_release`), so a long search
  does not block other Python threads.
- **Argument forwarding**: all 38 Python parameters reach `symbolic_regression_cpp` in the
  order the C++ signature expects — nothing accepted-but-ignored.
- **`to_sympy`/`to_latex` index substitution**, including the ≥10-feature `x_{1}` vs
  `x_{10}` collision case, and underscore escaping in supplied names.
- **Empty / single-member Pareto fronts** in `print`, `summary`, `plot` and
  `as.data.frame` are guarded.

## Not changed

`tournament_size = 0` and `max_depth = 0` are now rejected alongside the others for
consistency, but neither was dangerous: the core already mapped `tournament_size = 0` to 1,
and `max_depth = 0` merely produced degenerate one-node trees.

## Verification

- `diag_search_digest`: byte-identical before/after on Windows and Ubuntu — no search
  behaviour changed.
- Windows: ctest 30/30, testthat 368, pytest 88.
- Ubuntu: ctest 30/30, testthat 368, pytest 79 (+9 skipped for optional deps).
- Every defect above was reproduced first and re-run after the fix; the
  `population_size = -1` case now returns an R error with the session intact.
- New tests: `test-validation.R` (factor response, non-finite data, non-positive counts)
  and `test_rsymbolic2.py` (parametrised over NaN/±Inf and over all six counts at 0 and −1).
