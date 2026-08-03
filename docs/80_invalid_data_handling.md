# 80 — Invalid and degenerate data across the three interfaces

What each shipped interface (web GUI, R, Python) should do when it is handed data that
symbolic regression cannot meaningfully be run on, what it actually did, and the fixes.

This follows docs/74, which closed the *first* layer of that question (NaN/Inf, factors,
negative counts). Every case below was found by probing the interfaces after those fixes,
so this document is about what docs/74 did **not** reach.

## 1. The principle

Bad data splits into three kinds, and each kind wants a different answer. Mixing them up is
the whole failure mode: the engine is happy to run on any of them and return a
plausible-looking number.

| Kind | Example | Right answer | Why |
|---|---|---|---|
| **Unusable** — no defensible reading | `X` has no columns; all weights zero; `y` is a matrix of several outputs | **Refuse, naming the argument** | Any result would be arbitrary. There is nothing to fall back on and no reading to guess. |
| **Degenerate** — a legal dataset the search cannot say anything about | constant `y`; a constant feature column; a scale that overflows the loss | **Run, but warn** | The run is well-defined and the user may know exactly what they are doing (a constant column is common in a wide table). But the *result* does not mean what it looks like, and nothing else on screen says so. |
| **Coercible** — a different spelling of good data | a logical `0/1` column; a `(n, 1)` column vector for `y` | **Accept silently** | Refusing here is a false rejection, not a safety feature. |

Two rules follow from the table and are applied throughout:

1. **Refusal must happen before the search, not during it.** The engine maps a non-finite
   prediction to an infinite loss *per candidate*, so bad data does not stop a run — it
   starves it, and the loss it reports can still look ordinary. A guard that fires mid-search
   is a guard that fires too late to be believed.
2. **The three interfaces must agree.** A dataset accepted by Python and refused by R is a
   bug in one of them regardless of which behaviour is better, because users move code and
   files between them.

Where the guard lives follows docs/74's layering: the **binding (C++)** holds the guard that
must hold for *every* caller including WASM; the **R/Python layer** repeats it only to make
the message name the user's argument.

## 2. What the interfaces actually did

Probed with `generations=20, n_populations=2, population_size=10, seed=1` on a
12-point `y = 2x + 1`. R and Python agreed on every row (they share the core), so one
column covers both.

| Case | R / Python before | Web GUI before | Verdict |
|---|---|---|---|
| NaN/Inf in `X`/`y` | error | non-numeric column dropped + named | OK (docs/74) |
| 0 rows, length mismatch, factor/character | error | n/a | OK (docs/74) |
| negative counts | error | error | OK (docs/74) |
| **`X` with 0 columns** | **accepted → `expr = "1"`** | cannot occur (needs ≥1 feature) | **bug** |
| **all-zero `weights`** | **accepted → `loss = 0` for everything, `expr = "0.996372"`** | n/a (not exposed) | **bug** |
| **`y` shaped `(6, 2)` with a 12-row `X`** | **accepted (Python `ravel()`d it)** | n/a | **bug (Python)** |
| **constant `y`** | accepted, silent; 13-node expression, `R² = "—"` | accepted, silent | **warn** |
| **constant feature column** | accepted, silent | accepted, silent | **warn** |
| **scale overflow (`y × 1e200`)** | accepted, silent; `sst = Inf`, `loss = 1.2e21` | same | **warn** |
| **logical `X`** | **R: error. Python: accepted.** | n/a | **inconsistent** |

Three findings deserve spelling out.

### 2.1 All-zero weights silently voided the search

```python
rs.symbolic_regression(X, y, weights=np.zeros(12))
# expr='0.996372'  loss=0.0  sst=0.0
```

`weights` was checked for finiteness and non-negativity — but not for whether any weight
was left. With every weight zero the weighted SSE is identically `0`, so **every candidate
ties at a perfect loss**, and the reported expression is simply whichever one the tournament
happened to hold. `loss = 0.0` is the most confidence-inspiring number the API can print,
and here it means the opposite.

### 2.2 A multi-output `y` was flattened instead of refused (Python)

```python
rs.symbolic_regression(X_12_rows, np.arange(12.0).reshape(6, 2))
# accepted; fits the interleaved flattening of two different targets
```

`y_arr = np.asarray(y, dtype=float).ravel()` was written for the harmless `(n, 1)` column
vector, but it flattens *any* shape. When the flattened length coincides with `nrow(X)` the
length check passes and the search fits the interleaving of two unrelated series. This is
the same failure class as docs/74's factor response — a wrong answer with nothing anywhere
to suggest it. R was never exposed: `length()` on a matrix counts all cells, so its existing
`nrow(X) != length(y)` check already refused this.

### 2.3 R refused logical columns that Python accepted — and that R itself accepted when mixed

```r
symbolic_regression(matrix(c(TRUE, FALSE), 12, 1), y)   # Error: X must be numeric
symbolic_regression(data.frame(a = 1:12, b = rep(c(TRUE, FALSE), 6)), y)   # works
```

`is.numeric()` is `FALSE` for a logical matrix, so the guard docs/74 relied on to catch
character data also caught `0/1` indicator columns. It only fired on a *purely* logical `X`,
because `as.matrix()` on a mixed data frame promotes logicals to numeric — so the same
column was accepted or refused depending on what sat next to it. Python has always accepted
it (`np.asarray(..., dtype=float)`).

## 3. The fixes

### Refusals (new errors)

| Guard | Message | Where |
|---|---|---|
| `ncol(X) == 0` | `X must have at least one column` | R layer, Python layer, R bridge, Python bridge (WASM already had it) |
| weights not finite / negative / summing to 0 | `weights must not be all zero; their sum must be positive` | R layer, Python layer, all three bridges |
| `y` with more than one column | `y must be a 1-D array (a single target)...` | Python layer |

The bridge-level guards are the load-bearing ones, per docs/74: they hold for any caller.
The R/Python layer repeats each so the message names the user's argument rather than
surfacing a C++ exception.

### Warnings (new; the run still happens)

Emitted as `warning()` in R and `warnings.warn(..., UserWarning)` in Python — not
`message()`. The existing large-row advisory stays a `message()`, and the split is
deliberate: an advisory suggests a faster way to do what you asked, a warning says the
answer you are about to get does not mean what it looks like.

Three exact, checkable degeneracies — no invented thresholds:

1. **Constant `y`** (weighted variance `0`). Also covers the single-row case, whose variance
   is zero by construction. This is the same condition that already makes `R²` print `—`,
   so the warning finally says *why*.
2. **Constant feature column(s)**, named as `x0 (colname)` so the name matches the fitted
   expression's variables.
3. **Non-finite SST** — `y`'s scale overflows the sum-of-squares loss, so both the loss and
   `R²` are meaningless. Checked with the same weighted formula the core uses.

### Acceptance (removed false rejection)

R promotes a logical `X` to numeric before the `is.numeric()` check, matching Python and
matching what R already did for a logical column mixed with a numeric one.

### Web GUI

The GUI cannot reach the refusal cases (it requires ≥1 feature, exposes no weights, and
builds `y` from one column). It could reach all three degeneracies and said nothing, so
`renderIntakeNotice()` — which already reports non-numeric columns and skipped rows — gained
a third line naming constant columns and a fourth for a column whose scale overflows.

Constancy is a property of the parsed file, which is exactly the contract that notice
already documents ("rebuilt only on load"), so it needs no new update path. The notice names
the columns and states the consequence for each role (a constant target makes the whole run
meaningless; a constant feature merely cannot explain anything), because the default target
is the last numeric column and the user may not connect the two facts themselves.

## 4. Why the search cannot have moved

No `diag_search_digest` run is quoted here, and that is deliberate: **not one core
translation unit changed.** `git diff` over `src/rsymbolic/**` and every core `.cpp` is
empty; the whole change lives in the three bridges and the two binding layers, and every
guard runs *before* `run_evolution()` is called. A digest comparison would be comparing a
file against itself.

The three ways a binding-only change could still move the search were each checked:

- **A new refusal on a path that used to succeed.** Every refusal added here fires only on
  input that already produced a meaningless result (no columns, zero total weight,
  multi-output `y`). The bridge weight checks reject non-finite and negative weights too,
  but the R and Python layers already did, so no previously-successful call is affected.
- **A new acceptance changing what reaches the core.** The logical-`X` promotion admits data
  that previously raised an error, so it adds runs rather than altering any.
- **A degeneracy check mutating its input.** All three are read-only reductions over `X` and
  `y`; nothing is written back.

## 5. Results

Everything below was run after the change, on both mandatory platforms.

| Suite | Windows 11 (Rtools45 / MinGW-UCRT) | Ubuntu 24.04 (WSL2, GCC 13) |
|---|---|---|
| C++ core, `ctest` | **30/30 passed** | **30/30 passed** |
| R, `testthat` (`NOT_CRAN=true`) | **382 passed, 0 failed, 0 skipped** | **382 passed, 0 failed, 0 skipped** |
| Python, `pytest` | **96 passed** | 87 passed, 9 skipped (matplotlib absent) |
| `R CMD check --no-manual` | **Status: OK** | — |
| WASM parity gate (`web/wasm/test/parity_test.cjs`) | **PARITY TEST PASSED** | — |

The R and Python probe scripts from §2 were re-run against the built packages, and the two
interfaces now agree on every row: the three unusable cases raise, the three degenerate
cases warn and still return, and a logical `X` runs.

One existing test had to change, and it is worth recording why rather than treating it as
noise: `test-print-summary.R`'s *"summary reports NA R-squared for a constant target
(SST = 0)"* is built on a zero-variance target, so it now trips the new warning. It was
updated to `expect_warning(...)` rather than silenced — the `NA` that test asserts *is* the
consequence the warning describes, and the two belong together.

The GUI notice was verified in a real browser (`web/serve.py` + Playwright) on a table
carrying a constant column, an overflowing column and a non-numeric cell at once:

> 1 of 4 columns is not numeric and cannot be modelled: "note" (row 1 is "ok"). … Column
> "flat" is constant (the same value in every row). As the target that makes the run
> meaningless — every constant fits perfectly, so R² is undefined; as an input it cannot
> explain anything and only enlarges the search. Column "huge" is on a scale whose sum of
> squares overflows to infinity. …

and on a clean two-column table, where the notice stays hidden and a run still recovers
`x·2+1` at `loss = 1.03e-29`.

`web/wasm/test/parity_test.cjs` gained three assertions for the new weights guard
(all-zero refused, negative refused, a single positive weight accepted), alongside the
count and NaN/Inf guards docs/74 put there.
