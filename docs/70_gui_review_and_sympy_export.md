# 70. GUI review fixes, and a SymPy-parseable export

**Date:** 2026-07-31
**Status:** implemented; verified on Windows and Ubuntu (WSL).
**Change:** new `expression/sympy.hpp` in the shared core plus `sympy`/`sympy_simplified`
in all three bindings (R, Python, WASM); an advisory large-data message in the R and Python
entry points; nine corrections to the web GUI. Tests: `standalone/tests/test_to_sympy.cpp`,
`python/tests/test_sympy_export.py`, `r-package/.../test-sympy.R`, new assertions in
`web/wasm/test/parity_test.cjs`.
**Search behaviour is unchanged on every platform.** Everything here is display, export or
advisory; no default moved, and the PySR default-parity rule is untouched.

This note records a review of the web GUI's controls and of the result strings the library
hands back, the two things that turned out to be wrong, and what was measured rather than
assumed.

---

## 1. The export problem: `square(x0)` is not Python

`to_string()` produces the frozen, evaluatable round-trip form — `predict()` parses it back
(`parse_expression.hpp`), so it cannot change. It is also what every surface shows and
copies. Four of its tokens look like they would not survive SymPy. **Measured against SymPy
1.14, only three of them actually do not:**

| `to_string()` token | `sympify()` result | verdict |
|---|---|---|
| `square(x0)` | `square(x0)` — an **undefined applied function** | broken, silently |
| `inv(x0)` | `inv(x0)` — undefined applied function | broken, silently |
| `neg(x0)` | `neg(x0)` — undefined applied function | broken, silently |
| `(x0 ^ 2.0)` | `x0**2.0` (`Pow`) | **fine** |
| `abs(x0)` | `Abs(x0)` | **fine** |

The last two were expected to break and do not. `^` survives because `sympify()` passes
`convert_xor=True` by default — `parse_expr()` on its own does not, which is where the
"`^` is xor" belief comes from. `abs` survives because it is a Python builtin and
`Symbol.__abs__` returns `Abs`; it is a language mechanism, not a parser feature.

The three that do break are the bad case: `sympify()` **does not raise**. It manufactures
an undefined function and returns an expression that differentiates, simplifies and prints
without complaint while meaning nothing. An error would have been better.

### 1.1 What was built

`to_sympy()` in `rsymbolic/expression/sympy.hpp` — a second precedence-aware printer beside
`to_latex()`, same postfix walk, same shape:

| operator | rendering |
|---|---|
| `square(a)` | `a**2` (base parenthesized below atom: `-3**2` is `-9` in Python) |
| `inv(a)` | `1/a` (Mul precedence, so `1/x*y` and `z/(1/x)` both come out right) |
| `neg(a)` | `-a` |
| `pow` | `**` (both operands parenthesized below atom — `**` is right-associative) |
| everything else | its own name; all nine remaining unary ops are SymPy functions already |
| `NaN` / `±Inf` | `nan` / `oo` / `-oo` (SymPy's spellings; `float('nan')` would not parse) |

`abs` is emitted as `abs`, **not** `Abs`: the lowercase form works under `sympify()` *and*
under plain `eval()`/NumPy, where `Abs` is undefined. The result is therefore valid input to
`sympify()`, `parse_expr()`, `eval()` and `lambdify()` alike.

Variables stay `x0`, `x1`, …, and feature names are **not** substituted by default — unlike
`to_latex()`, which does substitute them. A column name is free text, and `flow rate` is not
a Python identifier; a default that produced unparseable output would defeat the point.
`to_sympy(variable_names=)` opts in and rejects anything that is not an identifier.

### 1.2 The caveat that must travel with it

This is the **mathematical** form, not the engine's. rsymbolic2's operators are
domain-guarded (`docs/69`): `sqrt`, `log` and `pow` return NaN outside their domain where
SymPy returns a complex or symbolic value. `predict()` evaluates; this differentiates,
simplifies and typesets. Said in every doc surface (R `?to_sympy`, Python `.sympy()`, the
GUI tooltip, the header).

### 1.3 Verification

`test_to_sympy.cpp` pins the syntax (48 checks). What C++ cannot check is whether SymPy
*agrees*, so `python/tests/test_sympy_export.py` fits a real front over
`square/inv/neg/sqrt/abs/exp/log/sin/cos` × `add/sub/mul/div/pow` and, for every member:

* asserts `sympify()` produces **no** `AppliedUndef` atoms;
* `lambdify`s the parsed expression and compares it against `predict()` on the training
  inputs, `rtol=1e-8` (NaN points excluded and counted — see §1.2).

Both pass. The WASM parity test asserts the same no-`square()/inv()/neg()/^` property on the
front the GUI actually reads, and — usefully — that the *display simplifier* introduced
`square()` on its own, from an operator set that did not contain it (`x*x → square(x)`). So
the equation on screen can spell an operator the user never enabled, which is exactly when a
copied string is most likely to be pasted somewhere and quietly mean the wrong thing.

### 1.4 Surfaces

`pareto_front` gains `sympy` and `sympy_simplified` in R, Python and WASM, mirroring
`latex`/`latex_simplified` exactly. R gets `to_sympy()` (generic + method), Python gets
`.sympy()`, the GUI gets a **SymPy** button in the Best-formula copy row and a `sympy` column
in the Pareto CSV. SymPy is **not** a dependency anywhere — these are strings.

---

## 2. Large data: what each surface does, and the one gap

Reviewed against `docs/59` (browser) and `docs/65` (memory).

| | web GUI | R / Python |
|---|---|---|
| row ceiling predicted before the run | yes (`maxRowsForBrowser`) | n/a — no fixed heap |
| forced deterministic sampling over the ceiling | yes | no |
| slow-run warning | yes | **was: nothing** |
| batching (`O(batch_size)` evaluation) | yes, in Search settings | yes, default off |
| timeout / max_evals | yes | yes |
| progress + ETA | yes | progress callback |

The web side was already complete and is the surface where the constraints actually bite.
The gap was R/Python: passing a million rows started a multi-day run in silence. Batching is
the lever and it is off by default **because PySR's is** — parity is not ours to trade — so
the fix is to say so, not to change it:

* R: `message()` above 10,000 rows when `batching = FALSE`. `suppressMessages()` silences it.
* Python: `warnings.warn(UserWarning, ...)`. The standard filters silence it.

10,000 is not a new measurement; it is the figure the batching documentation already named
("for most problems fewer than ~10,000 rows are enough without batching"), now a named
constant in both languages so the prose and the message cannot drift. Tests assert the
message fires, is suppressible, does not fire when batching is on or below the threshold,
and that **the result is identical with and without it**.

---

## 3. Web GUI corrections

Two were defects, the rest were consistency or discoverability.

### 3.1 The empty-state button discarded the user's data (defect)

`#placeholder-run-example` — the largest primary control on the page — unconditionally ran
`loadTable(EXAMPLES[0])` then `run()`, and its card stayed visible until the *first* result.
So after dropping a CSV, the most prominent button on screen still said "Try an example ▶"
and pressing it replaced the table just loaded. Split into two buttons swapped by
`body.has-data`: the example loader before there is data, **Run** afterwards. This also gives
the sidebar's top-to-bottom reading order a terminus — Run otherwise lives only in the pinned
header, diagonally far from where the eye leaves the configuration rail.

### 3.2 "Reset to PySR defaults" did not restore PySR defaults (defect)

`resetDefaults()` covers `DEFAULTS` (the numeric fields) and `CHECKBOX_DEFAULTS`, which is
`{batching: false}` alone. It deliberately does **not** touch the sidebar's high-accuracy
opt-ins — `linear_scaling`, `eval_cache` — on the sound principle that a button must not
rewrite state the user cannot see from where they clicked. But `linear_scaling` *is* a PySR
divergence, so a button promising "every field back to its shipped PySR-parity default" while
leaving it on was the misleading combination. Fixed in the name, not the scope: **"Reset
these fields"**, with the boundary spelled out in the tooltip. The scope itself was right.

### 3.3 An operator-library reset

Operators are the one problem input built up over a session, and the only one the app itself
changes: `applyExampleOps()` ticks the operators an example's formula needs. Measured, not
assumed — it only ever **adds** (`setOpChecked` never unchecks), so nothing is destroyed; but
an operator deliberately switched off coming back is still a change nobody made. Neither
existing reset reached it (§3.2, and the rail's "Use defaults" only appears when the settings
restored *on arrival* differ). A scoped `reset` now sits on the Operator library heading,
**present-but-disabled** when the selection is already the shipped one — matching `#run-btn`
and `#print-btn`, and doubling as a statement of the current state.

### 3.4 Features became pills; operators and features are the same decision

Operators were toggle pills, features plain checkboxes, one card apart. Both are "which
subset defines the search", and an unticked entry in either removes possibilities the search
cannot work around. Unified on pills (presentation only — `currentFeatureIndices()` is
unchanged), with a width cap and ellipsis because a column name is arbitrary text where `+`
and `x²` are not. **all / none** added, because the list is rebuilt fully ticked on every
target change and a wide table otherwise costs one click per column.

### 3.5 A disabled checkbox became a radio pair

When a table cannot fit the fixed heap, sampling is not a preference. That was expressed by
ticking the Sample checkbox and then **disabling** it — a control you cannot untick and that
gives no way to read why. Replaced by `All 150,000 rows` / `Sample [n] rows`, where the
option that is unavailable is the one that is disabled and carries the reason in its own
tooltip. Same pill shape as the operator toggles.

### 3.6 The slow-data threshold now counts cells, not rows

`ROW_WARN_THRESHOLD = 5000` ignored width, so 5,000 × 20 — an order of magnitude more work
than the 5,000 × 2 the constant was calibrated on — passed without a word. Now
`SLOW_CELL_THRESHOLD = 10000` on rows × fitted columns, which preserves the old behaviour
exactly at the 2-column case all the `docs/59` measurements use.

### 3.7 The notice now performs the action it asks for

It used to end "enable Batching in Search settings" — three navigation steps and a modal
section away. There is an **Enable batching** button in the notice itself, absent once
batching is on, and the notice re-renders when batching is changed from the dialog so it
never offers what is already done.

### 3.8 `score` was defined nowhere in the GUI

R (`?symbolic_regression`, `?summary.rsymbolic2`) and Python (`pareto_front` docstring) both
define it: *the drop in log-loss per unit of added complexity against the next-simpler front
member; 0 for the simplest*. The GUI had no tooltip on the `score` column, and the recommend
dropdown said "score = highest score overall", which is circular. Both now carry the same
sentence the packages do. `loss` and `R²` got tooltips at the same time.

### 3.9 Log-loss became a select; the sidebar note shrank

The Pareto card's head held two `<select>`s and one checkbox, all doing the same kind of
thing. `loss axis: log / linear` now matches its neighbours.

The permanent sidebar note — "Single-threaded here — for large or long searches use the R or
Python package" — was removed. The same sentence is already produced by `#data-notice` at
the moment it applies, with the row count and the measured cost in it, and for the
few-hundred-row tables this page is mostly used with it named a cost the visitor never pays.
The link stayed, as a link: this is a demo of a library, and that is worth saying every
session. §3.6 is the other half of this change — dropping the permanent warning is only safe
if the conditional one fires when it should.

---

## 4. What was checked

* `test_to_sympy.cpp` (48), full standalone suite 30/30.
* `pytest python/tests` 73 passed, including the 7 new SymPy round-trip tests.
* R `devtools::test()` — all pass, including 23 new `to_sympy` / large-data checks.
* WASM parity test passes with 5 new assertions; the module was rebuilt (emsdk 6.0.2).
* The GUI driven end to end in a browser: operator reset enable/disable and its restored
  values, the placeholder swap, features all/none, the forced-sampling radio state and its
  tooltip, the batching action and its persistence, the loss-axis select redraw, the SymPy
  copy (`sin(x0) + x1**2` from a displayed `(sin(x0) + square(x1))`) and the CSV column.
  Zero console errors.
