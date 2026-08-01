# 71. `square(x0)` is not notation: square, inv and neg now render as operators

**Date:** 2026-08-01
**Status:** implemented; verified on Windows and Ubuntu (WSL).
**Change:** one `switch` in `tree.hpp::to_string()`. The unary `Square`, `Inv` and `Neg`
nodes render as `(a ^ 2)`, `(1 / a)` and `(-a)` instead of `square(a)`, `inv(a)` and
`neg(a)`, on every surface that shows or copies an expression string — R, Python, the WASM
module and the web GUI alike. The three equation-tree renderers label a unary minus `-`
rather than `neg`, for the same reason.
**Search behaviour is unchanged on every platform.** This is a renderer change: no tree, no
cost, no default and no RNG draw is touched, and the PySR default-parity rule is not
involved.

---

## 1. The complaint

A result reads

```
(square(x0) - (x1 * (x1 - x0)))
```

`square` is the *engine's* name for the node. It is not how anyone writes a square, and it
is not a function in any of the environments a copied equation is likely to land in —
SymPy included, where `sympify()` accepts it as an **undefined applied function** and
returns something that simplifies and prints while meaning nothing (docs/70 §1). docs/70
answered that by adding a second string, `sympy`. The remaining question is why the
*first* string spells it that way at all.

There is no reason. `^`, `/` and unary minus are already in this grammar; `to_latex()` has
always rendered the same nodes as `a^{2}` and `-a`. The call form was an artefact of
`to_string()` rendering every unary the same way — `unary_name(op) + "(" + arg + ")"` —
which is right for `sin`, `exp` and `sqrt`, because those *are* the notation, and wrong for
the three operators that have symbols.

| node | was | is |
|---|---|---|
| `Square` | `square(a)` | `(a ^ 2)` |
| `Inv` | `inv(a)` | `(1 / a)` |
| `Neg` | `neg(a)` | `(-a)` |

## 2. Why `^` and not `**`

`^` is what the rest of the string already uses: a genuine `Pow` node has always printed as
`(a ^ b)`, and the three `predict()` parsers (R, Python, JavaScript) each accept that
grammar. Rendering `Square` as `**` would introduce a second power spelling into a format
that has one.

`**` remains exclusively the `sympy` field's business — see §6, which is where that field's
justification moved.

## 3. Why the rewrites are exact, not merely close

Printing these nodes as operators is only legitimate if reading the string back gives the
same numbers. It does, and not by luck:

* **`(a ^ 2)`.** SR.jl's `safe_pow`, transcribed literally in `dual.hpp`, guards only
  `y < 0 && x == 0` when the exponent is an integer. With `y = 2` that guard cannot fire, so
  `pow(x, 2)` is plain `x*x` for **every** double — the negatives, the signed zeros, ±Inf
  and NaN included — which is exactly what `square(x)` computes.
* **`(1 / a)`.** `recip(a)` is literally `1.0 / a` and `Div` is `a / b`, both unguarded
  (`dual.hpp`: "UNGUARDED, exactly like operator/"). The same IEEE operation, so the same
  signed zeros and infinities.
* **`(-a)`.** Exact by definition.

The same holds through the three `predict()` wrappers. Their `^` maps to the host language's
power (`^` in R, `**` in Python and JavaScript), which parts from `safe_pow` only at `0 ^ -1`
and `(-Inf) ^ 0.5` (docs/70 §1.2, commit c490484) — neither reachable with an exponent of 2.
`benchmarks/diag_structural_audit.R` rebinds `^` to the transcribed `engine_pow` and is
likewise unaffected.

This is asserted rather than argued:
`test_tree_eval.cpp::test_operator_renderings` pins the renderings and then compares
`square(x)` against `pow(x, 2.0)`, and `recip(x)` against `1 / x`, over
`{-3, -0.0, +0.0, 2.5, +Inf, -Inf, NaN}`.

## 4. The parentheses are load bearing

Every form is fully parenthesized, like the binaries, so the printer needs no precedence
tracking and no context can be misread. For `neg` that is not a style choice:

```
square(neg(a))   ->   ((-a) ^ 2)      correct
                 ->   (-a ^ 2)        WRONG: Python and R parse -(a^2)
```

A unary minus binds *looser* than `**`/`^` in both languages, so the unparenthesized form
computes `-(a²)` where the tree means `(-a)² = a²`. The two differ in sign for every
non-zero `a`. `test_operator_renderings` pins this case specifically.

## 5. What the string loses, and what it does not

A `Square` node and a `Pow` node over the constant `2` now print alike, as do an `Inv` node
and a `Div` under a `1`. That is a property of the string, not a loss of information anyone
had: each pair computes the same function, and the string has never carried node identity —
complexity is counted from the tree, and it is displayed beside the equation. Nothing reads
structure back out of the string except `predict()`, which only evaluates.

**Backward compatibility.** `square(...)`, `inv(...)` and `neg(...)` are still *accepted*
everywhere they ever were: the names stay in `op_names.hpp` (so macro bodies like
`exp(neg(square(x)))` are unchanged, and that is still the spelling a macro body must use),
and the R, Python and JavaScript `predict()` namespaces keep their bindings so an expression
string saved by an earlier version continues to evaluate. Only the *renderer* stopped
emitting them.

## 6. Side effect: `expression` is now sympify-clean, and the export's reason narrows

The three names were exactly the tokens `sympify()` mis-parsed into undefined applied
functions. With them gone, every remaining token in an `expression` string is either an
operator SymPy knows or a function that exists in SymPy under the same name (`exp`, `log`,
`sin`, `cos`, `sqrt`, `tanh`, `erf`, `sinh`, `cosh`, and `abs` via `Symbol.__abs__`). So
`sympify(expression)` is now correct on its own — `test_sympy_export.py` asserts it over a
whole fitted front.

The `sympy` field is **not** thereby redundant, but its justification changes and every doc
surface has been updated to say the new one:

> The engine's power operator is `^`, and Python reads that as **xor**. `eval()`,
> `parse_expr()`, `lambdify()` and NumPy all compute the wrong function, silently, for any
> expression containing one — and since a squaring now prints as `(a ^ 2)`, that is most of
> them. `sympify()` alone is the exception, because it passes `convert_xor=True`.

That is a narrower claim than docs/70's ("`sympify()` itself is wrong on these strings") and
a sharper one: the failure has moved from SymPy to everything that is not SymPy.

## 7. Surfaces touched

| surface | before | after |
|---|---|---|
| `expression`, `expression_simplified` (R/Python/WASM) | `square(x0)`, `inv(x0)`, `neg(x0)` | `(x0 ^ 2)`, `(1 / x0)`, `(-x0)` |
| web GUI hero equation, Pareto table, CSV, copy buttons | as above | as above |
| equation tree (R `plot(type="tree")`, Python, GUI) | `neg` node | `-` node |
| `latex` / `latex_simplified` | `x_{0}^{2}`, `-x_{0}` | unchanged |
| `sympy` / `sympy_simplified` | `x0**2`, `1/x0`, `-x0` | unchanged |

The equation-tree renderers needed no *parser* change: all three already mapped `^` (R's
`BINARY_LABEL`, JavaScript's `BINARY_LABEL`, Python's `ast.Pow`/`ast.BitXor`) and already
handled a unary minus over a non-constant, because `%.6g` negative literals and `Pow` nodes
could always reach them. What changed is the label on that unary-minus node: it read `neg`
while the equation beside it read `-`, and one operator under two names on one screen is the
defect those glyph tables exist to prevent. A one-child `-` is unambiguous next to the
two-child subtraction.

The GUI keeps the `square`, `inv` and `neg` operator pills — the *input* names are
unchanged — and their tooltips now say what results will print.

## 8. Verification

* Windows: standalone `ctest` 30/30; R `testthat` 0 failures; `pytest` 73 passed; WASM
  parity test PASSED, including that the display simplifier still introduces a squaring
  (detected as `" ^ 2)"`, so the SymPy check stays exercised) and that **no** expression
  string on the front spells `square(`, `inv(` or `neg(`.
* Ubuntu (WSL): standalone `ctest` 30/30; R `testthat` 0 failures; `pytest` 64 passed,
  9 skipped (matplotlib/pandas absent from that venv — the SymPy export tests ran).
  The WASM module is built once, on Windows, and is platform-independent.
* New/updated tests: `test_tree_eval.cpp` (new case, §3 and §4), `test_display_simplify.cpp`,
  `test_simplify.cpp` and `test_macro_op.cpp` (goldens), `test-sympy.R` and
  `test_sympy_export.py` (both now assert the engine names never appear in `expression`, and
  the Python one additionally asserts `sympify(expression)` is clean), `parity_test.cjs`.

An end-to-end result, for the record:

```
10 | ((((-x0) ^ 2) * 3) - (1 / (x0 / 2)))
sympy: (-x0)**2*3 - 1/(x0/2)
```
