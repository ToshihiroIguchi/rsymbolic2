# `inv` Operator (1/x)

**Date:** 2026-07-23
**Status:** implemented, **off by default** (opt-in via `unary_ops`)
**Scope:** adds one unary operator to the closed operator set; no default changes.

---

## 1. Why

The operator set *is* the search space (CLAUDE.md Benchmarking Requirements: it is the one
shared problem input, not a tuning knob). Before this change, `1/x` cost **three nodes plus
a fitted constant** (`div(c, x)`), while inverse and inverse-square structures dominate the
Feynman corpus. `inv` makes the reciprocal a **single node**, exactly the argument that
justified `square` alongside `pow` (`docs/18` §2.1): it does not add expressiveness, it
lowers the cost of a common structure so the search can reach it in one mutation.

Adding it to the enum costs nothing at search time — mutation samples only the operators
the caller enabled (`mutation.cpp` draws from `space.unary_ops`), and the shipped defaults
are unchanged:

| Interface | Default `unary_ops` |
|---|---|
| R | `c("neg", "exp", "log", "sin", "cos")` |
| Python | `("neg", "exp", "log", "sin", "cos")` |
| Web GUI | same five checked; `inv` appears unchecked under "Power / other" |

**Name compatibility is a requirement, not a coincidence.** SymbolicRegression.jl exposes
`inv` with the same meaning, so a benchmark can hand the identical operator set to both
tools. An operator with no SR.jl counterpart would make the required PySR comparison
impossible, and is therefore not an acceptable addition.

## 2. Semantics

- Value: `inv(x) = 1/x`.
- **Unguarded**, deliberately: `BinaryOp::Div` (`dual.hpp` `operator/`) also divides
  without a guard and relies on the loss finiteness guard to reject candidates that go
  non-finite. `inv` must behave like `div`, not like `sqrt`/`pow` (whose guards exist
  because the *real-valued function itself* is undefined on part of the domain). A zero
  argument yields ±Inf, the loss becomes non-finite, and the candidate is discarded.
- Derivative: `d(1/x) = -1/x²`, implemented once and mirrored verbatim in the three
  evaluation paths (`dual.hpp`, `multi_dual.hpp`, `soa_eval.hpp`), which are asserted
  **bit-identical** to each other by `test_multi_dual.cpp` / `test_soa_eval.cpp`:

  ```cpp
  const double r = 1.0 / a.value;
  return {r, -a.deriv * r * r};      // same left-associative product in all three paths
  ```

- **Naming:** the C++ function is `recip`, not `inv`, because `operator/` in both
  `dual.hpp` and `multi_dual.hpp` already declares a local variable named `inv`. The
  user-facing name is `"inv"` everywhere (`to_string`, `unary_ops`, the GUI checkbox).
- Dimensional analysis (`units/dimensional_analysis.hpp`): `inv` negates the exponents
  (`dim.scaled(-1)`), so `1/s` is `Hz` and `inv(inv(x))` returns the original dimension.
- LaTeX: `\frac{1}{x}` — `\frac` braces group, so operands never need parentheses.
- **No simplifier rules were added.** `inv(inv x) → x` is *not* an admissible rewrite under
  the display-simplifier's floating-point policy (`display_simplify.hpp`): for a subnormal
  `x`, `1/x` overflows to `Inf` and the round trip returns `0` — a finite value changed to
  a different finite value. Constant folding still applies generically via `apply_unary`,
  and `norm_unary`'s `default:` branch passes `inv` through untouched.

## 3. Where the operator set is defined

Operator *names* now live in one place: `expression/op_names.hpp`
(`unary_from_name` / `binary_from_name` / the name lists used in error messages). The R,
Python and WebAssembly bridges each carried a private copy of that table before this change;
they now share it, so adding an operator is one enum entry plus one row in that header.
Rendering back to a string stays in `tree.hpp` (`detail::unary_name`).

Prediction re-parses the expression string in each host language, so each needs a shim for
the operators that are not host builtins — `neg`, `square` and now `inv`:
`predict_rsymbolic2.R`, `_eval_namespace()` in `python/rsymbolic2/__init__.py`, and
`UNARY_FNS` in `web/app/js/predict.js`.

## 4. Verification

- AD vs central finite differences for `inv` (`test_dual.cpp`), the standing rule for any
  new operator.
- Bit-identity of the scalar / vector-mode / SoA paths on a tree containing `inv`
  (`test_multi_dual.cpp`, `test_soa_eval.cpp`).
- Value + `to_string` round trip (`test_tree_eval.cpp`), LaTeX (`test_to_latex.cpp`),
  dimensions (`test_dimensional_analysis.cpp`).
- Recovery of `y = 1/x` and a `predict()` round trip in both R (`test-operators.R`,
  `test-predict.R`) and Python (`test_rsymbolic2.py`).
- **Parity guard:** the fixed-seed expressions printed by `test_evolutionary_search` are
  byte-identical before and after the change. Appending an enumerator that nobody selects
  must not move the RNG stream, and it does not.

## 5. What is NOT claimed

This document does not claim `inv` improves recovery. It is search-space *coverage*: a
structure that was three nodes away is now one node away. Whether that converts into a
measurable Feynman recovery gain is a separate screen (protocol per `docs/44` / `docs/47`:
medians over ≥5 seeds, the same operator set given to PySR, no threshold weakening). Until
that screen runs, the honest statement is: **effect unmeasured**, the option is available,
and the defaults are untouched.
