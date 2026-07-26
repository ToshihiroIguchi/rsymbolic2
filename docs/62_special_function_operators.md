# Special-Function Operators: `erf`, `sinh`, `cosh`

**Date:** 2026-07-26
**Status:** implemented, **off by default** (opt-in via `unary_ops`)
**Scope:** three unary operators added to the closed operator set; no default changes.
**Interfaces:** R, Python, WebAssembly (web GUI)

---

## 1. Why these three, and not the other special functions

The question this answers is "which special functions of physical chemistry are worth adding
as operators". The corpus is large — erf/erfc, Γ and ψ, Bessel `J_n`/`I_n`/`K_n`, Legendre
and Hermite polynomials, the Debye and Langevin functions, exponential integrals, Lambert W,
Voigt/Faddeeva, Airy, Mathieu, elliptic integrals, Fermi–Dirac integrals — and almost none of
it belongs here. Five criteria decided it, in this order:

1. **A SymbolicRegression.jl counterpart must exist.** This is the same gate `inv` had to pass
   (`docs/56` §1): the required PySR comparison hands the *identical* operator set to both
   tools, so an operator with no SR.jl name makes the benchmark impossible. Verified against
   the installed source, not from memory: `SymbolicRegression/src/Operators.jl` has
   `using SpecialFunctions: erf, erfc`, `src/Core.jl` and `src/SymbolicRegression.jl` export
   both, and its own comment lists the implicitly-available unary set as
   "exp, abs, log1p, sin, cos, tan, **sinh, cosh**, tanh, asin, acos, atan, asinh, acosh,
   atanh, **erf**, erfc, gamma, relu, round, floor, ceil, round, sign". All three names pass.
2. **The macro layer must not already reach it.** A macro operator (`docs/57`) is an
   expansion template over the primitives and costs no C++ at all, so anything expressible as
   one is not a candidate for a primitive. The decisive constraint is that a macro body uses
   its argument **exactly once**.
3. **Defined and differentiable on the whole real line, with an elementary derivative**, so
   forward-mode AD stays closed and no new guard policy is invented.
4. **In the C++ standard library**, so no dependency and no per-platform toolchain work.
5. **Not redundant with an operator already present.**

Criterion 2 is what selected this particular three, and it is the finding worth recording:
**the physical-chemistry motifs with the most value are precisely the ones a macro cannot
express**, because each needs its argument more than once.

| Motif | Why the existing layers cannot reach it |
|---|---|
| `sinh`, `cosh` | `(e^x - e^-x)/2` uses `x` twice — never a macro |
| `erf` | not an elementary composition at all |
| Langevin `coth x - 1/x` | `x` twice (`docs/57` ships `coth` as "the reachable part") |
| Hill `x^n/(1+x^n)` | `x` twice |
| Yukawa `e^-x / x`, `sinc` | `x` twice |
| Einstein heat capacity `x²e^x/(e^x-1)²` | `x` three times |

### What was rejected, and why

- **Bessel, Airy, elliptic, Mathieu, hypergeometric, Coulomb wave.** C++17's special-math
  functions are present in libstdc++ but not portably everywhere, which is a **portability**
  cost, and CLAUDE.md ranks portability above performance and treats a dependency that raises
  Windows build cost as a major architectural penalty. Worse for criterion 3: their
  derivatives are other members of the same family, so AD would not close over the operator
  set. They are also rarely the object of a regression over tabular `(X, y)`.
- **Voigt / Faddeeva.** Arguably the most useful function in spectroscopy, but it takes two
  shape parameters — outside the unary frame entirely. A new binary operator is not worth it
  when `gauss + lorentz` is already reachable as a sum.
- **Γ, lnΓ.** `std::lgamma` exists, but the derivative needs a hand-written digamma and the
  function has poles at the non-positive integers, so it would need a guard policy of its
  own. Its argument is rarely the regression variable.
- **`E₁` (exponential integral), Lambert W.** Both genuinely central to their niches
  (non-isothermal kinetics, integrated rate laws) and both absent from the standard library.
  Deferred, not rejected: revisit when a concrete problem asks.
- **`erfc`.** Present in SR.jl, but `1 - erf(x)` uses `x` once, so it is a **macro**, and it
  now ships as a web-GUI preset. Adding a second primitive for it would fail criterion 5.
  The caveat is stated in the preset's tooltip: for large `x` the subtraction cancels to a
  few digits, which does not matter for the shape the search is fitting but would matter for
  evaluating a diffusion tail.

## 2. Semantics

| Operator | Value | Derivative | Guard |
|---|---|---|---|
| `erf` | `std::erf(x)` | `(2/√π)·e^{−x²}` | none — bounded on all of ℝ, so none is possible or needed |
| `sinh` | `std::sinh(x)` | `cosh(x)` | none — like `exp` |
| `cosh` | `std::cosh(x)` | `sinh(x)` | none — like `exp` |

The guard question has a settled answer in this codebase and these three land on the existing
sides of it. `sqrt`/`pow` are guarded because the *real function itself* is undefined on part
of the domain; `div`/`inv`/`exp` are not, and rely on the loss finiteness guard to reject a
candidate that leaves the double range. `sinh`/`cosh` overflow for a large argument exactly as
`exp` does, so they follow `exp`. `erf` cannot leave `[-1, 1]`, so the question does not arise.
SymbolicRegression.jl uses all three unguarded too.

The `2/√π` factor is defined **once**, as `kTwoOverSqrtPi` in `dual.hpp`; `multi_dual.hpp` and
`soa_eval.hpp` include that header and use the same constant, so the three evaluation paths
cannot drift apart in the last bit — which is a hard requirement, not a nicety (below).

- Dimensional analysis: all three are transcendental — a dimensioned argument violates, and
  the result is concretely dimensionless, exactly like `sin`/`tanh`.
- LaTeX: `\sinh`, `\cosh`, and `\operatorname{erf}` (LaTeX has no `\erf`).
- Display simplification: `erf` and `sinh` join the **odd** class (`f(-t) → -f(t)`), `cosh`
  the **even** class (`f(-t) → f(t)`), alongside `sin`/`tanh` and `cos`. `test_dual.cpp`
  asserts the exact floating-point antisymmetry these folds rely on.
- **No e-graph rules were added** (`egraph.cpp`), deliberately. The rule set shares a fixed
  saturation budget (56 rules, 10 iterations, 10k e-nodes, 10 ms), so adding rules can change
  which form is chosen for expressions that contain none of the new operators. That is a
  display-only risk with no upside worth taking for three opt-in operators; they pass through
  the e-graph untouched. `inv` was left the same way, for a different reason (`docs/56` §2).

## 3. Where the changes are

The C++ side is the enum (`node.hpp`, **appended** — the RNG maps through the search space's
operator vector, not the enum, so appending is inert), one row each in `op_names.hpp`, and one
case each in the three evaluation paths, `unary_name`, `latex.hpp`, `dimensional_analysis.hpp`
and `display_simplify.cpp`. Every binding reads operator names through `unary_from_name`, so
none of the three bridges needed a parser change.

`predict()` re-parses the expression string in each host language, so each host needs whatever
the host lacks:

| Host | `sinh` / `cosh` | `erf` |
|---|---|---|
| R | base R | `2 * stats::pnorm(x * sqrt(2)) - 1` — exact to double precision (`pnorm` is Cody's algorithm). Adds `Imports: stats`, a base R package. |
| Python | `np.sinh` / `np.cosh` | `math.erf` (the C library's — the same function the core calls) vectorised with `np.frompyfunc` |
| Browser | `Math.sinh` / `Math.cosh` | **nothing exists** — see below |

### The browser's `erf`

JavaScript has no error function, so `web/app/js/predict.js` carries its own. It uses the two
standard series, each inside its well-conditioned range:

- `|x| < 3`: the confluent form of the Maclaurin series,
  `erf(x) = (2x/√π)e^{-x²} Σ (2x²)ⁿ / (1·3···(2n+1))`. Every term is positive, so it has none
  of the catastrophic cancellation the alternating form suffers.
- `|x| ≥ 3`: `1 - erfc|x|` with `erfc` from its continued fraction, evaluated backwards.
  `erfc(3)` is 2.2e-5, so even a loose relative error there sits far below one ulp of the
  `erf` value it is subtracted from.

Measured against C's `erf` on 1613 points spanning ±8 and both branches: **worst relative
error 7.6e-16** (about 3 ulp). That is the same order as the WASM-vs-native libm divergence
the project already documents (`docs/51`), so it introduces no new class of disagreement.

## 4. Verification

- AD vs central finite differences at five points per operator, plus exact odd/even symmetry
  and the unguarded-overflow behaviour (`test_dual.cpp`).
- Bit-identity of the scalar / vector-mode / SoA paths on a tree containing all three
  (`test_multi_dual.cpp`, `test_soa_eval.cpp`).
- Value + `to_string` round trip (`test_tree_eval.cpp`), LaTeX (`test_to_latex.cpp`),
  dimensions (`test_dimensional_analysis.cpp`), odd/even folds and the randomised
  semantics-preservation corpus, which now draws from the new operators too
  (`test_display_simplify.cpp`).
- Recovery of `y = erf(x)` and `y = sinh(x)` with a `predict()` round trip in both R
  (`test-operators.R`) and Python (`test_rsymbolic2.py`). Python's
  `test_unary_op_names_have_one_definition` already runs the engine once per accepted name,
  so the new names are covered there automatically.
- Browser: `parity_test.cjs` §2g checks `predict.js`'s `erf` against C's at eight points
  across both branches (1e-14), then recomputes the engine's loss for a fitted
  erf/sinh/cosh expression through `predict.js` alone. That second check is deliberately
  loose (1e-3 relative): `to_string` prints constants with `%.6g`, so the printed expression
  fits slightly rounded constants (`docs/48` D2) — it is a composition check, not a
  precision one.
- **Parity guard:** the fixed-seed expressions in `test_evolutionary_search` are byte-identical
  before and after. Appending enumerators that nobody selects must not move the RNG stream.

## 5. What is NOT claimed

No accuracy claim. This is search-space **coverage**: three motifs that no combination of the
existing operators or macros could express are now one node each. Whether that converts into
measurable Feynman recovery is a separate screen (protocol per `docs/44` / `docs/47`: medians
over ≥5 seeds, the same operator set given to PySR, no threshold weakening). Until it runs the
honest statement is: **effect unmeasured**, the option is available, the defaults are
untouched, and PySR default parity is therefore unchanged.
