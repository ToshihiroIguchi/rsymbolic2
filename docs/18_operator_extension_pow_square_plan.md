# Operator Extension Plan: `square` and `pow` (+ AD verification)

**Date:** 2026-06-07
**Scope:** Add the `square` (unary) and `pow` (binary) operators required before the
Feynman benchmark, and close the still-open Phase 1 item "AD vs finite-difference
verification". This is the prerequisite step for `docs/19` (Feynman benchmark plan);
it does NOT itself run the Feynman benchmark.

---

## 1. Why these operators, and why now

The operator set is **the search space**, and per CLAUDE.md it is "the one shared
problem input given identically to both tools". It must therefore be settled *before*
benchmarking — otherwise we would benchmark, change the search space, and have to
re-benchmark (a wasted cycle, and a methodological smell).

The Feynman dataset is power-heavy: inverse-square laws (`1/r^2`), kinetic-energy and
area terms (`x^2`), and assorted fractional/integer powers. With the current operator
set (unary `{neg,exp,log,sin,cos,sqrt,tanh,abs}`, binary `{add,sub,mul,div}`) a large
fraction of Feynman equations is **structurally unreachable**. Benchmarking without
powers would report an artificially low recovery rate that reflects operator coverage,
not search quality — a misleading result. Adding `square` + `pow` is therefore
*justified* scope, not speculative scope expansion (CLAUDE.md: "Build what the current
phase needs").

**Deliberately NOT added now:** `cube`, `inv`, `sigmoid`, `arcsin`, etc. They are not
demonstrably required by the target subset and would expand the search space (and the
test/maintenance surface) without evidence. If `docs/19` analysis shows a specific
unreachable cluster that only one of these unlocks, add it then, with that evidence.

---

## 2. Operator semantics (the decisions)

### 2.1 `square` (unary) — cheap, safe, no domain issue

- Forward value: `square(x) = x * x`.
- Derivative (dual): `d = 2 * x.value * x.deriv`.
- No NaN/Inf risk for finite inputs (overflow to `+Inf` is handled by the existing
  loss-guard path, same as `exp`).
- Rationale for adding it *in addition to* `pow`: it is the single most common Feynman
  power, it is a far smaller search step than `pow(x, 2)` (one node, no constant to
  fit, no `log`), and it keeps `pow` from being the only route to `x^2`. This biases
  the search toward the common case at near-zero cost.

### 2.2 `pow` (binary) — must be a *safe* operator

Naive `pow(x, y) = exp(y * log(x))` is undefined for `x <= 0` with non-integer `y`,
producing NaN that poisons the LM solver. This codebase already uses guarded ops
(safe `sqrt` → 0 on negative input, `div` via reciprocal). `pow` must follow the same
convention.

**Chosen safe semantics — `safe_pow`, mirroring SymbolicRegression.jl `^`:**

```
safe_pow(x, y):
    if x > 0:        return exp(y * log(x))      # standard branch
    if x == 0:       return (y > 0) ? 0 : <guard>  # 0^positive = 0
    if x < 0:
        if y is (numerically) an integer: return sign-correct x^y
        else:        return <guard>             # x<0, fractional y → undefined
```

`<guard>` returns a large finite penalty value (the existing `kInf`/loss-guard
sentinel used by `safe_log`/`safe_div`), never NaN, so the optimizer treats the region
as bad rather than crashing.

**Critical fairness constraint (CLAUDE.md):** because the operator set is given
*identically* to both tools, the PySR-side comparison (`docs/15`, SymbolicRegression.jl)
must use the **same** `safe_pow` semantics. SymbolicRegression.jl's default `^` already
*is* `safe_pow` with the analogous integer/negative handling — confirm the exact branch
behaviour matches before the Feynman comparison run, and record any divergence in
`docs/19`. Do **not** alter PySR settings to match us; if semantics differ, document it
as a known caveat of the shared input.

### 2.3 AD for `pow` (both partials)

`pow` is the first operator where **both** operands carry derivatives and the second
partial needs `log(x)`. Dual rule (value `p = x^y`):

- `∂p/∂x = y * x^(y-1)`  → contributes `y.value * pow(x.value, y.value-1) * x.deriv`
- `∂p/∂y = x^y * log(x)` → contributes `p * log(x.value) * y.deriv` (guard `x>0`;
  for `x<=0` the `log` term is set to 0, consistent with the safe-value guard)

Combined: `p.deriv = y.value*pow(x,y-1)*x.deriv + (x>0 ? p*log(x)*y.deriv : 0)`.

This is exactly why §4 (AD verification) is folded into this work: a wrong `∂p/∂y` is
silent — the search still runs, it just optimizes constants badly — so it must be
caught by a numerical check, not assumed.

---

## 3. Code touch points

Adding an operator is a known, localized change (sqrt/tanh/abs followed this path).
Files and edits:

| File | Edit |
|------|------|
| `expression/node.hpp` | `enum class UnaryOp` += `Square`; `enum class BinaryOp` += `Pow` |
| `expression/dual.hpp` | `Dual square(const Dual&)`; `Dual pow(const Dual&, const Dual&)` (safe, both partials) |
| `expression/tree.hpp` | `apply_unary` += `Square` case; `apply_binary` += `Pow` case (double path uses `safe_pow`, not raw `std::pow`); `op_name`/`op_char` serialization += `"square"` / `"^"` |
| `src/rsymbolic2_r.cpp` | string→op mapping: `"square"`→`UnaryOp::Square`, `"pow"`→`BinaryOp::Pow` |
| `src/simplify.cpp` | constant folding already routes through `apply_unary`/`apply_binary` (covered); add identity rules **only if** measured to matter: `pow(x,0)→1`, `pow(x,1)→x`, `pow(x,2)→square(x)` (canonicalization). Keep minimal. |
| `evolution/search_space.hpp` (+ random_tree/mutation if they enumerate ops) | ensure new ops are selectable when requested, and *not* forced on by default |
| `R/symbolic_regression.R` (roxygen) | document `"square"` in `unary_ops`, `"pow"` in `binary_ops`; defaults unchanged |
| `man/symbolic_regression.Rd` | regenerate from roxygen |

**Double vs Dual evaluation must agree on the safe branch.** The `double`
specialization currently relies on ADL/`std::` (e.g. `std::sqrt` with a NaN→kInf
guard upstream). For `pow` we must route the `double` path through the *same*
`safe_pow` logic, not raw `std::pow`, or the value path and the dual path will
disagree on `x<=0`. Add a shared `safe_pow(double,double)` helper used by both.

---

## 4. AD verification (closes the open Phase 1 item)

CLAUDE.md requires "numerical checks for automatic differentiation (compare against
finite differences)". This is listed as still-missing in `docs/05`. Implement it now,
covering the **whole** operator set, not just the two new ones.

- New test `standalone/tests/test_dual_vs_finitediff.cpp` (Catch2):
  - For each unary and binary operator, build small random trees (depth ≤ 5) over
    random constants in the operator's valid domain.
  - Compute the analytic Jacobian via `Dual` (the existing
    `least_squares_problem.hpp` path) and a central finite-difference Jacobian
    (`(f(c+h) - f(c-h)) / 2h`, `h = 1e-6`).
  - Assert max relative error `< 1e-6` over ≥ 100 random trees (the roadmap's
    "100+ random expressions" bar).
  - Restrict sampling to each op's defined region (e.g. `pow` base `> 0`) so the
    test checks the analytic branch, not the guard branch; add a few explicit
    guard-branch cases asserting *finite, non-NaN* output.
- This is the gate that protects against a silent `∂pow/∂y` error.

---

## 5. Test plan

1. **Unit (C++ / Catch2):**
   - `test_dual.cpp`: add `square`, `pow` value+deriv known-answer cases (incl. `x<=0`
     guard returns finite).
   - `test_tree_eval.cpp`: postfix eval of trees using `square`/`pow`; round-trip
     serialize→parse→serialize stability for the new tokens.
   - `test_dual_vs_finitediff.cpp`: §4.
2. **R (testthat):** `symbolic_regression(..., unary_ops=c(...,"square"), binary_ops=c(...,"pow"))`
   runs, returns valid output, reproducible under fixed seed; a tiny known target
   (e.g. `y = x^2 + 1`) recovers.
3. **Simplifier:** if identity rules added, unit-test each (`pow(x,1)→x`, etc.).

---

## 6. No-regression and platform gates ("done" criteria)

A change is not done until ALL hold (CLAUDE.md Platform Constraints):

- [ ] C++ unit tests pass (including §4 AD verification).
- [ ] **Nguyen gate re-run, no recovery regression** at shipped defaults
      (`01_nguyen_gate.R` / `01b_nguyen_gate_sqrt.R`). New ops are *available* but the
      Nguyen runs do not enable them → the suite must reproduce the prior 10/10. This
      confirms adding ops to the enum/registry did not perturb existing search.
- [ ] Builds + tests pass on **Windows 11 (Rtools45 MinGW/UCRT)** AND **Ubuntu LTS**.
- [ ] `R CMD check --as-cran` stays clean: no new ERROR/WARNING, no new NOTE beyond
      the benign ones recorded in `docs/17`. New compiled code uses `REprintf`, never
      C `stderr` (the CRAN fix from commit 5359f42 must not be reintroduced).
- [ ] Rd/roxygen regenerated; `unary_ops`/`binary_ops` docs list the new values; the
      **defaults are unchanged** (opt-in only).

---

## 7. Sequencing

1. Implement `square` (smallest, safe) → unit test → green.
2. Add `safe_pow` helper (shared double/Dual) → `pow` value+AD → unit test → green.
3. Add `test_dual_vs_finitediff.cpp` over the full operator set → green.
4. R-side wiring + roxygen + Rd; testthat.
5. Both-OS build + `R CMD check --as-cran`.
6. Nguyen no-regression re-run; archive CSV.
7. Commit. **Then** write `docs/19` (Feynman benchmark plan): equation subset
   selection (honestly excluding still-unreachable forms), self-generated data within
   documented ranges (no Julia/large download), held-out test split + NMSE<1e-4 +
   structural check, multi-variable handling, staged dev-subset→full-set scale, and
   the PySR-at-defaults comparison with the matching `safe_pow` semantics from §2.2.

---

## 8. Open risk

- **`safe_pow` semantics drift vs SymbolicRegression.jl.** If the integer-detection or
  negative-base branch differs, the "identical shared input" claim weakens. Mitigation:
  verify branch-by-branch against SymbolicRegression.jl `safe_pow` before the Feynman
  comparison; document any residual difference in `docs/19` rather than papering over it.
- **Search-space inflation.** `pow` with a free exponent enlarges the space and can
  slow convergence / increase bloat. Mitigation: `square` covers the common case
  cheaply; watch the Nguyen no-regression timing and the Feynman dev-subset runtime
  before committing to the full run.
