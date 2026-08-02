# 69. safe_sqrt and safe_pow: out of domain must be NaN, not 0

**Date:** 2026-07-31
**Status:** implemented; verified on both platforms. Closes difference **#12** in
`docs/29` and an equivalent divergence for `sqrt` that was never catalogued at all.
Supersedes the guard convention chosen in `docs/18` §2.2.
**Change:** `dual.hpp` (the single definition), `multi_dual.hpp`, `soa_eval.hpp`,
`tree.hpp`; tests in `test_tree_eval.cpp`, `test_dual.cpp`, `test_soa_eval.cpp`,
`test_display_simplify.cpp`. **Search behaviour changes on every platform.**

## 1. How this surfaced

`docs/68` §10 recorded an incidental finding: pointing `bench_soa_eval` at the shipped
evaluator made it report `bit-exact NO` for a tree containing `sqrt`. The two evaluators
disagreed for a negative argument — `tree.hpp::apply_unary<double>` returned NaN (plain
`std::sqrt`), the SoA evaluator returned 0.

The obvious repair is to give `apply_unary` the same 0-guard. That is the wrong direction,
and checking the authoritative source is what showed it.

## 2. What SymbolicRegression.jl actually does

`SymbolicRegression.jl/src/Operators.jl`:

```julia
function safe_sqrt(x::T)::T where {T<:FloatOrDual}
    return x >= zero(x) ? sqrt(x) : T(NaN)
end
```

Every protected operator there follows the same convention — `safe_log`, `safe_log2`,
`safe_log10`, `safe_log1p`, `safe_asin`, `safe_acos`, `safe_acosh`, `safe_atanh` and
`safe_pow` all return **NaN** out of domain. The NaN is not an accident to be defended
against; it is the mechanism. A NaN prediction makes the loss non-finite, and a non-finite
loss **rejects the candidate** (`docs/29` #6 already records our matching "non-finite child
loss is skipped" behaviour).

`safe_pow` is a branch table that reduces to one exception:

```julia
if isinteger(y)
    y < zero(y) && iszero(x) && return T(NaN)
else
    y > zero(y) && x < zero(x) && return T(NaN)
    y < zero(y) && x <= zero(x) && return T(NaN)
end
return x^y
```

Written out against IEEE `pow`, every branch except one is already what `pow` does: it
returns NaN for a negative base with a non-integer exponent and the correctly-signed real
result for an integer one. The single departure is `0^negative`, which IEEE `pow` gives as
±Inf and SR.jl gives as NaN.

## 3. What rsymbolic2 was doing, and why it mattered

| case | rsymbolic2 (before) | SR.jl | effect |
|---|---|---|---|
| `sqrt(x)`, x < 0 | **0** | NaN | candidate survives that PySR discards |
| `pow(x,y)`, x<0, y non-integer | **0** | NaN | ditto |
| `pow(x,y)`, x<0, y within **1e-6** of an integer | `pow(x, round(y))` | NaN | `pow(-2, 2.0000001)` answered **4** |
| `pow(0,y)`, y < 0 | **0** | NaN | ditto |
| `pow(0,0)` | **0** | **1** | plain wrong |
| `pow(x,y)`, x > 0 | `exp(y·log(x))` | `x^y` | a few ulp, and `x^1 != x` |

The 1e-6 tolerance deserves singling out. Constants are continuous and the optimiser moves
them freely, so "near-integer exponent" is not a rare corner: the old code answered a
finite number over a whole neighbourhood where PySR rejects.

Under CLAUDE.md's PySR Default Parity rule this is not a trade-off to weigh. It is a bug.

### 3.1 Why the original reasoning was wrong

`docs/18` §2.2 chose 0 to "prevent NaN from poisoning the LM solver". That risk does not
exist in this codebase: `self_lm_optimizer.cpp` already clamps every non-finite residual
and Jacobian entry to `kLargeResidual = 1e10`, so a NaN makes the solver treat the point as
a very poor fit and step away. And `sse_current` returns `kInf` on the first non-finite
residual, which is exactly the rejection SR.jl performs. The defence was already in place
before the guard was written.

There is also a user-facing argument independent of parity: with the 0-guard, `predict()`
on data outside the model's domain returns a silently wrong finite number. NaN says
"outside the domain", which is the honest answer.

## 4. What changed

One definition, in `dual.hpp`:

```cpp
inline double sqrt(double x) {
    if (!(x >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(x);
}
inline double pow(double x, double y) {
    if (x == 0.0 && y < 0.0) return std::numeric_limits<double>::quiet_NaN();
    return libm::pow(x, y);
}
```

`!(x >= 0.0)` rather than `x < 0.0` so a NaN argument yields NaN, matching Julia's
`x >= zero(x) ? … : NaN`. Everything else routes through these: the `Dual` and `MultiDual`
overloads take their value from them, the SoA residual and Jacobian kernels call them
per point, and `tree.hpp`'s `apply_unary` no longer pulls in `std::sqrt` — `Sqrt` now
resolves to `rsymbolic::sqrt` for `double` exactly as it already did for `square`/`recip`/
`pow`. **The two paths can no longer disagree because there is only one definition.**

Derivatives are deliberately unchanged in structure: carried on the standard branch,
0 elsewhere. Where the value is NaN the candidate is rejected on loss regardless, and
`sqrt`'s NaN case propagates NaN into the gradient lanes so the SoA Jacobian stays
bit-identical to the scalar `Dual` reference.

The `pow` rewrite also drops one `exp` + one `log` per Pow node in favour of a single
`pow` call.

### 4.1 The branch table is transcribed, not simplified — and that mattered

The first version of this change *did* simplify it. Written out against IEEE `pow`, SR.jl's
table looks like it collapses to one exception: `pow` already returns NaN for a negative
base with a non-integer exponent and the correctly-signed real result for an integer one,
so surely only `0^negative` differs.

That reading is wrong at the infinities. SR.jl guards on `x < 0` and `x <= 0`, which
**include −Inf**, while IEEE `pow` is defined there and returns a number:

| case | IEEE `pow` | SR.jl `safe_pow` |
|---|---|---|
| `pow(-Inf, 0.5)` | `+Inf` | **NaN** |
| `pow(-Inf, 1.5)` | `+Inf` | **NaN** |
| `pow(-Inf, -0.5)` | `+0` | **NaN** |
| `pow(-Inf, -1.5)` | `+0` | **NaN** |
| `pow(-Inf, 2.0000001)` | `+Inf` | **NaN** |

A candidate whose base overflows to −Inf therefore survived under the "simplified" version
and is rejected by PySR — the same class of bug the change set out to remove.

**This was found by diffing against Julia, not by reading the code and not by the
benchmark.** `safe_sqrt` and `safe_pow` were dumped from the installed
SymbolicRegression.jl over a grid of edge cases (±0, ±Inf, NaN, denormals, near-integer
exponents) and compared entry by entry against the C++:

```
julia -e 'using SymbolicRegression: safe_sqrt, safe_pow; ...'   # 224 cases
```

The simplified version failed **9 of 224**. The literal transcription fails 3, all of them
the same accepted implementation difference: Julia's `^` special-cases integer-valued
`Float64` exponents and evaluates them by repeated squaring rather than through `pow`, so
`(-3.0)^(-3.0)` and `(1e10)^(-3.0)` differ from C's `pow` by **1 ulp**, and
`(-Inf)^(-3.0)` differs in the sign of zero. These are not chased: they are the same class
as the libm ULP differences the project already accepts across platforms (`docs/51`,
`docs/58`), matching them would require reimplementing Julia's `pow_body`, and a signed
zero cannot change a search decision (both zeros give the same SSE, and dividing by either
gives a non-finite value that is rejected either way).

`safe_sqrt` matches on **all** 14 cases with no exceptions.

The permanent protection is `test_tree_eval.cpp`, which now asserts the ±Inf rows above
directly, rather than a Julia dependency — CLAUDE.md forbids one at runtime, and this is a
development-time check.

## 5. Three tests had encoded the old behaviour

This is the part worth remembering. The divergence survived because the test suite
asserted it:

- `test_dual.cpp` asserted `pow(-2, 1.5)` is **finite** ("-> 0 (not NaN)").
- `test_tree_eval.cpp::test_safe_boundaries` asserted the same through the tree evaluator.
- `test_soa_eval.cpp` exercised `sqrt` only with **positive** arguments, so the
  scalar-vs-SoA disagreement was never reachable.

All three are updated, and the new coverage asserts the SR.jl table directly rather than
just cross-path agreement — agreement alone would still pass if both paths drifted back to
0 together. `test_soa_eval.cpp::test_sqrt_negative` drives the argument negative and checks
both bit-identity and the NaN itself, including the `x == 0` boundary that is *in* domain.

A fourth test changed meaning rather than being wrong:
`test_display_simplify.cpp` asserted that `pow(x,1) != x` (true of `exp(1·log(x))`) and
cited that as the justification for `display_simplify` having no Pow rewrite (`docs/54`).
IEEE `pow` is exact at `y = 1`, so the test now asserts exactness and records that the
justification has lapsed. **The `t^1 → t` rewrite is not added here** — that is a separate
decision with its own measurement; the test simply no longer claims it is impossible.

## 6. Effect on the search

Rejecting more candidates is a real change to the search, not a neutral one, so it was
measured rather than assumed. Feynman dev gate (`benchmarks/02_feynman_gate.R stage=1`),
**3 seeds per problem**, everything else at the frozen parity values, run on the same
machine with the baseline installed from the parent commit:

| | recovered | verdict |
|---|---:|---|
| parent commit (0-guard) | **18/25** | PASS |
| this change | **19/25** | PASS |

One problem moved, and it moved the right way:

| problem | parent | this change |
|---|---|---|
| `boltzmann_dist` — `n0·exp(−m·g·x/(kB·T))` | 0/3, med NMSE **1.4e-02** | **2/3, med NMSE 2.4e-31** |

That is not a threshold flip. `1.4e-02` is nowhere near the `1e-4` recovery threshold and
`2.4e-31` is a clean structural recovery, in the same band as the problems that were
already solved. Every other problem keeps its verdict; the per-seed counts and median
NMSEs move around as expected from a changed trajectory.

**Three seeds is below CLAUDE.md's benchmarking bar of ≥ 5 runs**, and the full 5-seed gate
is ~8 h per arm. Read this as "no evidence of harm, one clean gain", not as a measured
+1. The harness's own guidance is that a reduced-seed pass exists to de-risk the full gate.

### 6.1 An earlier version of this change measured 16/25, and the benchmark was right

The first implementation — the "simplified" `safe_pow` of §4.1 — scored **16/25** against
the same 18/25 baseline, losing `lorentz_x` and `clausius_moss`. Both losses were
threshold-marginal (`7.3e-05 → 1.6e-04` and `4.0e-06 → 1.3e-04`, against a `1e-4` cut), so
it was tempting to write them off as noise around a cut-off and keep going.

The −2 is what prompted checking the implementation against Julia instead of against
intuition, which is how the −Inf bug surfaced. With the transcription corrected the same
gate reads 19/25. **The benchmark was not noise; it was pointing at a real defect**, and
the lesson is that a parity change scoring worse on the primary benchmark is a reason to
re-verify the implementation before invoking the parity rule to justify the regression.

## 7. Verification

| | Windows | Ubuntu 24.04 (WSL) |
|---|---|---|
| standalone `ctest` | 29/29 | 29/29 |
| R `testthat` (`NOT_CRAN=true`) | 327 pass, 0 fail | 327 pass, 0 fail |
| `pytest` | 66 passed | 57 passed, 9 skipped |
| WASM parity gate | PASSED | — |

Results change on **every** platform this time — unlike `docs/68`, which was Windows-only.
The `diag_search_digest` before/after diff is therefore expected to differ everywhere, and
is not a regression signal.

## 8. Left undone, deliberately

- **`t^1 → t` in `display_simplify`** (§5). Now sound; not added.
- **Derivative conventions off the standard branch.** SR.jl differentiates through
  `safe_pow` with ForwardDiff and would carry a derivative where we carry 0 (for a
  negative base with an integer exponent, for instance). This is an implementation
  difference in a place where the value already agrees, and it only steers the constant
  optimiser. Not changed, and not measured.
- **`sqrt`'s derivative at exactly 0** stays 0 rather than +Inf, unchanged from before.
- **The other SR.jl protected operators** (`log2`, `log10`, `log1p`, `asin`, `acos`,
  `acosh`, `atanh`) are not implemented in rsymbolic2 at all, so there is nothing to align.

> **Correction (docs/77).** This section originally continued: "Our `log` is unguarded,
> which is equivalent in effect: `log(x<=0)` gives NaN or −Inf and both are non-finite, so
> the candidate is rejected either way." **That was wrong**, and `log` is now guarded.
> Rejection happens on the *final loss*, not at the operator, and −Inf is not a fixed point
> of the operator set: `exp(-Inf)` is `0`, `tanh(-Inf)` is `-1`, `1/-Inf` is `-0`. So
> `exp(log(0))` scored as a finite `0` and the candidate survived, where SR.jl's `safe_log`
> gives NaN through all three. See `docs/77_safe_log_parity.md`.
