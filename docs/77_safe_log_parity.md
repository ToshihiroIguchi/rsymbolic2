# 77 — safe_log: the guard docs/69 left off

**Date:** 2026-08-02
**Status:** implemented and verified on Windows; Ubuntu verification pending at time of
writing (see §6). Closes the last unaligned protected operator, and **corrects** the
"equivalent in effect" reasoning recorded in `docs/69` §7.
**Change:** `dual.hpp` (the single definition), `multi_dual.hpp`, `soa_eval.hpp` (both the
value and the gradient kernel), `tree.hpp`; tests in `test_tree_eval.cpp` and
`test_soa_eval.cpp`.
**Search behaviour changes only where a `log` argument is exactly ±0.0** — see §3, which
is the honest scope of this fix.

## 1. How this surfaced

Reading the README for defects, not the code. Two sections contradicted each other:

| README section | claim |
| --- | --- |
| "Operators" | "`sqrt(x)` returns `0` for `x < 0`" |
| "Getting the formula out" | "`sqrt`, `log` and `^` return `NaN` outside their domain" |

The first was simply stale — pre-`docs/69` text that the `docs/69` change never came back
to update. But checking the second against the code to decide which one to keep showed
that *it* was not right either: `log` was **not** guarded. `dual.hpp` had

```cpp
inline Dual log(const Dual& a) {
    return {libm::log(a.value), a.deriv / a.value};
}
```

and `tree.hpp`'s `apply_unary` pulled in `using libm::log`, so the plain-double path went
straight to the C library. `docs/69` had considered this and deliberately left it alone:

> Our `log` is unguarded, which is equivalent in effect: `log(x<=0)` gives NaN or −Inf and
> both are non-finite, so the candidate is rejected either way.

## 2. Why "equivalent in effect" is wrong

The reasoning skips a step. **Rejection does not happen at the operator; it happens on the
final loss.** A non-finite value has to still be non-finite when it reaches the residual,
and −Inf is *not* a fixed point of the operator set. The very next node can map it back
into the finite range:

```
exp(log(0))  = exp(-Inf)  =  0        tanh(log(0)) = tanh(-Inf) = -1
inv(log(0))  = 1/(-Inf)   = -0        cos(log(0))  = cos(-Inf)  = NaN   (this one rejects)
```

Measured on the shipped evaluator before the fix:

```
log(0.0)            = -inf
exp (log(0.0))      = 0     finite=YES (candidate SURVIVES)
tanh(log(0.0))      = -1    finite=YES (candidate SURVIVES)
inv (log(0.0))      = -0    finite=YES (candidate SURVIVES)
cos (log(0.0))      = nan   finite=no  (rejected)
```

SR.jl's `safe_log` returns NaN at the log, and NaN survives every one of those four — so
each of the first three is a candidate rsymbolic2 scored and could keep while PySR
discards it. NaN is the *mechanism* of rejection (`docs/69` §2); −Inf only sometimes is.

`log` and `exp` are both in the **default** operator set, so `exp(log(x))` is a
default-path tree, not something reachable only through an opt-in operator.

## 3. The honest scope: the delta is exactly ±0.0

It would overstate this to say the whole `x <= 0` domain changed. IEEE `log` **already**
returns NaN for `x < 0`, which is what SR.jl returns too — those agreed before this
change. The only argument where the old and new behaviour differ is **exactly zero**:

| argument | old | new | SR.jl `safe_log` |
| --- | --- | --- | --- |
| `x > 0` | `log(x)` | `log(x)` | `log(x)` |
| `x == ±0.0` | **−Inf** | **NaN** | **NaN** |
| `x < 0` | NaN | NaN | NaN |
| `x == NaN` | NaN | NaN | NaN |

So the divergence needed a `log` argument that is *exactly* `0.0` at some data row —
a zero-valued feature, a constant the optimiser landed exactly on zero, or a subtraction
that cancelled exactly. That is narrow, but it is not unreachable, and under the project's
parity rule a divergence is a bug to fix rather than a frequency to weigh.

This is also why the change is invisible on the standard problem set (§5).

## 4. What was changed

`safe_log(x) = x > 0 ? log(x) : NaN`, transcribed as the literal Julia branch
(`x <= 0 -> NaN`, else `log`) rather than a folded `!(x > 0)`. `docs/69` §4.1 records that
folding these guards is exactly how the `pow` divergence survived; for a NaN argument the
Julia form falls through to `log(NaN) = NaN`, so both spellings agree here, and the literal
one is kept for the same reason as `safe_pow`.

Four evaluation paths must agree or the evaluators diverge (the `sqrt` failure mode
`docs/69` §1 found via `bench_soa_eval`), so all four were changed from one definition:

| path | change |
| --- | --- |
| `dual.hpp` | new `log(double)` overload + guard on the `Dual` overload (`{NaN, NaN}`) |
| `multi_dual.hpp` | guard on `MultiDual<N>`: value and **every** lane NaN |
| `soa_eval.hpp` value kernel | `libm::log` → `rsymbolic::log` |
| `soa_eval.hpp` gradient kernel | guarded; see below |
| `tree.hpp` | dropped `using libm::log`, so `Log` resolves to `rsymbolic::log` for double too |

The gradient kernel is the subtle one. It divides by the **argument**, so:

- the division must be taken before `val` is overwritten with the log; and
- the NaN must be **forced from the value**, not inherited from the division. At `val == 0`
  the division alone yields ±Inf, and at `val < 0` a perfectly finite number — neither is
  NaN. Inheriting would have left the gradient finite under a NaN value, which is precisely
  the trap `soa_eval.hpp`'s existing `Sqrt` comment warns about.

The UCRT redirect (`docs/68`) is unaffected: `rsymbolic::log` calls `libm::log` internally,
so only the guard is added, not a different libm.

## 5. Verification

- **`diag_search_digest`: byte-identical, 661/661 lines, before vs. after.** The search
  trajectory on the standard problem set does not move — consistent with §3, since no
  problem there feeds a `log` an exact zero. This is the evidence that the fix closes a
  parity hole without disturbing tuned behaviour, and it is why no benchmark re-run was
  needed to accept it.
- **Standalone suite: 30/30 pass**, including two new tests:
  - `test_tree_eval.cpp::test_safe_operator_semantics` — the `safe_log` truth table of §3,
    including `-0.0` (`-0.0 <= 0.0` is true, so it is out of domain) and `log(+Inf) = +Inf`.
  - `test_tree_eval.cpp::test_log_out_of_domain_is_not_rescued` — **the actual defect**:
    `exp`/`tanh`/`inv`/`neg` applied to `log(0)` and `log(-1)` must all be NaN, plus
    `exp(log(4)) == 4` to show the in-domain composition is untouched.
  - `test_soa_eval.cpp::test_log_nonpositive` — the twin of `test_sqrt_negative`, covering
    the SoA gradient path, and asserting `log(0)` is NaN rather than −Inf.
- `test_all_unary` already contained a `log(c1+x1)` node; its data keeps the argument
  positive, so it is unaffected — checked rather than assumed.

## 6. Residual and not done

- **Ubuntu verification** had not been run when this document was written. The change is
  pure C++ with no platform-specific code and the digest is unchanged on Windows, so no
  divergence is expected, but the project's "done on both platforms" bar is not met until
  it is run.
- **`log`'s derivative at the boundary** is NaN along with the value, matching the `sqrt`
  convention. SR.jl/ForwardDiff would carry its own convention here; as with `safe_pow`
  (`docs/69` §7) this only steers the constant optimiser on a candidate that is already
  rejected on loss.
- **`docs/29`** catalogued the `pow` divergence as #12; the `log` one was never
  catalogued at all, exactly like `sqrt`. Both are now closed.
- **The manuals** carried the pre-`docs/69` "returns 0" claim in three places (README
  "Operators", the R `binary_ops` roxygen, and hence the generated `.Rd`); all were
  corrected alongside this change, and the `predict()` notes in both bindings now list
  `log(0)` among the IEEE-vs-engine edge differences.
