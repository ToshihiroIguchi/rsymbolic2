# 46. Dimensional analysis (PySR X_units / y_units): opt-in feature

**Date:** 2026-07-04
**Branch:** `feature/high-accuracy-options`
**Status:** Implemented and shipped **off by default**. PySR-compatible API and semantics;
own dependency-free C++ implementation (no Julia, no new third-party C++ dependency).
Correctness verified by known-answer tests mirroring SymbolicRegression.jl. The Feynman
**accuracy screen** (does it raise recovery?) is the declared next measurement — methodology
and reproduction are recorded in §6; it is **not yet run**, and no accuracy claim is made
until it is (medians over runs, per CLAUDE.md Benchmarking Requirements).

## 1. Why this exists

Dimensional analysis is a documented PySR feature that rsymbolic2 lacked — catalogued as
the absent feature **#7** in `docs/29_pysr_difference_catalog.md` §C (`X_units`, `y_units`,
`dimensional_constraint_penalty`). Because it is default-OFF in PySR, its absence was not a
parity violation, but it is useful to users whose variables carry physical units, and it was
named as the next opt-in high-accuracy candidate in docs/43. By user direction it is
implemented as a **PySR feature-parity item**: the SR.jl accuracy screen documents its
effect but does not gate the implementation.

It satisfies the CLAUDE.md "Opt-in high-accuracy options (the second layer)" contract:

1. **Off by default.** With no units declared the search is byte-identical to the units-off
   PySR-parity default. Verified: the entire pre-existing standalone/R/Python suites pass
   unchanged (the penalty helper's disabled path is a no-op).
2. **Documented with evidence.** Semantics/correctness are pinned by known-answer tests
   (§5); the accuracy effect is the screen in §6.
3. **Parity preserved when unused.** The units-off default comparison against PySR is
   unchanged.

## 2. API (matches PySR)

| Parameter | Type / default | Where |
|---|---|---|
| `X_units` | list/character of unit strings, one per feature / `None` | fit-time |
| `y_units` | single unit string / `None` (requires `X_units`) | fit-time |
| `dimensional_constraint_penalty` | float / `None` → effective **1000.0** | search config |
| `dimensionless_constants_only` | bool / `False` | search config |

Unit strings are DynamicQuantities-style: SI base (`kg, g, m, s, A, K, mol, cd`), common
derived units (`N, J, W, Pa, C, V, Ohm, T, Hz, F, H, Wb, S`; `rad`/`sr` dimensionless),
decimal SI prefixes (inert — `"km"` ≡ `"m"`), the dimensionless literal `"1"`, and
`* / ^ ( )` with integer or parenthesised rational exponents (`"m^(1/2)"`, Julia `//`
accepted). `µ`/`Ω` are accepted as `u`/`Ohm`. Products are explicit (`"N*m"`). Unknown
symbols, offset units (`°C`), and non-representable exponents raise an error naming the
token. Parsing lives in the shared C++ core (`units/unit_parser.hpp`), called once from each
bridge, so R and Python parse identically.

## 3. Semantics (mirrors SymbolicRegression.jl DimensionalAnalysis.jl)

A bottom-up pass over the postfix tree carries a `WildcardDim = {dimension, wildcard,
violates}` per node (a "wildcard" is a free constant whose dimension is still unfixed):

- **Leaf** constant → wildcard iff `!dimensionless_constants_only`; feature → its declared
  unit, concrete.
- `*` / `/` → add / subtract dimension exponents; never violate.
- `+` / `-` → require equal dimensions; a lone wildcard side is unified to the concrete
  side; two concretes of different dimension violate.
- `^` (generic power) → base and exponent must both be dimensionless-or-wildcard →
  dimensionless result; else violate. (`square` doubles exponents, `sqrt` halves them
  exactly via the fixed-denominator rational; these are the dimension-aware unary ops.)
- `exp/log/sin/cos/tanh` → argument must be dimensionless-or-wildcard → dimensionless result.
- **Top level**: violate if any subtree violated, or if `y_units` is set and the root is a
  **concrete** dimension `≠ y_units`. A wildcard root (e.g. an expression ending in `*c` for
  a free constant `c`) escapes the `y_units` check — a free constant can absorb any
  dimension. This is faithful to SR.jl; `dimensionless_constants_only=True` closes that
  escape.

Violation is boolean per expression. The penalty is a flat constant (default 1000) added to
the loss (§4), all-or-nothing.

## 4. Implementation

New header-only module `r-package/rsymbolic2/src/rsymbolic/units/`:
- `dimension.hpp` — `Dimension`: 7 SI base exponents as `int32` numerators over a fixed
  denominator `DEN = 25200` (= 2⁴·3²·5²·7), the exact `FixedRational{Int32,25200}` type
  DynamicQuantities.jl uses. `sqrt`/integer-power stay exact; equality is exact integer
  comparison.
- `unit_parser.hpp` — recursive-descent `parse_unit()`.
- `dimensional_analysis.hpp` — `violates_dimensions()` (mirrors the `tree.hpp` postfix stack
  machine, structure-only, one pass per expression), plus `DimAnalysis` and the
  `add_dim_penalty()` loss helper.

**Penalty placement.** `PopMember::loss` is the SSE that the hall of fame, Pareto front and
final model all rank on. To keep a dimensionally-violating expression out of the *reported*
model (not merely out of selection), the penalty is folded into that loss — exactly as SR.jl
does — via `add_dim_penalty` at every loss-assignment site in `evolutionary_search.cpp`
(`initialize_island`, both `evolve_island` child insertions, `optimize_and_simplify_population`,
and both `finalize_costs_and_merge` loops). The context is built once in `run_evolution`.

**Normalisation.** SR.jl adds `dimensional_constraint_penalty` to the per-sample loss (MSE
scale) before dividing by the baseline (= SST/n); rsymbolic2 works in SSE and divides by
`y_norm = SST`. The equivalent additive term in SSE units is therefore `penalty * W` with
`W = n` (or `sum(weights)` when weighted), so the selection cost gains `penalty*W/SST`,
matching SR.jl's `penalty*n/baseline`. `penalty_sse = penalty * W` is precomputed in
`build_dim_analysis`.

**Wiring.** Four trailing positional arguments (`X_units`, `y_units`,
`dimensional_constraint_penalty`, `dimensionless_constants_only`) were appended — order in
lockstep — to both bridges (`rsymbolic2_r.cpp`, `rsymbolic2_py.cpp`), the R wrapper
(`symbolic_regression.R`, with `NULL`→1000 resolution and `length(X_units)==ncol(X)` /
`y_units`-requires-`X_units` validation), the Python wrapper (`__init__.py`, same), and the
regenerated `cpp11.R`/`cpp11.cpp`. The units live on `SearchSpace`; the penalty scalar on
`SearchOptions`.

## 5. Correctness evidence (known-answer tests)

Standalone C++ (`standalone/tests/`), all passing:
- `test_dimension.cpp` (29 checks) — fixed-rational arithmetic, exact `sqrt(square(x))==x`,
  non-divisible roots rejected.
- `test_unit_parser.cpp` (35 checks) — `"kg*m^2/s^2"==J`, `N*m==J`, rational exponents,
  prefix inertness, `cd`≠centi·day, error cases.
- `test_dimensional_analysis.cpp` (20 checks) — `F=m*a` consistent, `F=m+a` violates,
  `sin(dimensioned)` violates, `sqrt(square(x))` round-trip, `y_units` mismatch, wildcard
  unification, `dimensionless_constants_only`, and off-path byte identity of the penalty
  helper.

R `tests/testthat/test-units.R` and Python `tests/test_rsymbolic2.py` — API acceptance,
validation errors, a units-enabled run, no-degradation of a consistent target, and a
mismatched-`y_units` run whose recommended loss carries the penalty
(`penalty_sse = 1000 * n` observed). Full suites: standalone ctest 22/22, R testthat
156/156, Python pytest 20/20 (Windows).

The two wildcard fine points (result-wildcard of `Pow` and of transcendental unaries) use
the conservative reading and are the specific items the SR.jl parity table in §6 confirms.

## 6. Accuracy screen (declared; NOT yet run)

**Question:** does declaring units raise Feynman ground-truth recovery, and does it never
lower it on already-solved problems? Prior evidence tempers expectations: several hard
Feynman failures are retention-bound / deceptive-basin (docs/38, docs/44 §5), not
search-space-size bound, so a space-pruning lever may not rescue them.

**Method (reuses existing harness):**
1. Add per-variable units to the 25-equation Feynman dev subset in
   `benchmarks/feynman_datasets.R` (the Udrescu & Tegmark source ships units); the same
   table is the ground truth for the C++ SR.jl parity assertions.
2. Run the SR.jl reference engine (`benchmarks/05_feynman_pysr_comparison.jl`), which already
   implements `X_units`/`y_units`/`dimensional_constraint_penalty=1000`, ON vs OFF at the
   frozen Stage-1 parity budget, ≥5 seeds. Report medians + spread.
3. Record: problems where ON raises recovery; confirmation ON never lowers a solved problem;
   effect on the Class-B deceptive-basin problems.

**Reproduction:**
```
Rscript benchmarks/export_feynman_data.R          # after adding units to feynman_datasets.R
julia -t N benchmarks/05_feynman_pysr_comparison.jl seeds=5   # ON vs OFF arms
```

Until this is run, no accuracy improvement is claimed; the feature ships on its correctness
and parity guarantees alone.

## 7. Decisions

1. Ship dimensional analysis as an **off-by-default, PySR-compatible** feature (user
   direction), implemented with a dependency-free fixed-rational unit type — no Julia, no new
   C++ dependency.
2. Fold the penalty into `PopMember::loss` (not a selection-only term) so violators are kept
   out of the reported model, matching SR.jl.
3. `docs/29` §C #7 is updated: the feature is now implemented (still default-OFF, so no
   parity change).
4. The Feynman accuracy screen (§6) is the declared next measurement; the feature is trusted
   for *correctness* now and for *accuracy* only once the screen reports.

## See also
- `CLAUDE.md` → "Opt-in high-accuracy options (the second layer)".
- `docs/29_pysr_difference_catalog.md` §C #7 (the parity gap this closes).
- `docs/28_pysr_default_parity_spec.md` (the default layer this sits on top of).
- `docs/43`–`45` (the Phase 0 screen discipline for the opt-in layer).
