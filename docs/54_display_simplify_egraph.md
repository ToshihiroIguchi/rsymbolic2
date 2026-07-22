# 54. Two-layer display simplification (normalisation + bounded equality saturation)

Status: implemented (2026-07-12). Supersedes the rule-set section of docs/52 (the
docs/52 sections on the frozen-expression contract and the pow/`x*0` exclusion
findings remain authoritative and are incorporated below).

## Purpose and scope

`display_simplify()` (`rsymbolic/simplification/display_simplify.hpp`,
`display_simplify.cpp`, `egraph.hpp`, `egraph.cpp`) produces the display-only
`expression_simplified` / `latex_simplified` companions. docs/52 introduced it as a
small fixed-point rewriter (constant chains + identities); this change replaces the
internals with a two-layer simplifier. Everything scope-related from docs/52 is
unchanged:

- **Called only at result finalization on a copy of the reported tree** — never from
  the search loop, never written back into `best.tree` / `m.tree`.
- **Frozen-expression contract (docs/48 D2):** `expression` / `recommended` stay the
  raw strings and remain the `predict()` round-trip source; the simplified fields are
  new columns beside them. No output-surface shape changed in R / Python / WASM
  (single-argument call sites compile unchanged).
- **Parity:** `simplify()` (the SR.jl-parity search simplifier) and the search loop
  are untouched. Bit-invariance evidence below.
- Layer 2 shares no code with `simplify.cpp`; the display path and the parity path
  still cannot influence each other.

## Architecture

### Layer 1 — deterministic normalisation (always runs)

Cohen-2003-style automatic simplification, applied bottom-up to a fixpoint (pass cap
4; each pass is non-increasing in node count):

- **Constant folding**, isfinite-guarded (a fold producing NaN/Inf is skipped).
- **Sum flattening:** `+`/`-`/`neg` spines flatten into an n-ary sum of
  (coefficient, core) terms plus a constant accumulator. `x - y` ≡ `x + (-y)` and all
  negation moves are IEEE-exact. Like terms merge: `c1*t + c2*t -> (c1+c2)*t`
  (guarded; see FP policy for the coefficient-0 case). Rebuild order: canonical order
  with positive-coefficient terms leading (a negative leader would cost an extra
  `neg` node); if every term is negative the constant leads (`c - t1 - t2`); the
  constant otherwise trails (`... + c` / `... - |c|`).
- **Product flattening:** `*`//`/`neg` spines flatten into numerator/denominator
  factor lists; nested divisions flip polarity (`a/(b/c)` puts `c` in the numerator).
  Constants merge into a single coefficient `q = cn/cd` (guarded; on a non-finite
  merge the constants stay as literal factors, e.g. `(x0 * 5) / 0` is preserved).
  Display form: `(N / D) * q` with the constant on the right, `q / D` when the
  numerator is empty, `N * q` when the denominator is empty — reproducing every
  docs/52 constant-chain shape, including `t / c` kept as a **division** when the
  numerator carries no constant (docs/52's precision/readability rejection of
  `t/c -> t*(1/c)`, carried forward).
- **Like factors:** structurally equal factors pair into `square(f)` (bit-exact:
  `square(double)` evaluates as `f*f`), recursively (`x*x*x*x -> square(square(x))`).
- **Canonical ordering** of commutative operand lists (variables < unaries <
  binaries < constants; deterministic structural compare) — reassociation drift, see
  FP policy.
- **Negation normal form:** `neg(neg t) -> t`, `neg(a - b) -> b - a` (IEEE-exact),
  sign absorption into product/sum constants, `t * -1 -> neg(t)`, `0 - t -> neg(t)`.
- **Identities:** `t+0`, `t-0`, `t*1`, `t/1`, and their mirrored forms via
  canonicalisation. (`t*0`, `t^0`, `t^1` stay; see exclusions.)
- **Exact unary rewrites:** `abs(neg t)`, `abs(abs t)`, `abs(square t)`,
  `abs(sqrt t)`, `abs(exp t)` drop the redundant `abs`/`neg`; `square(neg t)`,
  `square(abs t)` drop the inner op; odd/even pulls `sin(neg t) -> neg(sin t)`,
  `tanh(neg t) -> neg(tanh t)`, `cos(neg t) -> cos(t)`; and `sqrt(square t) ->
  abs(t)` (the one drift+overflow-caveat unary rule, class B below).

### Layer 2 — bounded equality saturation (egg-style e-graph)

`egraph_simplify()` seeds an e-graph with the Layer-1 tree and saturates the audited
rule set below, then extracts the **minimum-node-count** (= SR complexity) equivalent
term. Self-contained implementation (no new dependency): hash-consed e-nodes over
union-find e-classes; congruence closure by whole-graph re-canonicalisation sweeps
(simpler than egg's incremental parent repair, ample at this budget); top-down
e-matching; per-class constant analysis (the same isfinite-guarded folding as
Layer 1, egg's "modify" adds the folded literal to the class); extraction by
fixpoint cost relaxation with a deterministic total-order tie-break. References:
egg (Willsey et al., POPL 2021, arXiv:2004.03082); Herbie (PLDI 2015) and Ruler
(OOPSLA 2021, arXiv:2108.10436) for rule-audit practice; Caviar (arXiv:2111.12116)
for bounded-budget saturation.

**Limits (hard caps, `EGraphLimits`):** 10 iterations, 10 000 e-nodes, 10 ms
wall-clock per expression. Iteration/e-node caps are deterministic; the wall-clock
cap is a safety net (the only non-deterministic stop — see Determinism).

**Fallback contract:** the Layer-2 result is adopted **only when strictly smaller**
than the Layer-1 tree (after a final Layer-1 re-normalisation for canonical display
form). On any limit hit, non-improvement, or tie, the Layer-1 result is returned.
Because the input is itself in the e-graph, extraction can never be larger than the
input; the strict-improvement gate additionally keeps every docs/52 display shape
stable. `limits.max_iterations = 0` disables Layer 2 entirely (used by tests).

What Layer 2 adds over Layer 1 is cross-structure discovery, e.g. factoring
`x0*x1 + x0*x2 -> x0*(x1 + x2)` (7 -> 5 nodes), factoring hidden under sub/neg/div
juggling, and `square`/`abs` recombination across product boundaries.

## Rule inventory (Layer 2)

Floating-point classes: **A** = IEEE-exact (bit-identical evaluation);
**B** = exact over the reals, evaluated value may shift by rounding steps
(reassociation/redistribution — the same caveat `simplify()`'s combine pass and
PySR's sympy display path carry).

| Group | Rules (directions spelled out in `egraph.cpp build_rules()`) | Class |
|---|---|---|
| Commutativity/associativity | add-comm, mul-comm, add-assoc-l/r, mul-assoc-l/r | B |
| Sub/neg | sub-to-add-neg, add-neg-to-sub, neg-neg, neg-sub (`-(a-b) -> b-a`), sub-neg, neg-mul-in/out, neg-as-mul, mul-neg-one, div-neg-num-out/in, div-neg-den | A |
| Division chains | div-div-num, div-mul-den, div-div-den, mul-div-l, div-of-mul, mul-of-div | B |
| Identities | add-zero, sub-zero, mul-one, div-one (literal `+0.0` / `1.0`, bit-matched) | A |
| Distributivity/factoring | distrib, factor, factor-one, add-self (`a+a -> a*2`), div-add-common | B |
| Square | mul-self (`a*a -> square(a)`; exact — `square` evaluates as `a*a`), square-expand, square-neg, square-abs, square-mul(+join), square-div(+join) | A / B (mul/div splits are B) |
| Abs | abs-neg, abs-abs, abs-square, abs-sqrt, abs-exp, abs-mul(+join), abs-div(+join) | A |
| Sqrt | sqrt-square (`sqrt(square a) -> abs(a)`) | B + overflow caveat |
| Odd/even | sin-neg-out/in, tanh-neg-out/in, cos-neg | B (libm sign symmetry) |

56 directed rules total, plus the constant-folding analysis (an unbounded rule
family, isfinite-guarded).

## Floating-point policy

Every adopted rewrite is an exact identity over the reals on the expression's
evaluation domain, and **no rewrite may change whether the expression evaluates to a
finite value** (`test_semantics_preserved_random` asserts finite-ness equality at
sample points over 300 random trees, plus value agreement at 1e-9 hybrid tolerance).
The displayed value therefore never diverges from `predict()` beyond the rounding
drift class B describes.

**Excluded rules, with reasons (adoption decisions requested by the task):**

- `x*0 -> 0`, `0*x -> 0`, `x^0 -> 1`: discard a NaN/Inf that evaluating `x` would
  produce (docs/52 finding, kept).
- `x - x -> 0`, `x/x -> 1`: NaN at `x = ±Inf` (and `0/0`) becomes finite.
  Consequently, when like-term merging in Layer 1 sums coefficients to exactly 0 the
  term is kept as `core * 0` — never a bare `0` — preserving NaN propagation
  (`x - x -> x * 0`; at `x = Inf` both are NaN, verified in
  `test_cancellation_keeps_zero_times_term`).
- `t^1 -> t`: the core's safe pow evaluates `x>0` as `exp(1*log(x))`, which is not
  bit-exact to `x` (docs/52 finding, verified by
  `test_pow_one_not_exact_for_positive_base`).
- **All other Pow rewrites** (`pow(t,2) <-> square(t)`, exponent addition, etc.):
  safe_pow's guarded fallback returns **0** for NaN/undefined bases (e.g.
  `safe_pow(NaN, 2) = 0` but `square(NaN) = NaN`), so essentially any Pow rewrite
  changes NaN behaviour. Only fully-constant Pow subtrees fold (guarded).
- `exp(log x) -> x`: NaN for `x < 0` becomes finite.
- `log(exp x) -> x`: `exp` overflows to Inf for `x ≳ 709.8`, so Inf-ness changes on
  reachable inputs.
- `square(sqrt x) -> x`: NaN for `x < 0` becomes finite.
- `log`/`exp`/`sqrt` product-sum expansions (`log(ab) = log a + log b`, …): domain
  splits (negative operands) change NaN behaviour.

**Adopted with documented caveats:**

- `sqrt(square t) -> abs(t)`: exact over the reals; `square` can overflow to Inf for
  `|t| > ~1.3e154`, making `sqrt(square t) = Inf` where `abs(t)` is finite. This is
  the same intermediate-overflow class as constant-chain reassociation (accepted by
  docs/52) and only fires beyond 1e154.
- Like-term/factoring merges (class B) can exchange **NaN for Inf** in mixed-sign
  overflow corners (`3t + (-2t)` at `t = Inf` is NaN, merged `1*t` is Inf); finite
  inputs stay finite, non-finite stay non-finite. A merged coefficient of exactly
  `-0.0` is normalised to `+0.0`, and zero-coefficient terms keep the `* 0` factor,
  so the sign of a produced zero (not its finiteness) is the only zero-level drift.
- Odd/even libm rules assume sign-symmetric `sin`/`cos`/`tanh` (true of every libm
  in the support set; IEEE 754 recommends it). At worst this is class-B drift.

## Determinism

Given the same tree on the same platform, the output is fully deterministic
(sorted node lists drive matching and extraction; extraction ties break on a total
order over e-nodes, not hash order). Two caveats, both display-only:

- The wall-clock cap (10 ms) is the one non-deterministic stop; hitting it can only
  cause a fallback toward the Layer-1 form (which is itself deterministic).
- Across platforms (native vs WASM), libm ULP differences and merge-order-dependent
  representative ids can pick different equal-size forms — the same cross-platform
  caveat docs/51 already accepts for the search itself. Within a platform, repeated
  runs agree.

## Measurements (Windows, Rtools g++ 14.3, -O2, i7-class desktop)

300 random trees per row (`generate_random_tree`, 4 binary + 7 unary ops), default
limits:

| max_depth | nodes in→out (mean) | shrunk | L2 adopted | saturated | ms median / p90 / max |
|---|---|---|---|---|---|
| 4 | 5.6 → 4.2 | 140/300 | 10 | 153 | 0.009 / 0.38 / 11.3 |
| 6 | 7.9 → 5.5 | 151/300 | 14 | 123 | 0.003 / 1.23 / 14.6 |
| 8 | 11.5 → 8.3 | 175/300 | 40 | 114 | 0.012 / 6.4 / 13.2 |

Median cost is microseconds; the p90 stays in the low milliseconds; the 10 ms cap
binds only on the pathological tail (max observed 14.6 ms including extraction).
Reducing the cap from 50 ms to 10 ms changed **no** output in these runs (identical
node reductions and adoption counts) — the cap is genuinely a safety net. At
finalize time the function runs once per Pareto member (≤ ~25 calls).

## Verification

- **Standalone:** rewritten `test_display_simplify` — 3253 checks, all passed
  (every docs/52 fixture shape preserved: the GUI regression chain, all eight
  div-chain shapes, identities, exclusions, fold guards; new: like terms,
  `square` collection, sub normal forms, unary rewrites, Layer-2 factoring with
  stats assertions, tiny-limit fallback equivalence, Layer-1 idempotence over 100
  random trees, and the 300-tree property test with strict finite-ness matching).
  Full ctest suite: 27/27 passed.
- **Search bit-invariance (the docs/48-D2/parity gate):** a fixed-seed probe
  (2 seeds × 2 operator sets, 4 islands) built from clean HEAD vs. this change
  printed every Pareto member's complexity, loss (`%a` bits), raw expression and
  constant bit patterns: **byte-identical** across the change, and run-to-run
  deterministic on both builds. (An earlier draft of the probe passed the dataset
  column-major; `run_evolution`'s `X` is **row-major** — `X[i]` is one data point —
  and the mis-shaped input read out of bounds inside the SoA evaluator and looked
  like nondeterminism. The bridges validate shapes; the raw C++ entry point assumes
  them.)
- **R (Windows, Rtools45):** `install.packages(type="source")` clean;
  `testthat::test_dir` — 186 passed, 0 failed, 25 CRAN-gated skips. (Note: a header
  change under `src/rsymbolic/` requires deleting stale `src/*.o` first — the R
  build has no header dependency tracking.)
- **Python (Windows):** `pip install --no-build-isolation ./python`; pytest — 38
  passed.
- **WASM:** rebuilt via `web/wasm/build.ps1` (emsdk 6.0.2); Node parity test
  passed (including the docs/53 bit-identity gates); GUI checked in-browser — the
  quadratic example displays the simplified equation table/detail and KaTeX
  rendering correctly.
- **Ubuntu (WSL):** deferred to the next milestone pass per CLAUDE.md's
  verification cadence.

## Phase 2 (not in this change)

Search-time strong simplification (opt-in, default OFF, CLAUDE.md second layer) is
now implemented as `strong_simplify` (all bindings, default OFF). Its Feynman screen
(`docs/55`) returned **NO-GO**: a small, mixed accuracy effect (+1 threshold and +2
structural majority recoveries, 0 regressions) that falls below the pre-registered
GO bar, so it ships as a documented experimental opt-in rather than a recommended
option — the same outcome as `linear_scaling` (docs/50).
