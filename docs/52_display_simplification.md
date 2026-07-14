# 52. Display-time expression simplification

Status: implemented (2026-07-12). **Rule set superseded by docs/54** (2026-07-12):
`display_simplify()` is now the two-layer simplifier (deterministic normalisation +
bounded equality saturation); the "Rule set" / "Why a pass loop" sections below
describe the first implementation and are kept as history. Still authoritative here:
the purpose, the frozen-expression compliance section (docs/48 D2), and the
`pow(x,1)` / `t*0` exclusion findings (docs/54 carries them forward).

## Purpose

`simplify()` (`rsymbolic/simplification/simplify.hpp`, docs/29 §11/§12) is a
**parity** simplifier: it exists so the search's own `simplify` mutation and
per-iteration population pass mirror SymbolicRegression.jl's `simplify_tree!` +
`combine_operators` exactly, including SR.jl's own omissions — it does not eliminate
identities (`x+0`, `x*1`, `x*0`, `x/1`, `x^0`, `x^1`), cancel double negation, or fold
`Div`/`Pow` at all. Those omissions are correct *for parity*, but they leave visibly
redundant expressions in front of users: a real GUI bug report surfaced
`((((sin((x0 * -3)) / exp((x0 * 0.5))) * 0.600906) / 0.863117) / -0.696205)`, whose
trailing three-constant division chain `simplify()` is not allowed to touch (folding
it would change search behaviour on every tree with the same shape, breaking parity).

`display_simplify()` (`rsymbolic/simplification/display_simplify.hpp`,
`r-package/rsymbolic2/src/display_simplify.cpp`) is a **second, independent**
simplifier that is free to go further, because it is called **only at result
finalization, on a copy of the reported tree** — never from the search loop, never
written back into `best.tree`/`m.tree`/any tree the search reads. It produces the new
`expression_simplified` (and per-Pareto-member `latex_simplified`) fields; `simplify()`
and the search loop are byte-for-byte unchanged (verified: `simplify.cpp`,
`simplify.hpp`, `mutation_weights.hpp`, and `evolutionary_search.cpp`'s search loop
were not touched by this change; the only edit to `evolutionary_search.cpp` is the two
finalize-time lines described below).

## Rule set

Bottom-up, post-order, fixed-point loop (`kMaxPasses = 8`, stops early once a pass
changes nothing — see "Why a pass loop" below). At each node, after its children are
fully rewritten:

1. **Constant-subtree fold** (identical to `simplify()`'s `fold` stage): a unary or
   binary node whose operand(s) are all constants collapses to one constant, guarded
   by `std::isfinite` on the result. Subsumes `neg(c) -> literal`.
2. **Add/Sub/Mul constant reassociation and canonicalisation** — the same logic as
   `simplify()`'s `combine()` stage (commutative constant-to-the-right canonicalisation
   for `+`/`*`, the four `Sub` nested-constant rewrites), **duplicated** as a local
   ~50-line copy in `display_simplify.cpp` rather than shared with `simplify.cpp`
   (CLAUDE.md's instruction for this file: the two simplifiers must have no code
   dependency on each other, so a future change to one can never silently alter the
   other).
3. **NEW: Div/Mul constant-chain folding**, isfinite-guarded, not part of SR.jl's
   `combine_operators` and therefore not in `simplify()`:
   - `(t * c1) / c2 -> t * (c1/c2)`
   - `(t / c1) / c2 -> t / (c1*c2)`
   - `(t / c1) * c2 -> t * (c2/c1)`
   - `c1 / (t * c2) -> (c1/c2) / t`
   - `c1 / (t / c2) -> (c1*c2) / t`
   - `(c1 / t) / c2 -> (c1/c2) / t`
   - `c1 / (c2 / t) -> t * (c1/c2)`
   - `(c1 / t) * c2 -> (c1*c2) / t`
   Each strictly removes 2 nodes (a 5-node subtree becomes 3). The `c * t` mirror
   shapes need no rules of their own: rule 2's commutative canonicalisation has
   already moved a constant `Mul` operand to the right child by the time rule 3
   inspects the node. `t / c -> t * (1/c)` was
   considered and **rejected**: it does not reduce node count, changes the displayed
   form for no readability gain, and can lose precision relative to a direct division.
4. **Identity elimination**: `t*1`, `1*t`, `t+0`, `0+t`, `t-0`, `t/1` all rewrite to
   `t`. Runs after rule 2/3's canonicalisation, so by construction the identity
   constant (when present) is always the node's right child.
5. **Double negation**: `neg(neg(t)) -> t`.
6. **`t * (-1) -> neg(t)`**: IEEE-exact (multiplying by exactly -1.0 has no rounding
   error), applied unconditionally — display-only, so it is safe even when `neg` is
   outside the user's requested `unary_ops`.

### Excluded on purpose

- **`t * 0`, `0 * t`, `t ^ 0`**: eliminating these would silently discard a NaN/Inf
  that evaluating `t` itself would have produced (e.g. `log(-1) * 0` is `NaN * 0 =
  NaN`, not `0`). Every rewrite in this file must preserve NaN/Inf-ness exactly (see
  "FP-drift and NaN/Inf caveat" below), so these three stay untouched. Verified in
  `standalone/tests/test_display_simplify.cpp`
  (`test_identity_mul_zero_excluded`, `test_identity_pow_zero_excluded`).
- **`t ^ 1 -> t`**: investigated and rejected on a concrete finding, not a hunch. The
  core's `pow(x, y)` (`rsymbolic/expression/dual.hpp`, "safe pow") evaluates the
  `x > 0` branch as `exp(y * std::log(x))`, **not** `std::pow`. For `y = 1` this is
  `exp(log(x))`, a round trip through `log`/`exp` that is **not bit-exact to `x`** for
  almost any `x`:

  ```python
  >>> import math
  >>> math.exp(1.0 * math.log(3.0)) == 3.0
  False   # 3.0000000000000004
  ```

  Measured directly against the core's own `apply_binary<double>(Pow, x, 1.0)` in
  `test_pow_one_not_exact_for_positive_base` (`test_display_simplify.cpp`): every
  tested `x > 0` except one lucky value differs from `x` in the last bit or two.
  Rewriting `t^1 -> t` would therefore silently change the *evaluated* value one more
  rounding step further than the tree it claims to represent, which is a correctness
  problem, not a cosmetic one — so it stays excluded. (For `x < 0` the safe-pow
  integer-exponent branch does use `std::pow(x, 1.0)`, which most libm
  implementations return exactly as `x`, and `x == 0` is exact by construction; the
  `x > 0` branch is the one that fails, and it dominates in practice, so the rewrite
  is excluded unconditionally rather than conditioned on sign.)

## Why a pass loop, not a single traversal

A single bottom-up traversal already fully resolves most chains within one pass
(post-order visits children before the parent, so a rewrite at a child is visible when
its parent is processed in the same call). The loop exists for the one case where a
node changes *kind* mid-visit: rule 6 turns a `Mul` node into a new `Unary(Neg)` node,
which a *later* visit (rule 5, double negation) can only see on the *next* pass — e.g.
`neg(x0) * -1` needs two passes: pass 1 produces `neg(neg(x0))` (rule 6 fires on the
outer `Mul`), pass 2 collapses that to `x0` (rule 5 fires on the newly created
`Neg(Neg(...))`). `kMaxPasses = 8` is a generous bound; every mutating rule strictly
decreases node count except rule 2's canonicalise-only swap (no size change), so the
sequence terminates quickly in practice.
`test_display_simplify.cpp` asserts `passes_used < kMaxPasses` (via the header's
optional `int* passes_used` out-parameter, default `nullptr` so the finalize call site
stays a plain one-argument call) on every fixture, including a property test over 300
random trees — the cap is never exhausted.

## Frozen-expression rule compliance (docs/48 D2)

docs/48 D2 established that `expression`/`recommended` strings are frozen: both
`predict()` implementations `eval()` them directly, so any renderer bug in a
*different* rendering path can never produce a silent wrong prediction — it is
display-only, of the sort `latex` was for D2 itself.
`expression_simplified`/`recommended_simplified`/`latex_simplified` are new columns
that follow exactly that precedent, not a new exception to it:

- `SearchResult::expression` (search-loop-facing) is unchanged; `predict()` in both R
  and Python still evaluates `expression`/`recommended` verbatim, never
  `expression_simplified`/`recommended_simplified`. Verified in
  `test-display-simplify.R` and `test_display_simplify.py`.
- `SearchResult::expression_simplified` is a *new*, separate field
  (`rsymbolic/search/evolutionary_search.hpp`), set once at finalization:
  `result.expression_simplified = to_string(display_simplify(best.tree));` — note
  `best.tree` is passed **by const reference** into `display_simplify`, which builds
  its own internal AST copy and never mutates its argument.
- Same pattern in the Pareto-front loop of all three bridges (R `rsymbolic2_r.cpp`,
  WASM `rsymbolic2_wasm.cpp`, Python `rsymbolic2_py.cpp`): `display_simplify(m.tree)`
  is called on each member's tree to produce a local `Tree simplified`, which is
  rendered and discarded; `m.tree` itself is never assigned to.
- R additionally computes `recommended_simplified` alongside the existing
  `recommended`, mirroring how `recommended` itself is derived from
  `pareto_front[best_index]`. WASM/Python were kept to the two-field pattern that maps
  directly onto their pre-existing `expression`/`latex` shape (a user can already get
  the Pareto-best simplified form via `pareto_front[best_index]["expression_simplified"]`).

## FP-drift caveat

Like `simplify()`'s own reassociation pass (documented in `simplify.hpp`), several
`display_simplify()` rules reorder floating-point operations relative to the original
tree (rule 2's canonicalisation/merging, rule 3's constant-chain folding). The
evaluated value can therefore shift by a rounding step — this is the same caveat PySR
documents for its own `sympy_format`/`simplify` display path, not a regression
specific to this feature. Every rewrite that performs new arithmetic (rules 1 and 3)
is `isfinite`-guarded, so a rewrite is only ever applied when its result stays finite;
rules 4–6 (identity elimination, double negation, `*-1 -> neg`) perform no arithmetic
at all and are therefore exact. The property test in `test_display_simplify.cpp`
(`test_semantics_preserved_random`, 300 random trees over `{Add, Sub, Mul, Div}` and 7
unary ops, evaluated at 5 points each) asserts NaN/Inf-ness matches *exactly* between
the original and simplified trees, and finite values agree to a hybrid
absolute/relative tolerance of `1e-9`.

## Verification results

- **Standalone (Windows, Rtools/MinGW g++):** new test target `test_display_simplify`
  — 3494 checks, all passed (includes the GUI-regression fixture, all eight Div-chain
  shapes plus one nested-chain case, all six identity/negation rules, two NaN/Inf fold
  guards, the `pow(x,1)` inexactness finding, and the 300-tree property test).
  Regression net re-run clean: `test_simplify` (1820 checks), `test_to_latex` (32
  checks), `test_evolutionary_search` (30 checks) — `simplify()`'s own behaviour is
  unaffected.
- **R (Windows, Rtools45):** `R CMD INSTALL` from source succeeds with no warnings
  from the new files; full `testthat::test_dir()` — **186 passed, 0 failed, 25
  skipped** (pre-existing `# On CRAN`-gated recovery/timing tests, unrelated to this
  change). `test-output.R`'s `expect_named()` checks were updated to include the two
  new top-level fields and the two new `pareto_front` columns (additive, so this is
  the only pre-existing test file that needed a change).
- **Python (Windows):** `pip install --no-build-isolation ./python` succeeds; full
  `pytest python/tests/` — **38 passed** (34 pre-existing + 4 new in
  `test_display_simplify.py`, covering the new fields on a real fit, `predict()`/
  `get_best()` non-interference, and backward compatibility when a raw dict predates
  the new keys).
- Ubuntu/WSL and WASM verification are deferred to the orchestrator's milestone pass
  per this task's scope (not run here).
