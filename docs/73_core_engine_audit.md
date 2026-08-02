# 73 — C++ core engine audit

Full read-through of the shared C++ core (`r-package/rsymbolic2/src/`, ~9,600 lines):
search loop, constant optimisation, mutation/crossover, tree representation,
simplification, parser, units, and the binding boundary.

**Headline: no defect was found on the default search path.** The PySR-parity default
configuration and its search trajectory are sound. Everything fixed below is either an
opt-in feature, an input parser, a deadline-expiry corner, or defensive hardening of a
path the shipped entry points cannot currently reach.

Every change in this document is covered by the bit-invariance gate: `diag_search_digest`
produces a byte-identical 660-line golden before and after.

## What was fixed

### A1 — use-after-move of `y` (`evolutionary_search.cpp`)

`run_evolution` takes `std::vector<double> y` by value and moves it into the shared
`Dataset`. The `linear_scaling` finalize block then computed

```cpp
const double mean_y = y.empty() ? 0.0 : accumulate(y.begin(), y.end(), ...) / y.size();
```

from the moved-from vector. A moved-from `std::vector` is valid but unspecified — in
practice empty — so `mean_y` was always `0.0`. The comment fifteen lines above already
warned "read y back through `data` below, since the local has been moved from"; this one
site did not.

Effect (opt-in `linear_scaling` only): the identity test for the fitted affine map uses
tolerance `1e-12 * (1 + |mean_y|)`, which collapsed to a fixed `1e-12`. On targets with
large `|y|`, maps that are numerically the identity were still materialised, adding 4
nodes to every reported member and inflating its complexity.

Fixed by reading `data->y`.

### A2 — unary minus bound tighter than `^` (`parse_expression.hpp`)

The grammar had `factor := unary ('^' factor)?`, so `parse_factor` consumed a leading `-`
before it ever looked at `^`:

| input | was | R / Python / standard notation |
|---|---|---|
| `-x^2` | `(-x)^2` (always ≥ 0) | `-(x^2)` |
| `-2^2` | `+4` | `-4` |

`-2^2` was the worse case: the literal-folding shortcut in `parse_unary` built the
constant `-2` and then squared it. Nothing errored — the macro silently became a
different function.

The grammar is now

```
term    := unary (('*' | '/') unary)*
unary   := '-' unary | power
power   := primary ('^' unary)?      [right-associative]
```

so `-x^2` is `-(x^2)`, `2^-3` is `2^(-3)`, and `2^3^2` stays right-associative. Literal
folding is suppressed when the literal is the base of a power.

Blast radius is the opt-in macro-operator body parser (`macro_op.hpp`), the parser's only
caller. Engine output is unaffected: `to_string` renders `neg` parenthesised as `(-a)`
(docs/71), so no frozen expression is re-read differently.

### A3 — malformed numeric literals silently truncated (`parse_expression.hpp`)

The tokenizer scanned a run of digits-and-dots and handed the whole span to `strtod`.
`strtod` stops at the second `.`, but the scan advanced past it, so `"1.2.3+x"` parsed as
`1.2 + x` with no error. Now at most one decimal point is accepted; a second reports
`malformed number` through the file's existing bool+message convention.

### B1 — abort-contaminated SSE reported as a successful fit

When the stop predicate fires mid-evaluation, the residual closure fills the points it
never reached with `kLargeResidualSentinel = 1e10` — deliberately **finite**, so it
cannot poison JᵀJ with NaN/Inf.

Every evaluation inside the LM loop is followed by an `aborted()` check that discards the
result. The **first** residual evaluation of each `run_lm_from`, and the `k == 0` fast
path, were not: they returned `loss = <real residuals mixed with sentinels>` and
`success = isfinite(loss)`, which is unconditionally `true` because the sentinel is
finite. `RandomRestartOptimizer::evaluate` had the same hole, made reachable by
`result.loss` starting at `+Inf` so any finite value wins the first restart.

The contamination always biases the loss upward, so no fabricated *good* solution could
enter the archive. But `optimize_and_simplify_population` only tests `isfinite(loss)`, so
the member's loss and constants were overwritten with a fabricated number instead of the
pre-optimisation member being kept.

Fixed by reporting `+Inf` / `success = false` when the first evaluation aborts. A stop
that fires on any *later* iteration still returns the genuine progress made before it —
that is the documented "best result so far" contract and it is now covered by a test in
both directions.

**Test change worth flagging.** `test_optimizer_stop.cpp` previously asserted
`success == true` and a finite loss for exactly this case, on the reasoning "UserAsked is
not ImproperInputParameters" — a leftover from the removed Eigen backend, where `success`
meant "the solver accepted the input". Under SelfLM the field is documented as "true iff a
finite-loss solution was found", and a sum of sentinels is not a solution. The test
encoded the bug; its assertion was inverted and a companion test added.

### B2 — `n_restarts` doc comment contradicted the default backend

`OptimizerConfig::n_restarts` was commented "number of restarts (including the initial)".
That is true of `RandomRestartOptimizer` (non-default) and false of `SelfLMOptimizer` (the
default), which runs start 0 unconditionally and then `n_restarts` *additional* perturbed
starts.

**PySR parity was already correct** — SR.jl's `_optimize_constants` loops
`for _ in 1:optimizer_nrestarts` after an unconditional first fit, and the search default
`{seed, n_restarts = 2, max_iterations = 8, perturbation_scale = 0.5}` is exactly
`optimizer_nrestarts = 2`. Only the comment was wrong; it now states both conventions.

### C1 — `gen_random_tree_fixed_size` could spin forever (`random_tree.cpp`)

The growth loop's general branch had no progress guard. `append_random_op` returns `false`
when no operator fits, leaving the tree unchanged, so `cur_size` stopped advancing and the
calling thread hung with no diagnostic.

Not reachable from the search today (the R layer requires a non-empty `binary_ops`, and
`randomize_tree` draws `node_count` from `[1, max_nodes]`), but this is a public entry
point. It now breaks and accepts a short tree — the same outcome the unary branch already
took for SR.jl's `nuna == 0 && break`.

### C2 — unbounded recursion in the rotate mutation (`mutation.cpp`)

`rotate_tree` builds a temporary pointer tree, walks it, and tears it down. `build_pnode`
was already iterative; `serialize_pnode`, `collect_pnodes` and the `unique_ptr` teardown
chain were not, and all three recurse to the tree's structural depth. A chain of nested
unary operators has depth equal to its node count, and `max_nodes` is user-settable with
no upper bound. `rotate` carries mutation weight 4.26 — one of the highest — so this is an
ordinary path, not a corner.

**Fixing only the traversals would have moved the crash rather than removed it**: the
`~unique_ptr` chain overflows at the same depth. All three are now iterative, with an
explicit `destroy_pnode` teardown.

Emission order is load-bearing — `collect_pnodes`' preorder decides which node a rotation
lands on, so a reordering would move the search trajectory even with identical RNG
consumption. Order is preserved exactly (`serialize_pnode` postfix `l, r, self`;
`collect_pnodes` preorder `self, l, r`), verified by the digest gate and by a fixed-seed
stability test.

### C3 — `Dataset` accepted ragged input (`least_squares_problem.hpp`)

The constructor validated neither `Xcol[j].size() == y.size()` nor
`weights.size() == y.size()`. The SoA evaluators derive tile bounds from `m = y.size()`
alone, so a short column is an out-of-bounds heap read, not a wrong answer. Unreachable
from inside the engine, reachable from a binding or from `columns_from_rows` on a ragged
matrix. Now throws `std::invalid_argument`; the check runs once per run, off every hot
path.

## Recorded, deliberately not fixed

### D1 — class-B e-graph rules can flip finiteness

`factor`, `distrib`, the `*-assoc-*` pairs and the division-chain rules reassociate or
redistribute, which changes *where* an intermediate overflows. For
`x0*x1 + x0*x2` at `x0 = 1e300, x1 = 1e300, x2 = -1e300` the written form is
`Inf + (-Inf) = NaN` while the factored form the extractor prefers is `0` — from entirely
finite inputs.

docs/54 already carried a class-B caveat but claimed "finite inputs stay finite,
non-finite stay non-finite", which this case disproves; that claim is corrected there.

Not fixed, because these rules are the size-reduction driver and the divergence is
confined to the display-only `expression_simplified` string. `predict()`, the hall of fame
and every search decision read the untouched tree.

## Verified correct — no re-check needed

- **Every operator's derivative** in `dual.hpp`, `multi_dual.hpp` and `soa_eval.hpp`
  (`+ - * / neg exp log sin cos sqrt tanh abs square recip erf sinh cosh pow`), checked
  against calculus by hand. The scalar, batched and SIMD representations agree in formula
  *and* operation order, including NaN-gradient propagation for `sqrt` of a negative.
- LM damping and step acceptance (λ×10 / λ×0.1, `kMaxInner = 30`), `solve_spd` Cholesky
  including non-positive-definite detection, JᵀJ symmetrisation, and the bit-exactness of
  the blocked normal-equation accumulation. No unbounded loop.
- Postfix index arithmetic for every mutator and both crossover variants — no off-by-one;
  `reindex_constants` is called after every structural edit and correctly omitted after
  the two non-structural ones.
- Every `uniform_int_distribution(0, n-1)` construction is guarded by an emptiness or
  feasibility check, so the `n = 0` underflow to `SIZE_MAX` cannot occur.
- `HallOfFame::update / pareto_front / merge / select_best`.
- `eval_cache.hpp`'s hash + structural-equality double check: collisions degrade to
  misses, never to wrong hits.
- e-graph and display-simplify **determinism**: every `unordered_map`/`unordered_set`
  iteration is followed by a sort under a strict total order, so hash order never reaches
  the output. The previously-recorded `expression_simplified` non-determinism is not
  present in the current code (removed by docs/66).
- `dimension.hpp` / `dimensional_analysis.hpp` rational-exponent arithmetic and wildcard
  unification, including negative-modulo edges.
- `latex.hpp` / `sympy.hpp` precedence and parenthesisation tables.
- `platform_libm.cpp`'s `g_ucrt` global — constant-initialised, so no race with
  `g_resolved`'s dynamic initialiser.
- No mutable `static` or `thread_local` state anywhere in the core; the DLL/libgomp
  pitfall recorded in docs/65 has not regressed.

## Verification

- `diag_search_digest`: byte-identical, 660 lines, before vs after.
- standalone ctest: 30/30 pass on Windows (Rtools/MinGW/UCRT).
- New tests: power precedence and malformed literals (`test_macro_op`), abort contract in
  both directions (`test_optimizer_stop`), fixed-size termination (`test_random_tree`),
  deep-chain rotation and order stability (`test_mutation_operators`), ragged-input
  rejection (`test_constant_fitting`).
