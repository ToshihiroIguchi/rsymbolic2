# 72. Is the display simplifier weaker than PySR's sympy path? A measurement

**Date:** 2026-08-01
**Status:** DONE — verdict **NO-GO**. No rules are added, no budget is raised; the
display simplifier ships unchanged. This document records the evidence, including two
facts that correct earlier documents (§5, §6).

## 1. The question

`display_simplify()` (docs/52 → docs/54 → docs/66) is a hand-written two-layer
simplifier: Cohen-style normalisation plus a bounded e-graph, ~56 rewrite rules, no
third-party dependency. PySR reaches for sympy instead. The natural worry is that a
hand-written rule set is simply a weaker CAS, and that users therefore see longer
equations than PySR shows them.

Two separate comparisons are needed, because they have different answers:

1. against **what PySR actually runs** (§2), and
2. against **what sympy can do when fully invoked** (§3-§4), which PySR does *not* do.

Versions used throughout: PySR **1.5.10**, SymbolicRegression.jl **1.11.3**,
DynamicExpressions **1.10.4**, sympy **1.14.0** (Python 3.13.14), rsymbolic2 at
`8e877e3`, g++ 14.3.0 `-O2`, Windows 11.

## 2. What PySR actually does at display time

PySR's user-facing equation strings — `equations_["equation"]`, the printed table,
`get_best()` — are **the raw Julia strings**, carrying nothing beyond the search-time
parity simplifier (`simplify_tree!` + `combine_operators`, which rsymbolic2 matches
exactly; docs/29 §11). There is no finalisation-time simplification pass.

Only `model.sympy()` / `model.latex()` go through sympy, via
`pysr/export_sympy.py::pysr2sympy`:

```python
try:
    return sympify(equation, locals=local_sympy_mappings, evaluate=False)
except TypeError as e:
    if "got an unexpected keyword argument 'evaluate'" in str(e):
        return sympify(equation, locals=local_sympy_mappings)   # full evaluation
    raise
```

`evaluate=False` means sympy's automatic evaluation is switched **off**, so the normal
path performs essentially no simplification. But sympy's `EvaluateFalseTransformer`
injects `evaluate=False` into calls whose name is in its `functions` list — which
includes `sqrt` and `log` — and PySR maps those two to *lambdas*
(`"sqrt": lambda x: sympy.sqrt(x)`), which cannot accept the keyword. The `TypeError`
propagates to the top and the **whole expression** is re-parsed with full evaluation.

The observable consequence is that whether an equation is simplified at all depends on
whether it happens to contain `sqrt` or `log`:

| Julia string | branch taken | `model.sympy()` result |
|---|---|---|
| `(x0 - x0) + 2.0*3.0` | `evaluate=False` | `-x0 + x0 + 2.0*3.0` |
| `sqrt(x0) + (x0 - x0) + 2.0*3.0` | fallback | `sqrt(x0) + 6.0` |
| `exp(x0) + (x0 - x0)` | `evaluate=False` | `-x0 + x0 + exp(x0)` |
| `log(x0) + (x0*1.0)` | fallback | `1.0*x0 + log(x0)` |
| `square(x0)/x0 + 0.0` | `evaluate=False` | `0.0 + x0**2/x0` |
| `(x0 * 2.0) * 3.0` | `evaluate=False` | `x0*2.0*3.0` |

Against this baseline rsymbolic2 is not weaker; it is strictly stronger and, unlike the
above, consistent. That disposes of comparison (1). Everything below is comparison (2),
against sympy's real capability.

## 3. Where the rule set is genuinely weaker than sympy

Four categories, all of them deliberate exclusions from the docs/54 floating-point
policy ("no rewrite may change whether the expression evaluates to a finite value"),
not gaps in coverage:

| input | `display_simplify()` | `sympy.simplify()` | why excluded |
|---|---|---|---|
| `(x0 * x1) / x0` | unchanged | `x1` | no cancellation: `x/x -> 1` turns NaN (`±Inf`, `0/0`) finite |
| `square(x0) / x0` | `(x0^2)/x0` | `x0` | all Pow rewrites excluded (`safe_pow(NaN,2)=0` vs `square(NaN)=NaN`) |
| `x0^2 * x0^3` | unchanged | `x0^5` | same |
| `sqrt(x0)*sqrt(x0)` | `sqrt(x0)^2` | `x0` | domain: NaN for `x0 < 0` becomes finite |
| `exp(log(x0))` | unchanged | `x0` | domain |
| `exp(x0)*exp(x1)` | unchanged | `exp(x0+x1)` | domain / overflow |
| `sin(x0)/cos(x0)` | unchanged | `tan(x0)` | no `tan` operator (operators are a problem input, CLAUDE.md) |
| `square(sin)+square(cos)` | unchanged | `1` | no trigonometric identities |
| `(x0 + x1) - x1` | `x0 + (x1 * 0)` | `x0` | cancellation to 0 is excluded (NaN preservation) |
| `x0 - x0` | `(x0 * 0)` | `0` | same |

The exclusions are the price of the guarantee that the displayed expression never
diverges from `predict()` in its NaN/Inf behaviour. The reverse also happens, though
less often: `sqrt(square(x0))` renders as `abs(x0)` (2 nodes) here and stays
`sqrt(x0**2)` (3 ops) in sympy, and sympy also leaves `log(exp(x0))` alone.

## 4. What that costs on real output

Sample: **1750 distinct expressions** from the stored Pareto fronts in
`benchmarks/results/*front*.csv` (20 files) — real search output, not random trees.
Caveat: those runs enabled `pow`, so 69 % of the sample contains `^`; a `pow`-free
operator set would exercise §3's Pow rows less.

Each expression was rendered by `display_simplify()` and, independently, by
`sympy.simplify()` on the fully-evaluated sympify of the same string. Comparing node
counts directly is unfair to sympy (it prints `a - b` as `-b + a`, adds `1.0*` factors),
so both renderings were re-normalised through rsymbolic2's own Layer 1+2 before
counting. 146 expressions (every 12th) were compared; sympy's cost is what limits the
sample size.

| outcome | count | share |
|---|---|---|
| `sympy.simplify` strictly smaller | 2 | **1.4 %** |
| identical node count | 81 | 55.5 % |
| `display_simplify` alone smaller | 62 | 42.5 % |
| sympy output not representable in our grammar | 1 | 0.7 % |

Cost: **2 ms** vs **120 ms** per expression. The raw (not re-normalised) count gives
2 / 64 / 79 / 1 — the same conclusion.

sympy loses as often as it does because it optimises a different objective: its
`count_ops` over its own canonical form, which happily distributes
(`-0.062 + 5.36/(x2*x3) - 1.758*x1/(x0*x2*x3)`), whereas Layer 2 extracts the
**minimum-node-count** equivalent — which is exactly SR complexity.

**Both of the two expressions where sympy won were Pow algebra**: `(a^p)^q -> a^(p*q)`
and `(a*c)^p -> c^p * a^p` followed by constant folding. If this screen is ever
revisited, those two rules are the entire measured opportunity.

## 5. Layer 2 earns much less on real output than docs/54 suggested

Over the full 1750-expression sample:

```
shrunk by display_simplify   : 775/1750  (44 %)
Layer 2 (e-graph) adopted    :  49/1750  (2.8 %)
output still containing "* 0":   3/1750  (0.2 %)
```

docs/54's measurement table used `generate_random_tree` at `max_depth <= 8` (5-12 nodes
on average) and reported Layer-2 adoption of 10-40 per 300 trees. On real Pareto
output the second layer is adopted **2.8 %** of the time. This is the same class of
error docs/66 recorded for the wall-clock cap: a sample smaller than what the display
layer is actually handed. Layer 1 does nearly all of the visible work.

Consequences, in order of confidence:

* Layer 2 stays (it is written, tested, costs ~2 ms, and the cases it does hit —
  `x*y + x*z -> x*(y+z)` — are the visually striking ones).
* **No further investment in the rule set or the budget.** A 2.8 %-adoption layer does
  not repay 56 -> 70 rules.
* The `* 0` residue that the NaN-preservation policy leaves behind (`x - x -> x * 0`)
  is a known cosmetic wart. At 0.2 % of real output it is not worth changing the
  policy for.

## 6. Does a larger budget help — e.g. for the final equation only?

The natural follow-up: `display_simplify()` runs once per Pareto member, so the
recommended equation alone could be given a budget orders of magnitude larger. Measured,
this does not work.

**The default budget does bind.** 549 of 1750 expressions (31 %) stop at the caps
without reaching a rewrite fixpoint:

| input nodes | 1-10 | 11-20 | 21-30 | 31-40 | 41+ |
|---|---|---|---|---|---|
| hit the cap | 5 % | 28 % | 53 % | 50 % | 62 % |

**But raising it buys almost nothing.** Re-running those 549 with a 10x budget
(`max_iterations` 10 -> 20, `max_enodes` 2000 -> 20000):

| input nodes | capped | strictly smaller at 10x | mean nodes saved |
|---|---|---|---|
| 1-10 | 20 | **0 (0 %)** | — |
| 11-20 | 205 | **0 (0 %)** | — |
| 21-30 | 290 | 25 (8.6 %) | 1.6 |
| 31+ | 34 | 3 (8.8 %) | 2.7 |

Totals: 28/549 (5.1 %) improved, total node count 11320 -> 11271 (**0.43 %**),
Layer-2 adoptions 34 -> 59, per-expression cost 3.5 ms -> 33 ms. **400 of the 549
(73 %) saturate at 10x**, so for those, no budget whatsoever can change the result —
saturation is a fixpoint, not a timeout.

Hitting the cap therefore means "rules could still fire", not "a shorter equivalent
exists". The e-graph has already extracted the minimum-node form early; the remaining
iterations generate same-size alternatives.

At the sizes a reader actually reads, the effect is **exactly zero** (0/225 at <= 20
nodes). It appears only from 21 nodes up, at roughly one expression in eleven, for
about 2 nodes.

There is also a cost to spending the extra budget. More rewriting means more class-B
redistribution, which moves the *printed constants* away from the fitted ones:

```
default:  ((x0^2) / 1.59819) + 19.2709
10x    :  ((x0^2) / 5.34294) + 5.76435      # same value, different-looking coefficients
```

In the one equation a user is most likely to quote, that is a regression, not an
improvement.

### The "relax the rules for the final equation only" variant

Restricting *rule relaxation* (§3) to the single reported equation does not change the
verdict either:

* Cost was never the constraint — all ~25 Pareto members together take ~50 ms. "Only
  one equation" saves an expense that was never being paid.
* The actual constraint is the semantic contract (no rewrite changes finiteness), and
  that is independent of how many expressions it is applied to.
* The final equation is the one pasted into a paper and checked against `predict()`.
  "Strict everywhere on the front, loose for the recommended member" is not a rule
  anyone can explain, and it degrades fidelity precisely where fidelity matters most.

## 7. Verdict

**NO-GO on all three variants** — adding sympy-equivalent rules, raising the e-graph
budget, and doing either "for the final equation only". The display simplifier ships
as it is.

If it is ever revisited, the whole measured opportunity is **two Pow rules**
(`(a^p)^q`, `(a*c)^p`), which are worth about 1.4 % of expressions and must be
opt-in/default-OFF with a docs record, per CLAUDE.md's second layer, because
`safe_pow`'s semantics (docs/69) make them value-changing.

None of this touches PySR default parity: the display layer runs at finalisation on a
copy of the reported tree and the search never reads it (docs/48 D2, docs/54).

## 8. Reproduction

Not committed — this screen used a scratch harness, described here precisely enough to
rebuild:

1. **Rendering CLI** (~130 lines): a recursive-descent parser for the `to_string()`
   infix grammar (`x<N>` variables, the `op_names.hpp` unary names, `+ - * / ^`,
   literal negation) producing a postfix `Tree`, then
   `display_simplify(tree, &stats, limits)` with `max_iterations`/`max_enodes` taken
   from `argv`. Prints the rendered string plus `in->out` node counts and the
   `DisplaySimplifyStats` fields. Build: `g++ -O2 -std=c++17 -Ir-package/rsymbolic2/src
   tool.cpp display_simplify.cpp egraph.cpp platform_libm.cpp`.
2. **Corpus**: distinct non-trivial `expression` values from
   `benchmarks/results/*front*.csv`.
3. **PySR replication**: `pysr2sympy` copied verbatim from `pysr/export_sympy.py`
   (importing `pysr` pulls in Julia and is unnecessary), i.e. the `sympy_mappings` dict
   plus the `evaluate=False` / `TypeError` fallback shown in §2.
4. **Fair comparison**: feed `str(sympy.simplify(...))` (with `**` -> `^`, `Abs(` ->
   `abs(`) back through the CLI and compare its *output* node count with the CLI's own
   output node count for the original string.

`standalone/benchmarks/bench_simplify.cpp` remains the harness for the cost and
reproducibility questions (docs/66); it generates random trees and does not read
expressions, which is why it was not used here.
