# 32. Audit of external "why PySR is fast" claims (ChatGPT + Gemini)

**Date:** 2026-06-25
**Motivation:** The user asked two LLMs (ChatGPT, Gemini) to enumerate the reasons
PySR is fast at its default settings, then asked whether each claim is *true*, whether
it is *already implemented* in rsymbolic2, and — for anything true-and-unimplemented —
for an implementation plan. This file records the audit so the conclusions are not
re-derived from scratch next time.

## How each claim is classified

Per CLAUDE.md ("PySR Default Parity") every claim falls into exactly one bucket, and
the bucket — not the claim's plausibility — decides whether we act:

1. **Setting / search behaviour, default-ON** → parity-required; must match PySR. A gap
   here is a bug to fix.
2. **Setting, default-OFF** (turbo, batching, fast_cycle, Float32-as-speed, etc.) →
   parity-required to stay OFF. Implementing it would *violate* parity.
3. **Problem input** (operators, nested constraints, per-operator complexity) → given
   identically to both tools per problem; not a default to copy.
4. **Implementation / architecture** (tree layout, memory pooling, parallelism
   mechanism, optimiser algorithm, numeric precision) → "implementation method may
   differ"; rsymbolic2 uses its own design. PySR is a *reference, not a spec* here.

**Headline result: across all ~13 claimed factors, none is "true + default-ON + parity-
required + unimplemented."** Every parity-required default-ON factor is already
implemented; everything else is correctly OFF, a problem input, or our own
implementation choice.

---

## Part A — ChatGPT's 7 factors

| # | Claim | True of PySR? | Bucket | rsymbolic2 status |
|---|---|---|---|---|
| 1 | Island model (multi-population) | yes (`populations=31`) | setting, ON | ✅ `n_populations=31` |
| 2 | Migration | yes (ring + HOF) | setting, ON | ✅ implemented |
| 3 | Hall of Fame | yes | setting, ON | ✅ `hall_of_fame.hpp` |
| 4 | Pareto-front management | yes | setting, ON | ✅ `pareto_front` / `select_best` |
| 5 | Simplification | yes (`should_simplify`) | setting, ON | ✅ `simplify_expressions=true` |
| 6 | Structure / constant-opt separation | partly (mischaracterised) | impl + setting | ✅ cadence matched (docs/31) |
| 7 | Float32 (`precision=32`) | yes | **implementation** | ❌ Float64 (intentional) |

### Accuracy corrections to ChatGPT's *mechanism* explanations
- **#2 Migration (rated ★★★, overstated):** at PySR defaults ring migration is
  effectively **inert** — `round(fraction_replaced=0.00036 × pop=27) = 0`. Cross-island
  flow is carried by HOF migration (`fraction_replaced_hof=0.0614`). rsymbolic2
  reproduces exactly this (ring inert by default, HOF carries flow;
  `evolutionary_search.hpp`).
- **#4 Pareto front — mechanism is wrong:** the front (= HOF archive) does **not** shrink
  the evolving populations; each island always holds `population_size=27` members. The
  front is for *reporting* and HOF migration, not for pruning the working population, so
  "search space shrinks → faster" does not hold.
- **#6 ranked #1 — the least accurate:** PySR does **not** run a clean two-phase
  "find all structure, then fit constants." It interleaves: constants are part of the
  genome (`mutate_constant`, weight 0.0346) **and** are LM/BFGS-optimised once per
  iteration with `optimize_probability=0.14`. The "N_struct + N_const instead of
  N_struct × N_const" framing is an idealisation, not the SR.jl mechanism. The real,
  implemented principle is the *cadence* (optimise the population once per iteration),
  matched in docs/31 — this, not a phase split, was the throughput fix (13→18/25).
- **#3 HOF:** conflates HOF-as-output (Pareto archive) with HOF-as-search-cache; its
  search-acceleration role is only via HOF migration.
- **#7 Float32:** ChatGPT honestly rates it ★★ (lowest) — consistent with our position.

---

## Part B — Gemini's factors

| Claim | True of PySR? | Bucket | rsymbolic2 status |
|---|---|---|---|
| Adaptive parsimony | yes, default ON | **setting, ON** | ✅ `adaptive_parsimony_scaling=1040` |
| Nested constraints | feature exists, default `None` | problem input / OFF | not implemented (correct) |
| Per-operator complexity (`complexity_of_operators`) | feature exists, default `None` | problem input / OFF | not implemented (correct) |
| Lightweight binary tree | yes | implementation | own `node.hpp`/`tree.hpp` |
| Memory / deep-copy optimisation | yes | implementation | own design (see refutation below) |
| turbo (LoopVectorization) | feature exists, default OFF | setting OFF / Julia-only | not implemented (correct) |
| batching / fast_cycle | feature exists, default OFF | setting OFF | not implemented (correct) |
| Async multithreading | yes | implementation (parallelism) | OpenMP islands (allowed divergence) |
| Independent HOF constant-opt | **inaccurate** | implementation | re-designed away (docs/30/31) |

### Notes
- **Adaptive parsimony is the only parity-required default-ON factor here, and it is
  implemented.** Gemini's arXiv:2507.05858 quote is accurate and matches SR.jl's
  `use_frequency_in_tournament` mechanism (`member.cost * exp(1040 * normalized_freq[size])`,
  docs/28 §B1).
- **Lightweight-tree / memory claims do not apply to rsymbolic2's measured bottleneck.**
  docs/30 (per-phase profiling) shows **constant fitting (LM) is ≈ 93 % of compute** and
  the entire non-fit path (tree alloc + simplify + mutate/crossover + tournament + HOF)
  is **≈ 0.4 %**. Optimising tree layout/memory cannot move the needle here; the SelfLM
  choice already eliminated per-fit heap churn (docs/25).
- **"Independent HOF constant-opt" mischaracterises SR.jl:** SR.jl optimises the
  *population* once per iteration (`optimize_and_simplify_population`), not a separate
  HOF-only phase. rsymbolic2 previously had a per-child + HOF re-optimisation design,
  which diverged from PySR and dominated compute; removing it was the throughput fix
  (docs/30/31).
- Gemini's arXiv:2305.01582 `complexity_of_operators` quote is accurate, but that option
  is default-disabled — not a default speed factor.

---

## Part C — Float32 and BFGS (explicit user request to "implement like PySR")

The user asked to make Float32 (`precision=32`) and BFGS (`optimizer_algorithm="BFGS"`)
match PySR, *or* state strong reasons against. **Recommendation: do not implement
either as a core change.** Reasons:

1. **CLAUDE.md already designates both as allowed divergences.** The PySR Default Parity
   rule governs *settings and search behaviour*; it explicitly excludes numeric
   precision (Float64 vs Float32) and the optimiser algorithm (self-LM vs BFGS) as
   "implementation method may differ" (docs/28 §A marks both `class=implementation`). So
   "match PySR" does not require these.
2. **Correctness (priority #1).** Float32 lowers the precision of constant optimisation
   and loss evaluation. The primary benchmark (Feynman) is exact ground-truth recovery;
   reducing precision risks recovery-threshold regressions for, by ChatGPT's own ★★
   rating, the smallest speed gain. This is exactly the correctness-for-speed trade the
   priorities forbid without measured evidence.
3. **Portability (priority #3).** docs/06 gives primary-source Windows/Rtools(MinGW/UCRT)
   evidence behind the self-LM choice. LM is the correct algorithm for least-squares
   (dense NNLS), header-only via Eigen, zero per-fit heap allocation (the throughput fix,
   docs/25/30). A PySR-faithful BFGS adds Windows maintenance cost for **no parity
   benefit** (it is an allowed divergence) and converges more slowly on least-squares.
4. **What actually mattered is already matched.** docs/31: the speed-relevant thing in
   SR.jl was the constant-optimisation *cadence*, not the inner solver *kind* (BFGS).
   The cadence is matched.
5. **Performance (priority #5) requires measured evidence.** No benchmark shows Float32
   or BFGS would speed up rsymbolic2; implementing them now is speculative.

**Future, evidence-gated only:** if forward-pass evaluation throughput is ever *measured*
to be rate-limiting (it is not today — fitting is 93 %), a templated Float32 *evaluation*
path (keeping Float64 for constant optimisation) could be revisited. Full-core Float32 is
not on the table.

## CLAUDE.md

**No change needed.** CLAUDE.md already lists Float32 and BFGS as allowed divergences
and matches this audit's conclusion exactly. Forcing them to match PySR would contradict
the parity rule, not satisfy it.
