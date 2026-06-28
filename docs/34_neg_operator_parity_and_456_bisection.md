# 34. The `neg` operator-set parity gap, and exonerating mutations ④⑤⑥

**Date:** 2026-06-27

## Summary

The Feynman gate's operator set was **not identical to the PySR comparison's**, violating
CLAUDE.md ("operators are the shared problem input given identically to both tools"). Closing
that gap — removing `neg` from the gate — is the correct parity fix and lowers the
authoritative recovery count from a (non-parity) **19/25 to 18/25**, which equals PySR's
baseline. A controlled experiment separately **exonerates** the recent structural-search
commit `0ac5f43` (mutations ④ init-pop / ⑤ op-arity split / ⑥ two-child crossover): it did
**not** cause the `interference`/`boltzmann_dist` failures that prompted this investigation.

## The parity gap

| | unary | binary | op-arity split `nbin/(nuna+nbin)` |
|---|---|---|---|
| gate (`02_feynman_gate.R`, before) | neg,exp,log,sin,cos,sqrt,tanh,abs,square = **9** | 5 | 5/14 ≈ 0.357 |
| PySR comparison (`05_feynman_pysr_comparison.jl`) | exp,log,sin,cos,sqrt,tanh,abs,square = **8** | 5 | 5/13 ≈ 0.385 |

The sole divergence was **`neg`**. PySR omits it deliberately (negation is `0 - x`; both tools
have `sub`, so no expressiveness is lost — `05_…jl:22`). Including it (a) gave rsymbolic2 a
9th operator PySR never sees, and (b) skewed the SR.jl op-arity split (`MutationFunctions.jl`,
`docs/29` A#13c) from 5/13 to 5/14. **Fix:** remove `neg` from the gate's `unary_ops`,
equalizing on rsymbolic2's side (CLAUDE.md). After this the operator set and the arity split
are identical to PySR/SR.jl.

## Why this came up

`0ac5f43` (`docs`/memory: init-pop ④, op-arity ⑤, two-child crossover ⑥) appeared to regress
`interference` and `boltzmann_dist` from passing to 0/5. Inspecting the found expressions
showed the failures were **structural** (bloated trees using `neg`/`tanh`/`exp`/`pow`, not the
clean `add`/`sqrt`/`mul`/`cos` backbone), i.e. a search-trajectory effect, not a
constant-optimiser miss — which pointed at either ④⑤⑥ or the operator set.

## Controlled experiment: ④⑤⑥ are exonerated

With the parity-correct (neg-free) set, `interference` was measured at HEAD vs the parent
commit (the only difference being ④⑤⑥), reverted via
`git checkout 40ad3f2 -- r-package/rsymbolic2/src/`:

| build | interference (runs=5) | best seed |
|---|---|---|
| HEAD `0ac5f43` (④⑤⑥ present) | 0/5 (med 1.2e-3) | 2.5e-4 |
| parent `40ad3f2` (④⑤⑥ absent) | **0/5** (med 2.7e-3) | 1.4e-4 |

Both fail 0/5. **④⑤⑥ is not the cause.** The earlier "regression" was an artifact of measuring
under the non-parity neg-included config, where a single borderline seed tipped over the
threshold. All runs finished in <300 s having exhausted `generations=2800`, so the residual
gap is not wall-clock throughput but **per-generation search quality** from category-A
implementation differences (self-LM vs BFGS, Float64 vs Float32, RNG stream — all
CLAUDE.md-permitted, `docs/29` §A, §E). This matches `docs/26`'s pre-existing finding that
`interference` is an rsymbolic2 far-miss that SR.jl recovers with more effective search.

## Authoritative neg-free gate (runs=5, 2026-06-27): 18/25 PASS

The only PASS/FAIL flip from the old 19/25 (neg-included, 2026-06-26) is **`center_mass`
(3/5 → 2/5)** — the redundant `neg` operator had been propping up one borderline problem.
Every other movement is within-bucket seed wiggle (lorentz_x 3→4, lens_eq 5→4, clausius_moss
5→4, boltzmann 0→1, harmonic_ke 0→1, newtons_grav 0→1, bose_einstein 1→0). `center_mass`
thus joins `interference`/`boltzmann_dist` as a parity-correct borderline far-miss.

18/25 equals PySR's documented baseline; the previous 19/25 was inflated by the non-parity
operator. Per CLAUDE.md, the parity-correct configuration is kept even though it scores one
lower — parity outranks a borderline recovery obtained from a non-parity operator.

## Tooling added
- `target=key1,key2` filter in `02_feynman_gate.R` for isolated, bounded single-problem
  diagnosis; a subset run writes to a separate `<label>_diag` CSV so it never overwrites the
  authoritative dated full-gate result.
- `.gitignore`: `build-*/` glob (covers scratch trees `build-parity/`, `build-prof/`,
  `build-soa/`).

See also `docs/26` (Feynman PySR comparison, far-misses), `docs/29` A#13/§E (mutation
faithfulness, the throughput implication).
