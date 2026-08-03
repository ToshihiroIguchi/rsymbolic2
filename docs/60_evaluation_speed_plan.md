# 60. Evaluation-path speed plan (bit-identical levers only)

**Date:** 2026-07-26
**Status:** Phase 0 COMPLETE (2026-07-26, §7). **Phase 1 closed NO-GO** (§7.3).
**Phase 2 closed NO-GO** (§7.4). **Phase 3 gate PASSED but DEFERRED by decision**
(§7.6). **Phase 4 closed NO-GO** (§7.7). No shipped behaviour changed by any of it
(§7.5). §1's premise corrected by §7.1. **Plan complete.**
**Goal:** reduce wall-clock time of the default search **without changing the result**.
Recovery rate is explicitly *not* a goal of this plan (`docs/35`, `docs/44`, `docs/47`
already established that the recovery levers are exhausted); this is a pure
Performance (#5) exercise, undertaken only because every lever below is provably
result-preserving and therefore costs nothing from Priorities #1-#4.

## 0. Scope and the constraint that shapes it

Every lever in this plan must be **bit-identical with the mechanism on or off**, per
CLAUDE.md's allowed-divergences clause for behaviour-neutral accounting and caching.
This is deliberate and was re-confirmed on 2026-07-26: the clause is not a brake on
speed work but the *permission slip* that lets `eval_cache` exist as an implementation
detail rather than a search setting. Removing it would unlock essentially nothing —
every result-changing candidate is independently blocked (see §6) — while destroying
the measuring instrument that ON/OFF screens depend on (`docs/49` criterion 3 is
literally "all non-timeout expressions identical between arms").

Consequence: this plan contains **no** change to operator semantics, evaluation order,
summation order, or numeric type. Only (a) build flags that preserve IEEE semantics,
(b) memoisation behind an exact-equality guard, (c) elimination of *redundant* work
whose result is provably the same double, and (d) tuning of a tile width across which
points are independent.

## 1. Why the evaluation path is the only target

`docs/31` §97-100 recorded the phase mix after the constant-optimisation cadence fix:

| phase | docs/30 (before) | docs/31 (after) |
|---|---:|---:|
| constant fitting (LM) | 93 % | **10.7 %** (1 415 calls / 20 s) |
| forward-pass child eval (`evolve_sse`) | minority | **87.5 %** (321 515 calls / 20 s) |

So the target is `sse_current` -> `evaluate_soa_residual` and nothing else. Levers
aimed at the LM path (the per-`fit()` `Model`/SoA-pool allocation in
`least_squares_problem.hpp:154`) now address ~10 % of compute and are **out of scope**
for this plan.

> **CORRECTED by Phase 0 (§7.1).** The `docs/31` split above no longer holds: measured
> 2026-07-26, the mix is `evolve_sse` **~76 %** / `popopt_fit` **~21 %**. The forward
> path is still the dominant target, but the LM path is roughly twice the share this
> section assumed. See §7.1 for the numbers and the likely cause.

**Caveat that Phase 0 exists to remove:** those numbers are from 2026-06. Since then
`display_simplify`/e-graph (`docs/52`, `docs/54`), units (`docs/46`), `linear_scaling`
(`docs/50`), `strong_simplify` (`docs/55`), batching (`docs/28` B5), `warmup_maxsize_by`
(`docs/42`) and macro operators (`docs/57`) have landed. The mix must be re-measured
before anything is built on it.

## 2. Pre-registered bars (fixed now, before any measurement)

Per CLAUDE.md Benchmarking Requirements. These are not to be weakened after seeing
results.

1. **Bit-identity (hard gate, no exceptions).** Every non-timeout Feynman gate
   expression must be *identical* between arms, as in `docs/49` criterion 3 (25/25).
   Any single difference means the lever is not an implementation detail; it is
   rejected outright rather than re-classified.
2. **Adoption threshold.** Median per-problem wall change <= **-5 %**.
3. **No regression.** No problem more than **+2 %** slower. This is the bar that
   `eval_cache` failed in `docs/49` (driven_osc +6.8 %, heat_conduct +2.9 %,
   torque +2.8 %); Phase 2 is an attempt to *meet* it, never to lower it.
4. **Statistics.** Medians over >= 5 runs with spread reported; record version,
   hardware, thread count, and the exact build flags of both arms.
5. **Platforms.** Windows is the iteration loop. Ubuntu (WSL) verification is required
   at the milestone before any default is flipped or anything is committed
   (CLAUDE.md verification cadence).
6. **Run-time discipline.** Every diagnostic and benchmark invocation carries an
   explicit timeout so no step can hang unbounded.

A lever that passes 1 but fails 2 stays available as a documented opt-in (the
`eval_cache` precedent); a lever that fails 1 is deleted.

## 3. Phase 0 — Measure first (gates everything else)

Nothing in Phases 1-4 is built until this produces numbers.

**0a. Current phase mix.** Rebuild the compile-guarded profiler and re-run it:

```
cmake -S standalone -B build-prof -G Ninja \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-rtools.cmake \
      -DCMAKE_BUILD_TYPE=Release -DRSYMBOLIC2_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS=-DRSYMBOLIC2_PROFILE
cmake --build build-prof --target bench_profile
OMP_NUM_THREADS=4 ./build-prof/standalone/bench_profile.exe rel_mass 60
```

`bench_profile` generates its Feynman data analytically, so it is self-contained and
cheap. Run both `spring_pe` and `rel_mass` as in `docs/30`.

**0b. Parallel health.** Record `cpu/wall` at the 4-thread comparison cap *and* at full
cores. `docs/30` measured 3.9 at 4 threads (healthy). If cpu/wall is markedly
sub-linear at full cores, allocator contention is real and the two allocation levers
listed as out-of-scope in §6 come back into scope; if it is near-linear, they stay out.

**0c. Build-flag baseline.** Same binary set at `-O2` and `-O3` (see Phase 1).

**Decision rule:** a lever proceeds only if its target phase is >= 5 % of measured
compute. Record the mix in `docs/60` §Results regardless of outcome.

## 4. The levers, in cost-ascending order

### Phase 1 — Build flags (`-O2` vs `-O3`); zero source change — **CLOSED, NO-GO (§7.3)**

**Observation.** `r-package/rsymbolic2/src/Makevars{,.win}` set no optimisation flag,
so the R package compiles at R's default (`-O2` on Rtools/MinGW). The Python extension
is built by scikit-build-core at `Release`, i.e. `-O3`. **The same core is therefore
shipped at two different optimisation levels**, which is at minimum a finding worth
documenting and possibly a free speedup for R.

**Hypothesis.** `soa_eval.hpp:20-28` states that the SoA point-batched layout exists
specifically so the compiler can auto-vectorise the tile loops. But the tile length is
`P = min(kStride, m - lo)` — a *runtime* trip count — and GCC's `-O2` enables the
vectoriser only under the `very-cheap` cost model, which generally declines loops
requiring runtime peeling. If so, the design's intended speedup is being left on the
table in the R build.

**Bit-identity argument.** Vectorising `+ - * /` does not reassociate (no
`-ffast-math`), so per-point results are unchanged. Points are independent and the
summation in `sse_current` stays scalar and in order.

**Steps.**
1. Measure the standalone harness at `-O2` vs `-O3`, same seed.
2. Verify bit-identity explicitly (bar 1) — `test_soa_eval` already asserts the SoA
   path against the scalar reference; add a gate-subset expression diff on top.
3. If the win is real, decide *how to ship it to R*. Preferred candidate is **not**
   `-O3` (CRAN discourages overriding optimisation level in `Makevars`) but the
   narrower, additive
   `PKG_CXXFLAGS += -ftree-vectorize -fvect-cost-model=dynamic`, which enables the
   vectoriser without touching `-O`. Portability caveat: those spellings are GCC/Clang
   specific and `Makevars` cannot easily branch on compiler; the fallback is a
   `#ifdef __GNUC__`-guarded `#pragma GCC optimize("tree-vectorize")` scoped to
   `soa_eval.hpp` alone. Both options are ugly enough that they are justified only by a
   measured win — hence step 1 first.

**Cost:** hours. **Risk:** none to correctness (gated by bar 1).

### Phase 2 — `eval_cache`: capacity sweep, then reconsider the default

**Standing evidence (`docs/49`).** Bit-identical, 25/25 expressions identical,
overall wall **-15.8 %**, median per-problem **-9.71 %**, hit rate **0.19-0.34**.
It ships OFF solely because three problems exceeded the +2 % regression bar.

**Hypothesis.** `kEvalCacheSlots = 1024` (`evolutionary_search.cpp:56`), direct-mapped,
against a search that visits hundreds of thousands of distinct trees, is
capacity-limited. If so, 0.19-0.34 is not the mechanism's ceiling but the table's, and
the three regressions are cache *overhead not repaid* at a low hit rate — exactly the
symptom a larger table would remove.

**Steps.**
1. Sweep `kEvalCacheSlots` over {1024, 4096, 16384}, recording hit rate and wall per
   problem. One compile-time constant; no design change.
2. Measure the per-island memory cost at each size. The entry retains a `Tree` copy for
   the exact-equality guard, so memory is not `slots * 16 bytes`. **Hard constraint:**
   the web GUI runs on a fixed 128 MB WASM heap that cannot grow (`docs/51`), and the
   table is per-island with 31 islands by default — the sweep must report total
   footprint, and any adopted size must fit that budget or be scaled by
   `n_populations`.
3. If a size meets all three bars, re-run the full `docs/49` protocol at that size and
   propose flipping the default. Flipping is legitimate precisely *because* the
   mechanism is bit-identical: it is an implementation choice, not a search setting, so
   PySR default parity is untouched.
4. If no size meets bar 3, keep the default OFF and document the improved opt-in
   numbers.

**Cost:** one constant + benchmark time. **Risk:** memory footprint (step 2 gates it).
**This is the highest-expected-value item in the plan** — the mechanism is already
written, already measured, and already known to be worth double digits.

### Phase 3 — Constant-subtree scalarisation in the SoA evaluator

**Observation.** `evaluate_soa_residual` (`soa_eval.hpp:91-97`) broadcasts every
`Constant` node's value across all `P` lanes, and every downstream operator then runs
`P` times over identical values. A data-independent subtree such as `c1 * c2` is
recomputed `P` times per call for a result that is one double.

**Why this is not already solved by `simplify()`.** `simplify()` runs only in the
once-per-epoch `optimize_and_simplify_population` pass. Children produced by mutation
and crossover are scored by `score_sse` *immediately*, carrying unfolded constant
subexpressions for the remainder of the epoch. The 87.5 % path is precisely the path
that sees un-simplified trees.

**Design.** One tree walk marks each node data-independent (no `Variable` in its
subtree). Data-independent nodes evaluate once, scalar; the broadcast to `P` lanes
happens only at the boundary where such a subtree becomes an operand of a
data-dependent node (or never, if the whole tree is constant).

**Bit-identity argument.** Identical operators applied to identical operands produce
identical doubles; broadcast is a copy. The safe-guard variants (`safe_pow`,
`sqrt(x>0?x:0)`, log domain handling) are the same code on the scalar path, so domain
behaviour is unchanged. This must be *asserted*, not argued: extend
`test_soa_eval.cpp` with random trees containing constant subtrees, compared strictly
against the production scalar `evaluate<double>`.

**Steps.**
1. **Measure the ceiling before implementing.** Instrument a real run to report what
   fraction of node x P operations are data-independent. This follows the discipline
   `docs/43`/`docs/44` established (measure the oracle ceiling first, build second).
2. Proceed only if that fraction is >= 10 %. Otherwise stop and record the negative
   result.
3. Implement, extend `test_soa_eval`, measure against the bars.

**Cost:** the only substantive source change in the plan; gated on step 1.
**Risk:** moderate — it touches the hottest and most correctness-critical kernel. The
strict equality test is the mitigation.

### Phase 4 — Decouple and sweep the tile width

**Observation.** `kStride = 256` (`least_squares_problem.hpp:60-65`) is documented as
chosen for **stop-predicate poll cadence** ("one `std::function` call per ~256
evaluations"), and is then reused unchanged as the SoA **tile width**. There is no
record of the tile width being chosen on cache grounds.

**Steps.** Separate the two roles (poll cadence stays 256), sweep tile width over
{32, 64, 128, 256}, measure. Bit-identical by the same independence argument already
asserted in `test_soa_eval`.

**Cost:** the cheapest experiment here. **Expected value:** genuinely unknown, possibly
zero — the live working set is `stack_depth * P * 8` bytes, which already fits L1 at
`P = 256` for shallow trees, so the cache argument is weaker than it first appears.
Run it rather than argue about it; it is an afternoon.

## 5. Ordering and rationale

Cost-ascending crossed with evidence-descending:

| # | Lever | Cost | Standing evidence | Gate |
|---|---|---|---|---|
| 0 | Re-measure phase mix, cpu/wall, `-O2`/`-O3` | low | `docs/30`, `docs/31` (stale) | — |
| 1 | Build flags | hours | none (new hypothesis) | Phase 0c |
| 2 | `eval_cache` capacity | low | **strong** (`docs/49`) | memory footprint |
| 3 | Constant-subtree scalarisation | medium | none | measured >= 10 % ceiling |
| 4 | Tile-width sweep | lowest | none | — |

Phase 0 gates all. Phases 1, 2 and 4 are constant/flag-level changes. Phase 3 is the
only one that touches the kernel, and it is gated on a measured opportunity.

**Methodological note:** because R ships `-O2` and Python `-O3` (Phase 1), timings
taken through different bindings are not comparable. **All speed measurement in this
plan uses the standalone harness**, with flags recorded explicitly.

## 6. Explicitly out of scope, and why

| Candidate | Why not |
|---|---|
| `initial_constants()` per-call allocation in `sse_current` (`evolutionary_search.cpp:321`) | The last per-call heap allocation on the forward path, but small against the cost of evaluating m points. Was conditionally re-openable on Phase 0b; **Phase 0b came back healthy (cpu/wall 3.61-3.78 at the 4-thread cap, at the 97 % island-granularity ceiling), so there is no allocator-contention evidence and this stays closed** (§7.2) |
| Per-`fit()` `Model`/SoA-pool allocation (`least_squares_problem.hpp:154`) | Was dismissed on the `docs/31` figure of ~10 %. **Phase 0 measured the LM path at ~21 %** (§7.1), so the dismissal no longer holds on share alone. Still not scheduled: the allocation amortises over a whole multi-start fit (~5.2 ms/call measured), so its share of that 21 % is small — but it is now a legitimate candidate if Phases 2-4 disappoint |
| Incremental parent->child subtree evaluation | Highest theoretical ceiling of anything considered, but requires propagating the mutation site through `mutate`/`crossover` into the evaluator. Priority #2/#4 cost is real. Revisit only if Phases 1-4 disappoint *and* Phase 0 confirms evaluation still dominates |
| SLEEF / vectorised transcendentals | Rejected twice (`docs/30`, re-confirmed `docs/37`) on three independent grounds: PySR's `turbo=False` means parity says don't; Rtools/MinGW dependency cost with a mandatory serial fallback; not bit-identical. The first two survive any change to the bit-identity clause |
| Float32 | Measured (`docs/36`): **zero** speedup on the Rtools/MinGW scalar evaluator, ~100-1000x worse loss floor |
| BFGS | Measured (`docs/36`): 1.3-2.1x slower *and* worse quality |
| `-ffast-math` | Unsafe here independently of determinism: the search uses IEEE NaN/Inf as control flow (`sse_current` returns `kInf` on non-finite; `clamp_finite` protects the normal equations; the HOF and `evolve_island` reject non-finite losses). `-ffinite-math-only` folds `std::isfinite` to true and disables all of it |
| Raising `n_threads` further | Islands are the only unit of parallelism, capped at `n_populations` = 31 (`resolve_team_size`); measured in `docs/37` |
| Removing the bit-identity clause from CLAUDE.md | Considered and rejected 2026-07-26 (§0) |
| GPU offload (CUDA / OpenCL / SYCL / Vulkan) | Screened **NO-GO** in `docs/78` (2026-08-02) against §7.1/§7.2/§7.6 and `docs/36`: the child loss is consumed synchronously by the next evolution step so the round trip cannot be hidden, one dispatch is only ~15 k node x point ops, and the hot path is Float64 transcendentals — the axis consumer GPUs are weakest on, with Float32 already closed by `docs/36`. Vendor-neutral routes also fail Platform Constraints (Windows `nvcc` needs MSVC; R uses Rtools/MinGW) |

## 7. Measurement results (2026-07-26)

**Setup.** Rtools45 GCC 14.3.0, Windows 11, 6c/12t box. Three out-of-source trees from
identical sources, flags verified in `build.ninja`: `build-prof`
(`-DRSYMBOLIC2_PROFILE -O3 -DNDEBUG`), `build-o3` (`-O3 -DNDEBUG`), `build-o2`
(`-O2 -DNDEBUG`). Driver `bench_profile` at the faithful gate config (pop=27,
islands=31, gens=2800, tournament=15, maxsize=30, `optimize_probability`=0.14,
scaling=1040, n=1000). Budget 300 s, which every run completed, so each run is **fixed
work** and wall times are comparable within a problem/seed.

Note: `build-prof` runs carry instrumentation overhead (rel_mass seed 1: 62.0 s
profiled vs 54.5 s unprofiled, ~14 %). Never compare a profiled wall against an
unprofiled one.

### 7.1 Current phase mix — supersedes `docs/31` §97-100

Summed work-seconds over 31 islands, 4 threads, full 2800-generation runs:

| phase | rel_mass | spring_pe |
|---|---:|---:|
| `evolve_sse` | **75.8 %** (2 403 971 calls, 69.5 us) | **76.4 %** (240 369 calls, 123.1 us) |
| `popopt_fit` | **20.6 %** (8 655 calls, 5249 us) | **21.2 %** (1 019 calls, 8067 us) |
| `mutate_xover` | 1.9 % | 1.2 % |
| `tournament` | 1.4 % | 0.9 % |
| `simplify` | 0.2 % | 0.1 % |
| `hof_update` | 0.1 % | 0.0 % |
| `init_sse` / `migration` | 0.0 % / 0.0 % | 0.1 % / 0.0 % |

**The `docs/31` figure of 87.5 % / 10.7 % is stale: the LM path has roughly doubled its
share, to ~21 %.** The call *ratio* is essentially unchanged
(`evolve_sse`/`popopt_fit` = 278 here vs 227 in `docs/31`), so the shift is in
**per-fit cost**, not cadence.

**Likely cause (inference, not established):** `docs/33` added SR.jl-parity
constant-optimiser multi-start (`optimizer_nrestarts=2`, i.e. start 0 plus two
perturbed restarts) on **2026-06-25**, the day after `docs/31` (2026-06-24) measured
the mix. Up to 3x the LM work per fit matches the observed ~2x share shift. Note this
is a *setting* required by PySR parity, so it is not a lever — only its implementation
is.

Within `fit()` (rel_mass): **residual 42.9 % (260 677 calls, 72.9 us) / Jacobian
57.1 % (154 800 calls, 163.7 us)**. This supersedes `docs/30`'s 23.8 % / 76.2 %: the
SoA vector-mode AD removed the Jacobian's 13x per-call penalty, so Jacobian-specific
optimisation is much less attractive than it was.

### 7.2 Parallel health — healthy at the comparison cap

| problem | threads | wall | cpu/wall | efficiency | granularity ceiling |
|---|---:|---:|---:|---:|---:|
| rel_mass | 4 | 62.01 s | 3.61 | 90 % | 31/32 = 97 % |
| rel_mass | 12 | 46.52 s | 7.51 | 63 % | 31/36 = 86 % |
| spring_pe | 4 | 10.39 s | 3.78 | 95 % | 97 % |
| spring_pe | 12 | 6.66 s | 9.09 | 76 % | 86 % |

At the 4-thread benchmark cap the search runs at 90-95 % efficiency, essentially at the
ceiling imposed by distributing 31 islands over 4 workers. **No allocator-contention
evidence**, so both allocation levers stay out of scope (§6).

At 12 threads a gap opens that island granularity does not fully explain (rel_mass: 63 %
observed vs an 86 % ceiling — ~23 points unaccounted). This is consistent in direction
with `docs/37`'s finding that logical-core scaling is sub-ideal but still wall-clock
optimal. **Not a lever in this plan; recorded as a separate cheap follow-up.**

### 7.3 `-O2` vs `-O3` — NO-GO

Paired by seed, 4 threads, unprofiled builds:

| problem | seed | `-O2` | `-O3` | delta |
|---|---:|---:|---:|---:|
| rel_mass | 1 | 55.80 s | 54.49 s | -2.35 % |
| rel_mass | 2 | 56.96 s | 56.84 s | -0.21 % |
| rel_mass | 3 | 130.05 s | 134.28 s | **+3.25 %** |
| spring_pe | 1 | 11.54 s | 10.66 s | -7.62 % |
| spring_pe | 2 | 3.22 s | 3.28 s | **+1.86 %** |

**Median -0.21 %**, sign flips across seeds, spread (-7.6 % .. +3.25 %) far exceeds the
effect. Against the -5 % adoption bar this is **NO-GO**, and the result is
noise-dominated rather than a small real win.

The Phase 1 hypothesis — that `-O2`'s `very-cheap` vectoriser cost model declines the
SoA tile loops (runtime trip count) and leaves the design's intended speedup unclaimed —
is **not supported**. The coherent reading is the one `docs/30`/`docs/37` already
reached: the evaluator is **libm-bound**, so vectorising `+ - * /` does not move the
rate-limiting work. No `Makevars` change is warranted, and the `#pragma GCC optimize`
fallback is moot.

**Side finding:** the R (`-O2`) / Python (`-O3`) build-flag asymmetry noted in Phase 1
is a documentation curiosity, not a performance defect. Worth a one-line note where the
build is described; nothing to fix.

**Bit-identity: PASS, 5/5 pairs.** Every `-O2`/`-O3` pair returned an identical loss
*and* an identical expression string (e.g. rel_mass s1 `loss=2.3518e-06`, s3
`loss=9.1023e-06`; spring_pe s2 `loss=4.2481e-26`). The profiled build returned the same
expressions as both, so the instrumentation is result-neutral too. Bar 1 is satisfied
for the flag change that will not be made.

### 7.4 Phase 2 step 1 — `eval_cache` capacity sweep: NO-GO, keep 1024

**Setup.** Same box/toolchain as §7. Three trees differing only in
`-DRSYMBOLIC2_EVAL_CACHE_SLOTS` (1024 / 4096 / 16384), Release `-O3`, unprofiled. Grid:
{rel_mass, spring_pe} x {seed 1,2,3} x {OFF, ON@1024, ON@4096, ON@16384}, 4 threads,
300 s budget (all completed). Hit rate and peak RSS are deterministic per cell, so one
rep is exact for them; wall carries the ~3 % run noise measured in §7.3.

**Hit rate — saturated at 1024; capacity was never the limit.**

| problem | seed | @1024 | @4096 | @16384 | gain 1024 -> 16384 |
|---|---:|---:|---:|---:|---:|
| rel_mass | 1 | 0.3337 | 0.3446 | 0.3515 | +1.78 pts |
| rel_mass | 2 | 0.3316 | 0.3416 | 0.3472 | +1.56 pts |
| rel_mass | 3 | 0.3257 | 0.3379 | 0.3453 | +1.96 pts |
| spring_pe | 1 | 0.2457 | 0.2529 | 0.2557 | +1.00 pts |
| spring_pe | 2 | 0.2205 | 0.2239 | 0.2249 | +0.44 pts |
| spring_pe | 3 | 0.2403 | 0.2468 | 0.2490 | +0.87 pts |

**16x the table buys 0.4-2.0 percentage points.** The Phase 2 hypothesis — that
`docs/49`'s 0.19-0.34 hit rate was the table's ceiling rather than the mechanism's — is
**refuted**. In hindsight the reason is structural: an island holds only
`population_size` = 27 members, and the duplicates the cache catches are *temporally
local* re-evaluations, so a 1024-entry direct-mapped table is already far larger than
the working set. 0.19-0.34 is a property of the search, not of the memo.

**Wall — larger tables are neutral to worse.** Median change vs @1024: **+0.72 % at
4096, +1.57 % at 16384**. One cell (rel_mass s2 @4096) took 89.6 s against 55.0 s at
1024 for *identical* work (`n_evals` equal, `cpu/wall` normal at 3.51) — 1.6x the CPU
time for the same evaluations, i.e. not core contention but memory-hierarchy cost. A
148 MB table set randomly probed by 31 islands thrashes LLC/TLB, which is a coherent
explanation for why the slightly better hit rates never turn into time.

**Memory — the number `docs/49` did not record.** Peak RSS, rel_mass:

| arm | peak RSS | delta vs OFF |
|---|---:|---:|
| OFF | 14.9 MB | — |
| ON @1024 | 59.7 MB | **+45 MB** |
| ON @4096 | 145.5 MB | +131 MB |
| ON @16384 | 341.2 MB | +326 MB |

**Even the shipped default of 1024 costs ~45 MB when the option is switched on** (31
per-island tables). Against the web GUI's fixed, non-growable 128 MB WASM heap
(`docs/51`) that is a third of the budget, and 4096 alone exceeds the whole heap. This
is an independent argument for keeping `eval_cache` OFF by default that `docs/49` did
not have.

**ON/OFF speedup replicates `docs/49`.** ON@1024 vs OFF: -9.6 %, -18.0 %, -31.2 %
(rel_mass s1-s3), -12.0 %, -9.8 %, -11.1 % (spring_pe s1-s3); **median -11.6 %**,
against `docs/49`'s -9.7 % median / -15.8 % overall. The mechanism is exactly as good
as it was measured to be — it just cannot be made better by enlarging it.

**Bit-identity: PASS, 24/24.** Every cell returned the identical loss and identical
expression across all four arms, and `n_evals` was identical across arms within each
(problem, seed) — confirming that a cache hit is charged exactly like a real
evaluation, so even `max_evals`-budgeted runs are unaffected.

**Verdict.** `kEvalCacheSlots` stays **1024**. The default stays **OFF**. Step 2 (memory
budget) is answered above; **step 3 (the full 25-problem `docs/49` protocol) is not
run**, because there is no candidate size to promote. Note this screen could not have
tested bar 3 anyway: `bench_profile` carries only `spring_pe` and `rel_mass`, and the
three problems that failed bar 3 in `docs/49` (driven_osc, heat_conduct, torque) are not
among them.

**Kept from this work:** the `RSYMBOLIC2_EVAL_CACHE_SLOTS` compile-time override
(default unchanged, same idiom as `RSYMBOLIC2_JAC_BLOCK_WIDTH`) and `bench_profile`'s
new `eval_cache` arm switch, hit-rate/peak-RSS/`n_evals` reporting — the harness that
makes this re-runnable.

### 7.6 Phase 3 steps 1-2 — data-independence ceiling: GO

**Probe.** A compile-guarded (`RSYMBOLIC2_INDEP_PROBE`, zero footprint by default)
postfix pass in `sse_current` counts, over every forward evaluation of a real run: total
node visits, data-independent node visits (no `Variable` in the subtree), maximal
data-independent subtrees, and the transcendental split. Removable work is
`indep - maximal` (one broadcast per maximal subtree survives). 4 threads, cache OFF,
full 2800-generation runs; deterministic per (problem, seed), so one rep is exact.

| problem | seed | node visits | indep | maximal | removable | trans | indep trans |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel_mass | 1 | 36.20 M | 18.68 % | 10.27 % | 8.41 % | 13.24 % | 3.22 % |
| rel_mass | 2 | 35.61 M | 21.95 % | 12.74 % | 9.21 % | 13.85 % | 3.31 % |
| rel_mass | 3 | 36.06 M | 17.49 % | 9.11 % | 8.38 % | 24.31 % | 3.31 % |
| spring_pe | 1 | 3.60 M | 29.22 % | 14.50 % | 14.72 % | 22.55 % | 5.95 % |
| spring_pe | 2 | 1.12 M | 28.37 % | 13.86 % | 14.51 % | 22.43 % | 5.17 % |
| spring_pe | 3 | 2.10 M | 31.00 % | 14.87 % | 16.13 % | 23.60 % | 6.46 % |

**The pre-registered gate was ambiguous — recorded here rather than resolved in
hindsight.** "The fraction of node x P operations that are data-independent" reads
either as `indep` (17.5-31.0 %, every cell passes) or as `indep - maximal`
(8.4-16.1 %, node-weighted aggregate **9.05 %**, which fails). Worse, both readings
weight a node by *count*, and the natural experiment in the table above shows that is
wrong: rel_mass s1 and s3 have near-identical node visits (36.20 M vs 36.06 M) but s3
carries 1.84x the transcendental density and took 2.01x the wall (59.2 s vs 119.0 s).

**Settled by measuring the weight instead of assuming it.** Per-operator cost of the
production SoA tile kernels (`scratchpad/bench_opcost.cpp`, P=256, memcpy-baseline
subtracted, `-O3`), relative to `Mul`:

| op class | ops | relative cost |
|---|---|---:|
| cheap | Add, Sub, Mul, Neg, Abs, Square | 0.93 - 1.08 |
| reciprocal | Div, Inv | 3.0 - 3.3 |
| sqrt | Sqrt | 9.7 |
| tanh | Tanh | 43.9 |
| log | Log | 141.3 |
| exp / trig | Exp, Sin, Cos | 225 - 242 |
| pow | Pow | **450.2** |

So the transcendental weight is not ~15 but **10 - 450**, and the node-count metric
understated the opportunity by pricing a 237x `exp` node the same as a 1x `add`.

**Cost-weighted removable fraction** of forward-path work, computed at both ends of the
bucket (W = 9.7 prices every "transcendental" as its cheapest member `Sqrt`; W = 200 is
representative of the exp/log/trig that actually dominate):

| problem | seed | at W = 9.7 | at W = 200 |
|---|---:|---:|---:|
| rel_mass | 1 | 16.9 % | 23.7 % |
| rel_mass | 2 | 17.2 % | 23.4 % |
| rel_mass | 3 | 11.9 % | 13.5 % |
| spring_pe | 1 | 22.4 % | 26.1 % |
| spring_pe | 2 | 20.2 % | 22.9 % |
| spring_pe | 3 | 23.7 % | 27.1 % |

**Every cell clears 10 % even at the pessimistic weight** (minimum 11.9 %). Ceiling on
total wall = forward path (~76 %) x [11.9 %, 27.1 %] = **9 % - 21 %**.

**Why the opportunity exists at all** (and why it must live in the evaluator): `exp(c1)`
and `sqrt(c1*c2)` cost 237x and 9.7x a cheap op, and are recomputed for all 256 points
of every tile. `simplify()` would fold them, but it runs only once per epoch per
population member (`optimize_and_simplify_population`), while mutated children are
scored *immediately* by `score_sse`. Folding children earlier is **not** an option: the
simplify cadence is PySR-matched (`docs/29` §A#11), so changing it would change which
trees exist and violate parity. Scalarising inside the evaluator changes only how fast
an unfolded tree evaluates — bit-identical, parity-neutral. This is the design
constraint that makes the evaluator the right layer.

**Gate verdict: GO** — the measurement clears the pre-registered bar in every cell.

**Caveat:** 9-21 % is an *ideal-implementation ceiling*. The per-call independence pass,
the extra branch per node, and boundary broadcast bookkeeping all take from it; the
realistic end-to-end expectation is roughly **5-12 %**. Adoption would still be judged
on the §2 bars measured end-to-end, not on this ceiling.

**DECISION 2026-07-26: DEFERRED, not implemented.** With the gate passed, implementing
was the agreed next step, and it was declined on the cost/benefit: a realistic 5-12 %
does not justify rewriting `evaluate_soa_residual` — the hottest and most
correctness-critical kernel in the engine — which is Priority #5 bought with risk to
#1 (Correctness) and #2 (Maintainability). This is a deliberate hold, **not** a negative
result: the opportunity is real and measured, and the decision can be revisited if
wall-clock ever becomes a binding constraint (it currently is not — `docs/35`: every
gate problem completes its full generation budget in 20-30 % of the time limit).

**Re-entry cost is low.** The `RSYMBOLIC2_INDEP_PROBE` harness is retained (compile-
guarded, zero footprint by default, same idiom as `RSYMBOLIC2_PROFILE`), so the ceiling
can be re-measured on a new workload with one build flag. `scratchpad/bench_opcost.cpp`
is reproduced verbatim in §7.6's method note if the per-operator weights need refreshing
on different hardware.

### 7.7 Phase 4 — tile-width sweep: NO-GO, closed without running

Phase 4 proposed splitting `kStride`'s two roles (stop-poll cadence vs SoA tile width)
and sweeping the tile width over {32, 64, 128, 256} on the hypothesis that the tile
working set interacts with L1. The per-operator cost table (§7.6) makes the experiment
unnecessary, because it bounds the effect before it is run.

Tile width governs **cache and memory-layout behaviour**. But on rel_mass seed 1 the
forward path spends

    transcendental 13.24 % of nodes x weight ~200  =  2648
    everything else 86.76 % of nodes x weight ~1   =    87
    -> transcendental share of forward-path cost   =  ~97 %

i.e. **~97 % of forward-path time is inside scalar libm calls**, not in the loads,
stores and arithmetic that a tile-width change can influence. The ceiling for Phase 4 is
therefore about 3 % of the forward path = **~2 % of wall** — smaller than the ±3 %
run-to-run noise measured in §7.3, so the experiment could not resolve its own effect
even if the effect were at its ceiling.

This is the same finding that closed Phase 1, arriving from the other direction: §7.3
showed vectorising `+ - * /` does not move a libm-bound evaluator, and §7.6 quantifies
exactly how libm-bound it is. `kStride` stays as it is, serving both roles.

### 7.5 Effect on the plan

- **Phase 1: closed NO-GO.** Negative result recorded; no code change.
- **Phase 2: closed NO-GO** (§7.4). Hit rate is saturated at the shipped 1024;
  `kEvalCacheSlots` and the OFF default both stand. Side finding recorded: switching the
  option on costs ~45 MB, which matters for the WASM heap.
- **Phase 3: gate PASSED but DEFERRED by decision** (§7.6). Cost-weighted removable work
  is 11.9-27.1 % of the forward path (9-21 % of wall ideal, 5-12 % realistic) once
  per-operator cost is measured rather than assumed. Not implemented: the gain does not
  justify rewriting the hottest correctness-critical kernel. Harness retained.
- **Phase 4: closed NO-GO on the existing data, without running it** (§7.7).

**Net outcome of this plan: no shipped behaviour changed.** Phases 1 and 2 are measured
negatives, Phase 3 is a measured positive that was declined on cost/benefit, Phase 4 is
open. What the plan did produce is a corrected map of where the engine's time goes
(§7.1), a memory number for `eval_cache` that `docs/49` lacked (§7.4), the per-operator
cost table (§7.6), and re-runnable harnesses for all of it.
- **New, previously unbudgeted:** the LM path is ~21 %, not ~10 %. Nothing there is a
  lever under parity (multi-start is a required setting), but the per-`fit()` allocation
  moves from "dismissed" to "candidate if Phases 2-4 disappoint" (§6).
- **Confirmed closed:** allocator-contention levers (§7.2).

**Follow-up, 2026-07-29 (`docs/67`).** This plan measured everything on Windows and read
"libm-bound" as a statement about the workload. It is also a statement about the platform:
the same source on the same machine runs `exp` 10.5x and `log` 6.4x slower under
Rtools/MinGW than under glibc, and the full search is ~2.4-2.7x slower on Windows. `docs/67`
records the measurement, the isolation that attributes it to libm (pure-arithmetic trees
show *no* gap), and why reduced precision — the obvious reading of §7.7 — turns out to be
the smaller half of the opportunity.

**Raw logs:** not committed (scratchpad); the commands in §3 reproduce them.

## 8. Deliverables

- This file, extended with a Results section per phase (negative results recorded, per
  CLAUDE.md "let results change the plan").
- `docs/49` updated if the `eval_cache` default changes.
- `docs/29` §A entries updated for any adopted mechanism.
- Ubuntu (WSL) verification recorded before any commit that flips a default.
