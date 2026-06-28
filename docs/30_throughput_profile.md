# 30. Throughput profile: where the Feynman search spends its time

> **Superseded for current code (2026-06-27, `docs/35`).** The throughput framing below was
> correct *before* the `docs/31` cadence fix, when hard problems hit the 300 s timeout. After
> `docs/31` (LM 93 % → 10.7 %, ~10× candidates) **nothing times out** — every problem completes
> the parity-fixed `generations = 2800` in 20–30 % of the budget — so throughput is no longer
> the binding recovery constraint. The SoA/MultiDual optimisations recorded here are why the
> timeouts are gone; the SLEEF/vectorisation headroom is confirmed irrelevant to recovery. Read
> `docs/35` for the current bottleneck (per-generation search quality) and the recovery-parity
> result.

## Purpose

`docs/26`/`docs/29` established that matching PySR's defaults does not close the
Feynman recovery gap, because the gap is **throughput**, not feature coverage: at a
fair 4-thread budget every hard problem hits the 300 s timeout. Before optimising we
profiled one representative recovering problem and one timed-out problem to find the
rate-limiting phase. This file records the method and the result, which **refutes the
prior hypothesis** (docs/23 §4 / `bench_evolve`) that the per-child non-fit path
(tree alloc / simplify / HoF) limits throughput.

## Method

Compile-guarded (`RSYMBOLIC2_PROFILE`, off by default → zero footprint in the shipped
R package) per-phase timers at the hot-path call-site boundaries in
`evolutionary_search.cpp`, plus a within-`fit()` residual-vs-Jacobian split in
`least_squares_problem.hpp`. Per-island accumulators (each island is single-threaded
inside the OpenMP parallel-for) summed across islands at run end. Driver:
`standalone/benchmarks/bench_profile.cpp`, which runs `run_evolution` once at the
faithful gate config (`02_feynman_gate.R` BENCH_PARAMS: pop=27, islands=31,
gens=2800, tournament=15, maxsize=30, optimize_probability=0.14, scaling=1040) and
reports wall + cpu/wall.

Why call-site instrumentation and not a sampling profiler: `fit()`, `sse_current()`
and `reoptimize_hof()` all funnel into the same `evaluate<…>()` kernel, so a sampler
reports "N% in evaluate()" without separating child-fit, child-SSE and HoF re-opt —
the very distinction needed here.

Build:
```
cmake -S . -B build-prof -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-rtools.cmake \
      -DCMAKE_BUILD_TYPE=Release -DRSYMBOLIC2_BUILD_TESTS=OFF \
      -DCMAKE_CXX_FLAGS=-DRSYMBOLIC2_PROFILE
cmake --build build-prof --target bench_profile
OMP_NUM_THREADS=4 ./build-prof/standalone/bench_profile.exe rel_mass 60
```

## Result

Phase mix is stable across both problems and across 10/30/60 s budgets;
cpu/wall = 3.9 at OMP_NUM_THREADS=4 (healthy — numbers are not CPU-starvation noise).
Work-seconds summed across 31 islands:

| phase            | spring_pe 60s | rel_mass 60s | rel_mass 30s |
|------------------|--------------:|-------------:|-------------:|
| evolve_fit       | 73.6 %        | 71.7 %       | 69.7 %       |
| reopt_fit        | 16.0 %        | 16.4 %       | 15.2 %       |
| init_fit         |  3.8 %        |  4.2 %       |  7.8 %       |
| evolve_sse       |  6.2 %        |  7.3 %       |  6.9 %       |
| simplify         |  0.3 %        |  0.3 %       |  0.3 %       |
| mutate_xover     |  0.1 %        |  0.1 %       |  0.1 %       |
| tournament + hof | < 0.1 %       | < 0.1 %      | < 0.1 %      |
| migration        | ~0 %          | ~0 %         | ~0 %         |

**Constant-fitting (init+evolve+reopt fit) ≈ 93 % of all compute.** The non-fit
per-child path (simplify + mutate/crossover + tournament + HoF) is **≈ 0.4 %
combined** — so the docs/23 §4 hypothesis is wrong for this workload; optimising that
path cannot help.

Within `fit()` (residual vs Jacobian closures, rel_mass 30 s):

| closure   | time   | share of fit-eval | per call  |
|-----------|-------:|------------------:|----------:|
| Jacobian  | 83.4 s | **76.2 %**        | 4775 µs   |
| residual  | 26.0 s | 23.8 %            |  359 µs   |

The Jacobian is ~13× the residual per call because it runs **k forward-mode Dual
passes over all m = 1000 points, one per constant** (`least_squares_problem.hpp`
`for (kk…) for (i…) evaluate<Dual>`), each pass recomputing the full primal value
(including transcendentals exp/log/sin/cos/sqrt/pow) it could have shared.

**Headline: ≈ 0.93 × 0.76 ≈ 71 % of total search compute is the forward-mode
Jacobian's redundant k passes.**

## Implications (candidate optimisations, evidence-ranked)

1. **Batched / vector-mode dual Jacobian** — collapse the k separate passes into one
   traversal carrying a length-k gradient, eliminating k−1 redundant primal-value and
   transcendental recomputations and k−1 tree traversals. Targets the 71 %. Keeps
   Float64 (no accuracy/parity argument). The AD *method* is an allowed divergence
   (CLAUDE.md). Must verify recovery is unchanged on the gate (FP reassociation shifts
   the trajectory).
2. **Float32 residual + Jacobian, Float64 LM solve** (docs/27 #4) — halves per-eval
   cost on both closures; *more* faithful to PySR (`precision=32`), not less.
   Orthogonal to (1); composes multiplicatively.
3. Out of scope by parity: batching (subsample points), fewer restarts/iterations —
   these are matched PySR settings, not implementation knobs.

The non-fit-path levers (object pools, faster simplify/HoF) are **not** worth pursuing:
together they are 0.4 % of compute.

## Reproduction artefacts

- Instrumentation: `RSYMBOLIC2_PROFILE` blocks in
  `r-package/rsymbolic2/src/evolutionary_search.cpp` and
  `r-package/rsymbolic2/src/rsymbolic/expression/least_squares_problem.hpp`.
- Driver: `standalone/benchmarks/bench_profile.cpp` (target `bench_profile`).
- All instrumentation is compiled out of the shipped R package (macro off by default).

## Optimization 1: batched (vector-mode) dual Jacobian

Implemented `MultiDual<N>`
(`r-package/rsymbolic2/src/rsymbolic/expression/multi_dual.hpp`): a forward-mode dual
carrying an N-wide gradient block, so one traversal differentiates N constant
directions at once. The Jacobian closure in `least_squares_problem.hpp` now runs
`ceil(k/N)` tiled passes instead of k single-direction passes, sharing one primal
evaluation (incl. transcendentals) per block. Block width `kJacobianBlockWidth = 8`
(`-DRSYMBOLIC2_JAC_BLOCK_WIDTH` to override).

**Correctness:** each Jacobian entry is **bit-identical** to the old scalar k-pass code
(same value path, same per-direction formula computed together). `test_multi_dual.cpp`
asserts raw-bit equality of the production closure and a local N=4/8/16 tiling against a
scalar `Dual` k-pass reference, over trees covering every operator, the pow/sqrt guard
branches, and k that exceeds the block width (tiling). Bit-identity means the search is
provably unchanged — all 17 standalone tests (incl. `test_island_model`
thread-1≡thread-4 determinism) pass unmodified.

**Throughput** (`bench_profile rel_mass`, OMP_NUM_THREADS=4, cpu/wall 3.9):

| metric                         | before (k-pass) | after (batched N=8) |
|--------------------------------|----------------:|--------------------:|
| Jacobian per call              | 4775 µs         | **1095 µs** (4.4×)  |
| Jacobian share of fit-eval     | 76 %            | 38 %                |
| evolve_fit per call            | 25 870 µs       | 11 665 µs (2.2×)    |
| candidates evaluated / 30 s    | 23 285          | **46 695** (2.0×)   |

The per-call Jacobian win (4.4×) exceeds the ~2× estimate because the timed-out Feynman
trees are transcendental-heavy (exp/sqrt/pow), so eliminating k−1 redundant
primal/transcendental recomputations and traversals saves more than the derivative
arithmetic alone. **Net: ~2× more candidates per second.** Within `fit()` the residual
(now 62 %) overtakes the Jacobian (38 %) — so the residual/Jacobian eval cost is the
next target, which a Float32 pass (Optimization 2, docs/27 #4) would halve on both.

**Block-width sweep** (rel_mass 20 s, throughput = candidates evaluated):

| N  | Jacobian µs/call | candidates / 20 s |
|----|-----------------:|------------------:|
| 4  | 2043             | 22 839            |
| 8  | 1217             | 28 558            |
| 16 | 1099             | 28 565            |

N=4 splits typical k=5–8 trees into two passes (slower). N=8 and N=16 tie on throughput;
N=16's marginally faster per-call Jacobian is cancelled by wasted lanes on the common
small-k trees, at a larger memory footprint. **N=8 is the default** — same throughput as
16, smaller footprint.

## Optimization 2 (Float32): measured negative — NOT adopted

The plan's next step was Float32 residual + Jacobian (docs/27 #4), on the assumption it
"halves per-eval cost". A direct micro-benchmark refuted that assumption **on this
toolchain**: evaluating a transcendental-heavy Feynman tree (rel_mass, with
exp/log/sin/sqrt/pow) at 1000 points, double vs float, single-point scalar evaluation:

| precision | time   |
|-----------|-------:|
| double    | 4.6 s  |
| float     | 5.0 s  → **0.93× (slower)** |

Reasons: (1) our evaluator is a **scalar interpreter** — one point at a time through a
postfix stack — so Float32's real lever (SIMD lane width: 8 floats vs 4 doubles per AVX
register, plus half the memory bandwidth) never materialises; (2) on Rtools/MinGW/UCRT
the float transcendentals (`expf/sinf/sqrtf/powf`) are not faster than their double
forms, and the extra float↔double conversions add cost. PySR's Float32 speedup comes
from **vectorising over data points** (LoopVectorization.jl), which we do not do.

Float32 is also an *allowed* divergence, not a parity requirement (CLAUDE.md lists
"Float64 core vs PySR precision=32" among the sanctioned implementation differences), so
keeping Float64 is correct — and it avoids the recovery risk of float-precision constant
fitting (loss floor, `target_loss`/NMSE thresholds). **Not adopted.**

**Real prerequisite for a Float32 win (future, larger change):** restructure the
evaluator to process a *batch of points per node* (struct-of-arrays), letting the
compiler auto-vectorise. That would speed up residual AND Jacobian in double already
(no recovery risk), and only then would Float32 add the extra SIMD-width factor. This is
a substantial rewrite of the hot evaluator (`tree.hpp` `evaluate`, both closures, the
batched MultiDual), to be scoped with the user before undertaking.

## SoA point-batched evaluator: evidence micro-benchmark (residual, Float64)

Before committing to the substantial rewrite above, we measured whether an SoA
point-batched residual evaluator is actually faster under the *portable* optimisation
the R package ships with (`-O2`, generic x86-64), to avoid repeating the Float32 trap
(an assumed speedup that the toolchain did not deliver). Driver:
`standalone/benchmarks/bench_soa_eval.cpp` — compares the production
`evaluate<double>()` (one point per call, current path) against a hand-written SoA
evaluator (switch hoisted out of a tight per-point loop), on four trees of increasing
transcendental density. Both compute **bit-identical** residuals (same ops, same order,
no `-ffast-math`); the driver verifies this every run.

Medians of 5 runs, P=1000 points, reps=5000 (short reps on purpose: long reps
thermally throttle the CPU and corrupted an earlier measurement — see note):

| tree                       | ops                  | `-O2` generic | `-O3 -march=native` |
|----------------------------|----------------------|--------------:|--------------------:|
| poly                       | `+ - * square`       | **5.7×**      | **8.3×**            |
| rel_mass `c/√(1-(x/x)²)`    | `/ - square sqrt`    | **3.1×**      | **3.8×**            |
| trig `sin·cos+c`           | `sin cos * +`        | **1.24×**     | **1.25×**           |
| transc `exp+log(·²)+sin`   | `exp log sin square` | **1.23×**     | **1.24×**           |

Findings (the speedup decomposes into two independent effects):

1. **Dispatch-elimination floor (~1.24×, operator-independent, no SIMD).** The
   transcendental-heavy trees gain the *same* ~1.24× at `-O2` generic and at
   `-O3 -march=native` — SIMD adds nothing there because libm's scalar
   `exp/log/sin/cos/pow` do **not** auto-vectorise without a vector-math library
   (SLEEF/libmvec/SVML), confirming the Float32-section diagnosis. The ~1.24× is purely
   from hoisting the per-point interpreter dispatch (the `switch` + `std::vector` stack
   churn) out of the inner loop. `-fopt-info-vec` confirms the kernels do **not**
   vectorise at `-O2` generic yet poly still gains 5.7× — i.e. most of the `-O2` win is
   dispatch elimination, not SIMD.
2. **SIMD upside (vectorisable ops only).** On top of the floor, `+ - * /` and the
   hardware `sqrtpd` vectorise, lifting arithmetic/sqrt-dominated trees to 3.8–8.3× at
   `-O3 -march=native` (and already 3.1–5.7× at `-O2` generic, where GCC≥12 vectorises
   `+ - *` even under the conservative cost model).

**No regression once measured cleanly.** An earlier long-reps run reported trig at
*0.63×* (a regression); that was a CPU thermal-throttling artifact of the ~20 s
per-tree libm-bound loop, not a property of the algorithm. Short reps (≤5 s/tree) give
stable medians with **every tree ≥ 1.2×** — SoA never loses.

**Decision.** Unlike Float32 (a measured net *loss*), SoA point-batching is a measured
net *win* across the whole operator spectrum, and is bit-exact (search provably
unchanged). It is therefore worth implementing. Two caveats bound the expected
end-to-end gain:

- This measures the **residual** only. The Jacobian (38 % of `fit` after Optimization 1)
  needs an SoA *batched-`MultiDual`* (value + N-wide gradient, both SoA), which is the
  harder half and is not yet measured. Start with the residual (clear win) per docs.
- The real Feynman gain depends on the **operator mix of the trees the search actually
  fits**: arithmetic/sqrt-lean trees gain multiplicatively, transcendental-heavy trees
  gain ~1.24×. The documented timed-out trees are transcendental-heavy (`exp/sqrt/pow`),
  so expect the gain to sit between these bounds — `sqrt` is on the winning side,
  `exp/pow` on the floor side.

Vectorising the transcendentals themselves (the remaining headroom on the floor-bound
trees) would require a vector-math library (SLEEF). That is a **separate dependency
decision** (CLAUDE.md: default no; Windows/Rtools cost and a named fallback required),
deferred until the SoA rewrite's own gain is in hand.

## Optimization 3: SoA point-batched evaluator (implemented)

Implemented the SoA evaluator for **both** fit closures
(`r-package/rsymbolic2/src/rsymbolic/expression/soa_eval.hpp`):
`evaluate_soa_residual` (plain double) and `evaluate_soa_jacobian<N>` (vector-mode dual,
value + N-wide gradient, both SoA). The residual/Jacobian closures in
`least_squares_problem.hpp` now tile the points at `kStride` (256) and evaluate a tile
per node; `Dataset` gained a column-major `Xcol` view, transposed once at construction.
`sse_current` (the `evolve_sse` path) was routed through the same residual evaluator with
thread-local scratch. The scalar `evaluate<T>()` is retained for non-hot callers.

**Correctness (bit-identity).** Points are independent and the per-node op order within a
point is unchanged, so moving the point loop inside each node changes nothing
numerically. `test_soa_eval.cpp` asserts the SoA residual is bit-identical to the scalar
`evaluate<double>` and the SoA Jacobian (N=4/8/16) is bit-identical to the scalar k-pass
Dual reference, over every operator, the pow/sqrt guard branches, k past the block width,
and point counts crossing the tile boundary. All 19 standalone tests pass, including
`test_island_model` (thread-1 ≡ thread-4 determinism) — i.e. the search is provably
unchanged, only the evaluator is faster. Float64 throughout (no precision/recovery risk).

**Per-evaluator throughput (residual, `bench_soa_eval`, `-O3` generic, P=1000, reps=5000,
median of 3, every tree bit-exact):**

| tree                       | ops                  | speedup |
|----------------------------|----------------------|--------:|
| poly                       | `+ - * square`       | **8.8×**|
| rel_mass `c/√(1-(x/x)²)`    | `/ - square sqrt`    | **3.75×**|
| trig `sin·cos+c`           | `sin cos * +`        | **1.28×**|
| transc `exp+log(·²)+sin`   | `exp log sin square` | **1.26×**|

Confirms the Optimization-2/SoA diagnosis: arithmetic/`sqrt` trees gain
multiplicatively; libm-transcendental trees gain only the ~1.26× dispatch-elimination
floor (no SIMD without a vector-math library). Reps must stay short (≤5 s/tree); long
reps thermally throttle the libm-bound trees (reps=20000 measured trig at 0.74×, a
throttling artifact, not the algorithm — same caveat as the residual micro-bench above).

**End-to-end (bench_profile `rel_mass`, OMP_NUM_THREADS=4, seed 1, Float64) — honest
result.** Candidate throughput on this machine is **dominated by run-to-run variance**:
at a *fixed seed* the candidates evaluated in 20 s spanned 8.2k–25.8k (pre-SoA, HEAD) and
20.6k–27.2k (SoA) — a ~3× spread driven by turbo/thermal/scheduling, with cpu/wall
healthy (3.7–3.95) throughout. The medians (baseline 19.9k vs SoA 21.9k, ≈ **1.1×**) are
within that noise. This is expected and consistent: `rel_mass` is exactly the
transcendental-heavy *floor* case (≈1.26× per-evaluator), so the end-to-end gain on this
specific timed-out problem is small and cannot be cleanly isolated from machine variance
by this metric. The clean evidence is the per-evaluator micro-benchmark above.

**Conclusion (let results change the plan).** SoA is a bit-exact, zero-recovery-risk
evaluator speedup that is **large for arithmetic/`sqrt`-lean trees (3.75–8.8×)** and
**modest for transcendental-heavy trees (~1.26×)**. The broader Feynman suite contains
many arithmetic/`sqrt`-lean targets that benefit substantially, so it is worth keeping;
but it does **not**, on its own, close the gap on the transcendental-bound timed-out
problems like `rel_mass`. The remaining headroom there is vectorising the transcendentals
themselves, which needs a vector-math library (SLEEF).

### SLEEF: decided NOT to pursue (2026-06-22)

Verified from the installed authoritative source: PySR's defaults are `turbo=False` and
`bumper=False` (`pysr/sr.py`), and `turbo` is the *only* path to vectorised
transcendentals — it maps to `LoopVectorization.@turbo` in SymbolicRegression.jl
(`Options.jl:405`, `turbo::Bool=false`), whose transcendental vectorisation comes from
SLEEFPirates (a Julia SLEEF port). LoopVectorization is a **package extension / weakdep**
in both SymbolicRegression.jl (`Configure.jl` relevant_extensions) and DynamicExpressions.jl
(`[weakdeps] LoopVectorization` → `DynamicExpressionsLoopVectorizationExt.jl`), loaded only
when `turbo` is on. So **PySR's default does not vectorise transcendentals at all** — it is
the same scalar-per-point character as our Float64 path. This is `docs/29` category C #3.

Therefore SLEEF is **not required for PySR default parity** (matching PySR's default means
*not* using it). Adopting it would mean implementing PySR's non-default `turbo` feature for
pure performance — which CLAUDE.md ranks last (Performance #5) and the Dependency Policy
defaults to *no*, weighed against a real Windows/Rtools (MinGW/UCRT) build-and-bundle cost
for a header that must also remain serial-correct where unavailable (CRAN/macOS). **Decision
(user, 2026-06-22): do not add SLEEF in this work.** The SoA structure (node-level point
loops) is left in place so the option can be revisited later if a measured need arises; the
next lever to weigh is instead the *algorithmic* search efficiency (`docs/26` §5: mutation
weights, far-miss `interference`/`driven_osc`), which is throughput-independent.
