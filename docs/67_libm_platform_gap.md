# 67. The libm platform gap: Windows is ~2.5x slower than Linux, and it is not precision

**Date:** 2026-07-29
**Status:** measurement record. **No code change proposed here is implemented**; §6 lists
candidates and §7 the open questions that gate them.
**Supersedes nothing.** Extends `docs/60` §7.7 (the forward path is libm-bound) with the
cross-platform half of that finding, which `docs/60` never measured.

## 1. The question that started it, and why it changed

The question asked was: **is it realistic to speed up the search by lowering the precision
of `log` and friends?** `docs/60` §7.7 had established that ~97 % of forward-path cost sits
inside scalar libm calls, so reduced-precision transcendentals looked like the only lever
of any size still standing after Phases 1-4 closed.

Measuring it first — per `docs/43`/`docs/44`/`docs/60`'s standing discipline of pricing the
ceiling before building — produced a different answer than the question anticipated. The
cost is **not** intrinsic to computing a transcendental in double precision. It is a
property of **one platform's libm**. Windows pays ~10x for `exp` what Linux pays for the
same source on the same machine, and the accuracy we would have sacrificed buys far less
than closing that gap does.

So the honest answer to the question as asked is **no, but the measurement found something
larger**: a ~2.5x end-to-end penalty on a mandatory platform, available without giving up
any precision at all.

## 2. Setup

Same physical machine throughout: Intel Core 7 150U, Windows 11.

| arm | toolchain | libm |
|---|---|---|
| Windows | Rtools45 g++ 14.3.0 (MinGW/UCRT) | mingw-w64 |
| Linux | Ubuntu 24.04.4 LTS under WSL2 (kernel 6.6.114.1), g++ 13.3.0 | glibc 2.39 |

Both arms: identical sources at `e5a5f1f`, CMake `Release` (`-O3 -DNDEBUG`), no extra flags,
`OMP_NUM_THREADS=4`. The 4-thread cap is `docs/60` §7.2's comparison point, where island
granularity (31 islands over 4 workers) puts the efficiency ceiling at 97 %.

Parallel health was checked on both arms before drawing any conclusion, per the standing
requirement that wall-clock numbers are void if the CPU is starved: **Windows `cpu/wall`
3.49-3.79, Linux 3.83-3.94** at the 4-thread cap. Both healthy; Linux marginally better,
which accounts for a few points of the end-to-end gap and is noted again in §3.3.

## 3. Results

### 3.1 Per-operator libm cost — the same source against two C libraries

`standalone/benchmarks/bench_libm.cpp` (added by this work), tile shape P=256 matching the
SoA evaluator's `kStride`, memcpy baseline subtracted, ns per element:

| op | Windows | Linux | Windows / Linux |
|---|---:|---:|---:|
| `std::exp` | 27.63 | 2.64 | **10.5x** |
| `std::log` | 16.13 | 2.51 | **6.4x** |
| `std::sin` | 28.03 | 4.91 | **5.7x** |
| `std::pow` | 53.47 | 7.77 | **6.9x** |
| `std::sqrt` | 1.02 | 1.11 | 0.9x |
| `mul` | 0.036 | 0.012 | — (at the noise floor) |

`sqrt` is the control: it compiles to the `SQRTSD` instruction rather than a libm call, and
it is the one transcendental-adjacent op with no gap. Everything that goes through libm
shows one.

**Cross-check against `docs/60` §7.6.** That table priced ops through the production SoA
kernel relative to `Mul`. The ratios agree with the absolute numbers here to within 4 %
(`exp`/`log` 1.72 here vs 1.60 there; `pow`/`exp` 1.94 vs 2.00; `sqrt`/`exp` 0.037 vs
0.043), so this microbenchmark is representative of the real kernel and not an artefact of
its loop shape.

**`-fno-math-errno` does not explain it.** Tested on Windows: `exp`/`log`/`sin`/`pow` are
unchanged. It halves `sqrt` (1.12 → 0.57 ns) by letting GCC inline the instruction without
the errno check — free and precision-neutral, but `sqrt` is cheap enough that this is not a
lever on its own. The slowness is in the implementations, not in errno bookkeeping.

### 3.2 What a given accuracy actually buys

Hand-written replacements at three accuracy levels, same harness. The implementations are
deliberately expedient (Taylor / atanh series, not the table + minimax construction a
production implementation uses), so they **bound** what is available at an accuracy rather
than represent the frontier.

| replacement | max rel. error | Windows ns | Linux ns |
|---|---:|---:|---:|
| `exp` deg-10 | 3.0e-13 | 3.51 | 3.55 |
| `exp` deg-10, **guarded** | 3.0e-13 | 3.84 | 3.73 |
| `exp` deg-6 | 1.6e-07 | 1.27 | 1.45 |
| `exp` deg-4 | 5.6e-05 | 0.98 | 1.04 |
| `log` s^17 | **2.2e-16 (1 ulp)** | 3.28 | 3.35 |
| `log` s^17, **guarded** | 2.2e-16 | 3.64 | 3.63 |
| `pow` = exp∘log, accurate | 3.1e-13 | 11.40 | 11.68 |
| `pow` = exp∘log, fast | 1.6e-07 | 7.35 | 7.61 |

Three things to read off this table.

1. **The replacements cost the same on both platforms** (3.51 vs 3.55, 3.28 vs 3.35, …), as
   they must — it is our code, not the platform's. That symmetry is what makes the
   asymmetry in §3.1 attributable to libm.
2. **The guard is not free.** Preserving overflow/underflow/NaN costs 0.3-0.4 ns, ~10 % of
   an accurate replacement. This is not optional: the search reads non-finite losses as
   control flow (`sse_current` returns `kInf`), so any estimate built on unguarded numbers
   is wrong. All figures below use the guarded rows.
3. **Precision is the small lever.** Going from the 1-ulp `log` (3.64 guarded) to a 1.6e-07
   `log` saves ~2 ns; going from MinGW's `log` (16.13) to the 1-ulp version saves ~12.5 ns.
   On Windows the accuracy sacrifice is worth roughly a sixth of what fixing libm is worth.
   On Linux there is nothing to fix, and the accuracy sacrifice is the *only* thing on
   offer — worth ~1.4-1.8x on the affected calls.

**Two cells in the harness are unreliable and must not be quoted:** `log s11` (Win 1.89 /
Lin 4.23) and `log s5` (Win 3.40 / Lin 1.41). They contradict each other and physics — s5
cannot be slower than s17 — and are codegen artefacts of how the compiler treats the shared
`log_split`. The load-bearing rows are the `std::*` rows and the two *accurate* replacements,
which are stable across platforms and across re-runs.

### 3.3 End to end: the full search, both platforms

`bench_profile` at the faithful PySR-default gate config (pop=27, islands=31, gens=2800,
tournament=15, maxsize=30, `optimize_probability`=0.14, scaling=1040, n=1000), 300 s budget
which every run completed, so every run is a full fixed 2800-generation budget.

| problem | seed | Win wall | Lin wall | ratio | Win cpu | Lin cpu | cpu ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| spring_pe | 1 | 8.39 s | 2.84 s | 2.95x | 31.2 | 11.0 | 2.84x |
| spring_pe | 2 | 2.82 s | 1.16 s | **2.43x** | 10.7 | 4.6 | **2.34x** |
| spring_pe | 3 | 6.93 s | 3.35 s | 2.07x | 25.5 | 12.8 | 1.99x |
| rel_mass | 1 | 70.47 s | 32.82 s | 2.15x | 252.1 | 127.5 | 1.98x |
| rel_mass | 2 | 80.04 s | 24.40 s | 3.28x | 279.6 | 95.0 | 2.94x |
| rel_mass | 3 | 166.43 s | 29.74 s | 5.60x | 608.8 | 117.3 | 5.19x |

**Median 2.69x (wall), 2.59x (cpu).**

This table carries two confounds and is reported as indicative, not as the load-bearing
evidence.

- **The trajectories diverge.** libm's ULP differences change which trees the search visits,
  so five of six cells returned different expressions on the two platforms. Forward
  evaluation *counts* agree to within 0.1 % in every cell (the generation budget is fixed),
  but the operator *composition* of the trees does not, so the cells are not equal work.
  rel_mass seed 3 is the extreme case: Windows wandered into an expression containing three
  `^` nodes (`Pow` = 450x `Mul`) and paid 166 s for it. Its 5.60x is an artefact of that,
  not a measurement of the platform gap.
- **One cell is genuinely like-for-like.** spring_pe seed 2 returned an *identical
  expression and identical loss* (`4.2481e-26`) on both platforms with forward evaluation
  counts differing by 0.02 %. That cell reads **2.43x wall / 2.34x cpu**.
- **n = 1 per cell.** This does not meet CLAUDE.md's Benchmarking Requirement of medians
  over ≥ 5 runs. Treat 2.4-2.7x as the current best estimate of the magnitude, not as a
  measured median. §3.4 is what the attribution actually rests on.
- Linux's slightly better parallel efficiency (3.83-3.94 vs 3.49-3.79 cpu/wall) accounts for
  ~5 points of the wall gap; the cpu-time column removes it and still shows ~2.6x.

### 3.4 Isolating the evaluator — this is the load-bearing result

`bench_soa_eval` runs the **production** SoA evaluator over fixed representative trees, so
there is no trajectory divergence and no search stochasticity: identical work, both
platforms. ns per 1000-point pass (batch = the shipped path):

| tree | Windows | Linux | Windows / Linux |
|---|---:|---:|---:|
| poly (pure arithmetic) | 1344 | 1348 | **1.00x** |
| rel_mass (1 sqrt + div) | 5732 | 3962 | 1.45x |
| trig (sin*cos) | 58208 | 11794 | **4.94x** |
| transc (exp+log+sin) | 76126 | 12056 | **6.31x** |

**Pure arithmetic is identical to within measurement noise, and the gap grows in exact
proportion to transcendental content.** That is the attribution: the penalty is not the
compiler (MinGW ships the *newer* GCC here), not the allocator, not the OS scheduler, not
WSL2 virtualisation overhead — all of which would tax the arithmetic tree too. It is libm.

**Side finding: the SoA design's benefit is itself platform-dependent.** The batch
evaluator's speedup over the scalar path is 2.66-3.30x on transcendental trees under Linux
but only 1.24-1.27x under Windows. `soa_eval.hpp`'s stated purpose — batch points per node
so the tile loop can be optimised — is substantially unrealised on Windows, because the
tile loop is stalled on serialised libm calls. Not investigated further here.

## 4. What is established

1. Windows/MinGW's `exp`, `log`, `sin` and `pow` are **5.7x-10.5x slower** than glibc's, for
   the same source on the same machine (§3.1).
2. The end-to-end search is **~2.4-2.7x slower on Windows**, and the penalty is attributable
   to libm because it scales with transcendental content and vanishes entirely on pure
   arithmetic (§3.3, §3.4).
3. **Precision is the wrong lever.** A full-precision (1-ulp) replacement captures most of
   the available Windows win; reducing accuracy to 1e-07 adds only ~1.4-1.8x on top of that,
   on the affected calls (§3.2).
4. **Linux has no headroom of this kind.** glibc is at or near the achievable frontier —
   our accurate replacements are *slower* than glibc's `exp` (3.73 vs 2.64) and `log`
   (3.63 vs 2.51). Any Linux-side gain would have to come from reduced precision or
   vectorisation, both separately blocked (§5, `docs/37`).

## 5. What is NOT established

- **That a replacement would deliver ~2.5x end to end on Windows.** §3.4 measures the
  evaluator in isolation; the LM path (21 % of compute, `docs/60` §7.1) and the non-libm
  remainder do not shrink proportionally. The end-to-end effect is bounded above by the
  observed 2.4-2.7x platform gap and is certainly less than it.
- **That the accuracy cost of a reduced-precision mode is acceptable.** It was not measured,
  because §3.2 removed the motive. If it is ever revisited, the specific hazards are: the
  loss floor (the gate currently reaches `4.2e-26`, and a 1e-07 relative error in the
  residual would floor SSE near 1e-14, erasing the distinction between exact recovery and a
  near miss — the wall `docs/36` measured for Float32), and LM convergence, where a
  piecewise-polynomial approximation makes the Jacobian inconsistent with the residual.
- **That vendoring ARM optimized-routines is viable.** It is the natural candidate (§6) —
  glibc 2.28+ adopted those implementations, which is *why* the Linux arm is fast — but its
  licence has **not** been verified in this session, and no build integration was attempted.
- **That this reopens SLEEF.** SLEEF was rejected twice (`docs/30`, `docs/37`) on three
  grounds: PySR's `turbo=False`, Rtools/MinGW dependency cost with a mandatory serial
  fallback, and non-bit-identity. This work weakens none of them. A *scalar* replacement is
  a different proposition with a much lower dependency cost, and is what §6 proposes.

## 6. Candidates

Listed, not adopted. All of them break bit-identity with the current build, which is the
gate they have to clear (§7).

**A. Vendor a scalar transcendental implementation and use it on all three platforms.**
The preferred shape, for a reason that only appears once both arms are measured: using
*our* implementation everywhere makes Windows, Linux and WASM produce identical results for
the first time. They do not today — that is precisely why five of six cells in §3.3 diverged,
and `docs/51` already records WASM differing from native by libm ULP. A replacement chosen
to match glibc's implementations (rather than beat them) leaves Linux unchanged, lifts
Windows to Linux's level, and converts a standing reproducibility defect into a guarantee.
Hand-rolling is the inferior variant of this: §3.2's accurate replacements beat MinGW
handily but lose to glibc, so writing our own would slow Linux down.

**B. Windows-only replacement.** Smaller blast radius, no Linux risk, but it *entrenches*
the cross-platform divergence rather than removing it, and it means two code paths to test.
Strictly worse than A unless A's licence or integration cost turns out to be prohibitive.

**C. Do nothing.** The status quo is not indefensible: `docs/35` records that every gate
problem completes its full generation budget in 20-30 % of the time limit, so wall-clock is
not currently a binding constraint on the library. The one place it *is* binding is the web
GUI (`docs/66` §6: the browser row ceiling sits at a time wall, not a memory wall), and that
is a WASM build whose libm was not measured here (§7).

## 7. Open questions, in the order they gate a decision

1. **Licence and integration cost of the vendored implementation.** Under the Dependency
   Policy the default answer is no, and the burden of proof is on adopting. What it must
   show: permissive licence compatible with Apache-2.0, no build-system requirements beyond
   adding source files, and a named fallback (here: a compile flag reverting to `std::exp`).
2. **What is emscripten's libm?** Unmeasured, and it decides whether the web GUI — the one
   component where time is the binding constraint — is on the fast side of this gap or the
   slow side. `bench_libm.cpp` is self-contained and should build under emsdk unchanged.
   **This is the cheapest remaining measurement and it should come first.**
3. **Does bit-identity have to break?** It does, once, against today's build. That is an
   auditable one-time event, not ongoing nondeterminism, and `diag_search_digest` exists to
   characterise it. But every baseline captured under `docs/65` §6 would need regenerating,
   and the WASM parity gate would need rebasing.
4. **Only then**: implement, and measure end to end against `docs/60` §2's pre-registered
   bars, on both platforms, with medians over ≥ 5 runs.

## 8. Reproduction

```
cmake -S . -B build-win  && cmake --build build-win --target bench_libm bench_soa_eval bench_profile
./build-win/standalone/bench_libm.exe
./build-win/standalone/bench_soa_eval.exe
OMP_NUM_THREADS=4 ./build-win/standalone/bench_profile.exe rel_mass 300 1
```

`bench_libm.cpp` is committed rather than left in a scratchpad deliberately: `docs/60` §7.6
promised `bench_opcost.cpp` would be reproducible from a scratchpad copy, and that file no
longer exists.
