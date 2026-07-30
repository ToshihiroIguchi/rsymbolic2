# 67. The libm platform gap: Windows is ~2.5x slower than Linux, and it is not precision

**Date:** 2026-07-29 (WASM arm added the same day, §3.5)
**Status:** **superseded by `docs/68`** (2026-07-30), which measured a fourth arm this
document never tried — MSVC — found that the slow libm is mingw-w64's own rather than
Windows', reached the UCRT's implementations from a MinGW build with no dependency at all,
and implemented the redirect. `docs/68` §9 lists the three statements below that it
corrects: the `bench_soa_eval` batch column (§3.4), the "bounded above by 2.4-2.7x"
estimate (§4, §5), and the decision in §6.1. The measurements here are otherwise unchanged
and still stand.
Originally: measurement record; no code change proposed here was implemented.
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

**And then the third platform decided what to do about it.** Measuring the WASM arm (§3.5)
was supposed to size the prize; instead it removed the reason to claim it. The web GUI — the
only component where wall-clock actually binds — turned out to be on the *fast* side, so the
one urgent case for a replacement evaporated. §6 records the decision to leave the 2.4-2.7x
on the table, and why that is the right call under the Project Priorities rather than a
failure of nerve.

## 2. Setup

Same physical machine throughout: Intel Core 7 150U, Windows 11.

| arm | toolchain | libm |
|---|---|---|
| Windows | Rtools45 g++ 14.3.0 (MinGW/UCRT) | mingw-w64 |
| Linux | Ubuntu 24.04.4 LTS under WSL2 (kernel 6.6.114.1), g++ 13.3.0 | glibc 2.39 |
| WASM | emsdk 6.0.2 (clang), run under Node 22.16.0 / V8 | emscripten (musl-derived) |

All arms: identical sources at `e5a5f1f`, `-O3 -DNDEBUG`, no extra flags. The native arms use
CMake `Release` and `OMP_NUM_THREADS=4` — the 4-thread cap is `docs/60` §7.2's comparison
point, where island granularity (31 islands over 4 workers) puts the efficiency ceiling at
97 %. The WASM arm is single-threaded by construction (`web/wasm/CMakeLists.txt` ships no
pthreads so the site needs no COOP/COEP headers), and only the two single-threaded
micro-benchmarks were run on it; §3.3's end-to-end search was not.

Parallel health was checked on both native arms before drawing any conclusion, per the
standing requirement that wall-clock numbers are void if the CPU is starved: **Windows
`cpu/wall` 3.49-3.79, Linux 3.83-3.94** at the 4-thread cap. Both healthy; Linux marginally
better, which accounts for a few points of the end-to-end gap and is noted again in §3.3.

**Repetition counts differ by arm and the tables say which.** The native numbers in §3.1-§3.4
are n = 1. The WASM numbers in §3.5 are **medians of 5**, because run-to-run spread under V8
reached 17 % — larger than on either native arm, and large enough that a single run would not
have supported the comparison.

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

### 3.5 The third platform: WASM is on the fast side

This is §7's question 2, and it was measured first because it is the cheapest and because the
web GUI is the one component where `docs/66` §6 found time to be the binding constraint.
`bench_libm.cpp` and `bench_soa_eval.cpp` both build under emsdk unmodified, as intended.

**Raw libm, net ns/elem, medians of 5 (native columns repeated from §3.1, n = 1):**

| op | Windows | Linux | WASM | WASM / Linux | Windows / WASM |
|---|---:|---:|---:|---:|---:|
| `std::exp` | 27.63 | 2.64 | 8.27 | 3.13x | **3.34x** |
| `std::log` | 16.13 | 2.51 | 9.36 | 3.73x | 1.72x |
| `std::sin` | 28.03 | 4.91 | 9.44 | 1.92x | 2.97x |
| `std::pow` | 53.47 | 7.77 | 11.17 | 1.44x | **4.79x** |
| `std::sqrt` | 1.02 | 1.11 | 1.12 | 1.01x | — |
| `mul` | 0.036 | 0.012 | 0.61 | ~50x | — |

Read naively this puts WASM in the middle, 1.4-3.7x behind Linux. **That reading is wrong**,
and the reason is the same control that made §3.1 attributable in the first place.

**The codegen tax — the correct normaliser.** §3.2's load-bearing observation was that our own
replacements cost the *same* on Windows and Linux (3.51 vs 3.55), which is what isolated libm
as the variable. That symmetry does not survive the third platform:

| replacement (our code, not libm) | Linux | WASM | WASM / native |
|---|---:|---:|---:|
| `exp` deg10 | 3.55 | 5.91 | 1.66x |
| `exp` deg10, guarded | 3.73 | 7.98 | 2.14x |
| `log` s17 | 3.35 | 5.12 | 1.53x |
| `log` s17, guarded | 3.63 | 6.19 | 1.71x |
| `pow` acc | 11.68 | 15.07 | 1.29x |
| `pow` fast | 7.61 | 10.92 | 1.43x |

**WASM runs our own scalar floating-point code ~1.6x slower than native (median).** That is a
flat platform tax with nothing to do with libm. Dividing it out, emscripten's libm sits at
**0.9-2.3x** of glibc's quality — `pow` is actually better than glibc scaled — i.e. the same
class. MinGW is 5.7-10.5x off *with no such excuse available*, because native codegen is
identical on both native arms.

(The `log s11` / `log s5` cells remain untrustworthy on this arm too: WASM reports s11 at 6.67,
slower than the higher-degree s17 at 5.12. Three toolchains now disagree about these two rows
in three different directions, which confirms §3.2's diagnosis that they are codegen artefacts
of the shared `log_split` and not measurements of anything.)

**Production evaluator (`bench_soa_eval`), batch ns per 1000-point pass** — no search
stochasticity, identical work, the same fixed trees as §3.4:

| tree | Windows | Linux | WASM | WASM / Linux | WASM / Windows |
|---|---:|---:|---:|---:|---:|
| poly (pure arithmetic) | 1344 | 1348 | 3187 | **2.36x** | 2.37x |
| rel_mass (1 sqrt + div) | 5732 | 3962 | 4410 | 1.11x | 0.77x |
| trig (sin*cos) | 58208 | 11794 | 18221 | 1.55x | **0.31x** |
| transc (exp+log+sin) | 76126 | 12056 | 17663 | 1.47x | **0.23x** |

**This is the mirror image of §3.4 and it is the load-bearing result.** There, the gap *grew*
with transcendental content (1.00x → 6.31x), which is what convicted libm. Here the gap
*shrinks* with transcendental content (2.36x → 1.47x): WASM is at its worst on pure
arithmetic and better than its own average tax once transcendentals dominate. libm is not the
culprit on this platform — the flat codegen tax is, and the transcendental rows are where WASM
does *best* relative to Linux.

And against the platform that has the problem, **WASM is 3.2-4.3x faster than Windows** on
transcendental trees.

**Would a replacement help WASM? No.** Comparing `std::` against our guarded replacements
*within* the WASM arm: `log` 9.36 → 6.19 (1.51x faster), `exp` 8.27 → 7.98 (1.04x, noise),
`pow` 11.17 → 15.07 (**0.74x, slower**). Net ≈ zero, and that is with expedient replacements
measured against a real libm; a vendored implementation chosen to match glibc rather than beat
it would land in the same place. On Linux the replacements already lose (§4, point 4).

**Side finding, third data point.** §3.4 noted the SoA design's benefit is platform-dependent.
The batch/scalar speedup on transcendental trees is 1.24-1.27x on Windows, 2.66-3.30x on
Linux, and **3.81-3.84x on WASM** — the highest of the three. `soa_eval.hpp`'s stated purpose
is realised best exactly where it was never measured.

**A new lever appeared, and it is not libm.** The poly row above (WASM 2.36x slower than *both*
natives, on code containing no libm call at all) points at missing vectorisation rather than
missing precision, so `-msimd128` was tested:

| tree | WASM | WASM `-msimd128` | gain |
|---|---:|---:|---:|
| poly | 3187 | 2419 | 1.32x |
| rel_mass | 4410 | 2781 | **1.59x** |
| trig | 18221 | 17257 | 1.06x |
| transc | 17663 | 15617 | 1.13x |

Exactly the shape the whole document predicts: elementwise `+-*/` and `sqrt` vectorise,
libm calls do not. **On the realistic transcendental-heavy workload it is worth 1.06-1.13x** —
real, cheap, and small. Carried to §6 as candidate D. Two things are *not* established about
it: the driver's `bit-exact` column verifies batch against scalar *within one build*, and says
nothing about whether a SIMD build agrees with a non-SIMD one (elementwise IEEE operations
should, since no reassociation is licensed without `-ffast-math`, but that is an argument, not
a measurement); and browser engines were not tested, only Node.

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
5. **WASM is on the fast side, and it is the platform that mattered most** (§3.5). Emscripten's
   libm is glibc-class once WASM's flat ~1.6x scalar-codegen tax is divided out; the production
   evaluator's WASM/Linux gap *shrinks* from 2.36x to 1.47x as transcendental content rises,
   the opposite of the Windows signature. Replacing libm there would gain nothing (`log` 1.51x,
   `exp` 1.04x, `pow` 0.74x — net ≈ zero). **Windows is alone in having this problem.**
6. **The web GUI's residual WASM tax is vectorisation, not precision** (§3.5). It is worth
   1.06-1.13x on transcendental-heavy trees via `-msimd128`, and 1.32-1.59x on arithmetic ones.

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
  fallback, and non-bit-identity. This work weakens none of them. A *scalar* replacement
  would be a different proposition with a much lower dependency cost — that is candidate A,
  which §6 declines on its own merits, not by inheriting SLEEF's reasons.
- **That `-msimd128` preserves bit-identity** (candidate D, §3.5). The argument is that only
  elementwise IEEE operations are available to vectorise and no reassociation is licensed
  without `-ffast-math`, so results should be unchanged. That is reasoning, not a measurement;
  the driver's `bit-exact` column compares batch against scalar *within* one build and cannot
  answer it. Nor were browser engines tested — the WASM arm is Node/V8 only.

## 6. Candidates, and the decision

**Decision after §3.5: C — do nothing, for now.** A is not refuted, but its case has shrunk
to the point where it no longer clears the Dependency Policy's bar. The reasoning is in §6.1
below, after the candidates it refers to. A is demoted rather than rejected: see §6.2.

All of A/B/D break bit-identity with the current build, which is the gate they have to clear
(§7, question 3).

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
GUI (`docs/66` §6: the browser row ceiling sits at a time wall, not a memory wall) — and
§3.5 now shows that build is on the fast side, so A would deliver it nothing.

**D. Build the WASM target with `-msimd128`** (new, from §3.5). Not a libm change at all: it
addresses the *other* thing §3.5 found, WASM's flat vectorisation tax. One line in
`web/wasm/CMakeLists.txt`, no dependency, aimed squarely at the one binding constraint.
Measured 1.06-1.13x on transcendental-heavy trees and 1.32-1.59x on arithmetic ones — real
but small; it moves `docs/66`'s browser row ceiling by ~10 %, not by a factor. Before it could
be adopted it needs the SIMD-vs-non-SIMD bit-identity question settled by measurement rather
than by argument (§3.5), and a check on browser engines rather than Node alone. Recorded as
the cheapest live option, not proposed for implementation here.

### 6.1 Why C, and what changed

§6C used to carry an escape hatch: wall-clock does not bind the library, *but* it binds the
web GUI, and that arm was unmeasured. §3.5 closed that hatch in the negative. The web GUI is
on the fast side — 3.2-4.3x faster than Windows on transcendental trees — so the single
component with a real time constraint gains nothing from A.

What is left of A's case is (i) Windows-only speed, which `docs/35` says is not binding, and
(ii) cross-platform bit-identity, which is a **reproducibility** benefit, not a performance
one. Against that stands: a vendored dependency, where the Dependency Policy's default answer
is no and the burden of proof is on adopting; an unverified licence; a one-time bit-identity
break; and regenerating every `docs/65` §6 baseline plus rebasing the WASM parity gate.

The trade-off being made explicit, per Project Priorities: this leaves a **measured 2.4-2.7x
on a mandatory platform unclaimed**. That is deliberate. Performance ranks last, the gain is
not needed by any current constraint, and the cost lands on Portability and Simplicity, which
rank above it. If Windows wall-clock ever becomes binding — a much larger default `maxsize`,
a substantially bigger dataset, or a Windows-hosted service — this document is the evidence
that the lever exists and is worth ~2.5x, and the decision should be revisited then.

### 6.2 A is demoted, not rejected

A's surviving benefit is that Windows, Linux and WASM would produce identical results for the
first time. That is a genuine defect being lived with — five of six cells in §3.3 diverged,
and `docs/51` records WASM differing from native by libm ULP. But it should be judged **as a
reproducibility project, on reproducibility's merits and costs**, not smuggled in on a
performance argument that §3.5 has just weakened. Filed accordingly; no work proposed.

B is unchanged and remains strictly worse than A.

## 7. Open questions, in the order they gate a decision

1. **Licence and integration cost of the vendored implementation.** Under the Dependency
   Policy the default answer is no, and the burden of proof is on adopting. What it must
   show: permissive licence compatible with Apache-2.0, no build-system requirements beyond
   adding source files, and a named fallback (here: a compile flag reverting to `std::exp`).
   **Not investigated** — §6.1 decided against A before this became worth the effort. It
   reopens only if A is revived.
2. ~~**What is emscripten's libm?**~~ **ANSWERED (§3.5): the fast side.** It is glibc-class
   once WASM's flat ~1.6x scalar-codegen tax is divided out, and the production evaluator's
   WASM/Linux gap *shrinks* as transcendental content rises rather than growing. Windows is
   alone in having this problem. This answer is what decided §6 against A.
3. **Does bit-identity have to break?** It does, once, against today's build — for A, B *or*
   D. That is an auditable one-time event, not ongoing nondeterminism, and
   `diag_search_digest` exists to characterise it. But every baseline captured under
   `docs/65` §6 would need regenerating, and the WASM parity gate would need rebasing.
   Still open, and now gating D rather than A. D is the cheap case to settle first: if a
   `-msimd128` build turns out to be bit-identical to a non-SIMD one (§3.5 argues it should
   be, having only elementwise IEEE operations to vectorise), D costs nothing here at all.
4. **Only if A or B is revived**: implement, and measure end to end against `docs/60` §2's
   pre-registered bars, on both platforms, with medians over ≥ 5 runs.

## 8. Reproduction

Native (both arms, same commands):

```
cmake -S . -B build-win  && cmake --build build-win --target bench_libm bench_soa_eval bench_profile
./build-win/standalone/bench_libm.exe
./build-win/standalone/bench_soa_eval.exe
OMP_NUM_THREADS=4 ./build-win/standalone/bench_profile.exe rel_mass 300 1
```

WASM (§3.5). Both drivers build under emsdk unmodified — `bench_libm.cpp` links nothing by
design, and `bench_soa_eval.cpp` needs only the core include path. `-fexceptions` matches
`web/wasm/CMakeLists.txt`; add `-msimd128` for candidate D's rows. Report medians of 5: V8's
run-to-run spread reaches 17 %.

```
em++ -std=c++17 -O3 -DNDEBUG standalone/benchmarks/bench_libm.cpp \
     -o bench_libm.js -sENVIRONMENT=node
em++ -std=c++17 -O3 -DNDEBUG -fexceptions -I r-package/rsymbolic2/src \
     standalone/benchmarks/bench_soa_eval.cpp -o bench_soa.js -sENVIRONMENT=node
node bench_libm.js && node bench_soa.js
```

`bench_libm.cpp` is committed rather than left in a scratchpad deliberately: `docs/60` §7.6
promised `bench_opcost.cpp` would be reproducible from a scratchpad copy, and that file no
longer exists.
