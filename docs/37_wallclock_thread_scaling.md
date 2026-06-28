# 37. Wall-clock under PySR parity: thread scaling is the only result-preserving lever

**Date:** 2026-06-27

## Goal (decided with the user)

Reduce **wall-clock** so real-world problems can be tried casually, **without changing any
PySR parameter** — same settings, same per-generation search, same result, computed faster.
This is *goal A*: speed only, recovery unchanged. It is pure Performance (CLAUDE.md #5), so
it is pursued with measurement, not speculation. The contrast (*goal B*, spending saved time
on more-than-PySR generations to broaden the search) was explicitly **not** chosen — it would
violate the parity-fixed `generations = 2800` budget and is recorded only as the rejected
alternative.

This file supersedes nothing; it complements `docs/30`/`docs/35` (throughput is a dead lever
for *recovery*) by measuring the one thing that still matters for *wall-clock* under parity:
parallel scaling.

## Why threads, and why the result is unaffected

Under parity the only sanctioned divergences are *implementation* (CLAUDE.md): the parallelism
mechanism is one of them, the settings are not. The island model is **bit-deterministic across
thread counts** (`test_island_model` asserts thread-1 ≡ thread-4 identical output). So changing
the thread count changes *only* wall-clock: the same islands reach the same losses at the same
generations and return the identical expression. Thread count is therefore a pure wall-clock
knob with **zero parity or recovery impact** — unlike `generations`/`optimizer_*`, which are
must-match settings, the thread count is not a PySR search parameter at all.

Because the workload is identical across thread counts, `cpu/wall` (effective parallelism over
a fixed time window) **equals the wall-clock speedup vs serial** for the same total work.

## Measurement

Driver: `bench_profile rel_mass` — the canonical transcendental/`sqrt`-heavy Feynman problem
(`m0/√(1-(v/c)²)`), at the faithful gate config (pop=27, islands=31, gens=2800, tournament=15,
maxsize=30, optimize_probability=0.14, scaling=1040). Built at the **shipped optimisation**
(`-O2` generic, the flag the R package compiles with), **not** `-O3 -march=native`, so the
absolute numbers reflect what a user actually gets. Single machine, seed 1, `n = 1000`.

`rel_mass` runs the **full 2800-generation budget** here (it does not hit `target_loss`,
loss ≈ 2.35e-6), giving a fixed, repeatable workload — ideal for a clean scaling ratio.

### Scaling (cpu/wall = effective parallelism = wall-clock speedup vs serial)

| OMP threads | cpu/wall | parallel efficiency |
|------------:|---------:|--------------------:|
| 1  | 0.99 | 100 % (baseline) |
| 2  | 1.93 | 96 % |
| 4  | 3.68 | 92 % |
| 6  | 5.19 | **86 %** |
| 8  | 6.43 | 80 % |
| 12 | 7.99 | 67 % |

### Absolute wall (full 2800-gen `rel_mass`, serial cpu ≈ 298 s)

| threads | wall | speedup |
|--------:|-----:|--------:|
| 1  | ≈ 298 s | 1.0× |
| 4 (benchmark fairness cap) | ≈ 81 s | 3.7× |
| 6  | ≈ 57 s | 5.2× |
| 12 | **38 s (measured)** | 7.85× |

## Findings

1. **Threads are the only result-preserving wall-clock lever, and it is already implemented
   and near-ideal.** Scaling is near-linear to the physical core count (≥ 86 % efficiency
   through 6 threads). At 6 threads `cpu/wall = 5.19` against an ideal of `31/⌈31/6⌉ = 5.17`
   — i.e. the island load-balance is **already essentially perfect**; there is no meaningful
   headroom in the parallel scheduler to reclaim.

2. **12 threads is not 12× — this is a hardware (hyper-threading) ceiling, not software.**
   Efficiency falls 86 % → 67 % from 6 to 12 threads because the machine is 6 physical cores
   + HT (12 logical). The HT region (6 → 12) adds only ~1.5×. This loss cannot be recovered in
   software; the physical core count is the ceiling.

3. **The benchmark's "4 threads" is a comparison-fairness cap, not a runtime limit.** Real-world
   use should use all cores: ~2.1× faster than the 4-thread benchmark number (81 s → 38 s) with
   **bit-identical output**.

4. **No free transcendental-specific single-thread speedup remains.** Transcendental-heavy trees
   are libm-bound and sit on the SoA ~1.26× dispatch-elimination floor (`docs/30`); the only
   lever that breaks it is vectorised transcendentals (SLEEF). Under goal A that **improves no
   result** while taking on a heavy Windows/Rtools dependency with a required serial fallback —
   pure #5 for zero correctness benefit — so the `docs/30` decision *not* to add SLEEF is
   reinforced, not reopened, by choosing goal A.

## Decision

Under PySR parity, the practical way to make transcendental (and all) problems finish sooner is
**"use up to your physical core count."** That path is already shipped, near-ideal in efficiency,
and provably result-preserving. Beyond it, the per-thread evaluator is compute/libm-bound and the
only remaining lever (SLEEF) is not justified for a speed-only goal. No code change is warranted;
the actionable guidance is operational (set `OMP_NUM_THREADS` to the physical core count, not the
4-thread benchmark cap).

## Reproduction

```
# Build at the shipped -O2 generic flag (not -O3 -march=native), Rtools toolchain:
cmake -S . -B build-scale -G "MSYS Makefiles" \
      -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-rtools.cmake \
      -DCMAKE_MAKE_PROGRAM=C:/rtools45/usr/bin/make.exe \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG" \
      -DRSYMBOLIC2_BUILD_TESTS=OFF
cmake --build build-scale --target bench_profile -j 8

# Fixed-window efficiency sweep (cpu/wall = speedup); rel_mass runs full 2800 gens:
for t in 1 2 4 6 8 12; do
  OMP_NUM_THREADS=$t ./build-scale/standalone/bench_profile.exe rel_mass 25 1
done
# Full-completion absolute wall at 12 threads:
OMP_NUM_THREADS=12 ./build-scale/standalone/bench_profile.exe rel_mass 120 1
```

See also: `docs/30` (throughput profile, SoA, SLEEF decision), `docs/35` (throughput is a dead
lever for *recovery*; this file is its *wall-clock* counterpart), `docs/36` (BFGS/Float32
measured and rejected).
