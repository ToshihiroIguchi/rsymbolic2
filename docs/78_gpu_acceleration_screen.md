# 78. GPU acceleration screen: NO-GO

**Date:** 2026-08-02
**Status:** **NO-GO — closed without implementation.** Screened analytically against
measurements already on disk (`docs/60` §7.1/§7.2/§7.6, `docs/36`, `docs/37`,
`docs/67`/`docs/68`). No code was written, no default changed, no dependency added.

## Purpose

Question raised by the user: *the engine is CPU-only today — would offloading to a GPU
speed it up, and would that work on Intel and AMD GPUs as well as NVIDIA?*

This is a Priority #5 (Performance) question, so CLAUDE.md's rule applies: pursue only
with measured evidence, never speculatively. It is also a Dependency Policy question
(the default answer is **no**, burden of proof on adding) and a Platform Constraints
question (Windows/Rtools/MinGW is a mandatory target, and Portability outranks speed).

This screen is a **desk analysis**, not a benchmark. That is deliberate and is the
cheapest correct decision procedure here: the ceiling of a GPU port is computable from
the phase mix and per-call costs `docs/60` already measured, and the ceiling comes out
at or below zero. Building a prototype to confirm a negative ceiling would be the
`docs/43`/`docs/44` mistake in reverse — those docs established *measure the ceiling
first, build second*, and this is the measure-first step.

**Evidence provenance, stated up front.** Everything in §1 and §2(a)-(b) and §6 is
measured **in this repository**. The GPU-side quantities in §2(b)-(c) (kernel
round-trip latency, consumer FP64 throughput ratios, transcendental precision paths)
are **vendor specifications and general literature, not measured on this hardware** —
they are treated as assumptions, and §5 records exactly which of them would have to be
wrong to reopen the decision.

## 1. What would actually be offloaded

`docs/60` §7.1, summed work-seconds over 31 islands, 4 threads, full 2800-generation
runs:

| phase | rel_mass | spring_pe |
|---|---:|---:|
| `evolve_sse` (forward pass) | **75.8 %** (2 403 971 calls, 69.5 us/call) | **76.4 %** (240 369 calls, 123.1 us/call) |
| `popopt_fit` (LM constant fitting) | 20.6 % (8 655 calls) | 21.2 % (1 019 calls) |
| `mutate_xover` / `tournament` / `simplify` / `hof_update` | 3.6 % | 2.4 % |

So there is exactly one candidate kernel: `sse_current` -> `evaluate_soa_residual`. The
remaining 21 % is the LM path, whose per-`fit()` linear algebra is a handful of
parameters against a blocked `JtJ` (`docs/65`) — dense problems of order 1-10 unknowns,
which is the canonical worst case for GPU offload and is not considered further.

**Work per dispatch.** Feynman training sets are `n_train = 1000` rows
(`benchmarks/export_feynman_data.R:31`), and the measured node count is 36.20 M visits
over 2 403 971 calls (`docs/60` §7.6, rel_mass seed 1) = **~15.1 nodes/tree**. One
`evolve_sse` call is therefore **~15 000 node x point operations**.

## 2. The four independent blockers

Each of these is sufficient on its own. They are listed in the order that they bind.

### (a) The loss is consumed synchronously by the next step — measured, in-repo

`r-package/rsymbolic2/src/evolutionary_search.cpp:930-995` implements SR.jl's
regularized-evolution cycle: tournament-select a parent, mutate, evaluate the child with
`score_sse`, reject on non-finite loss, otherwise replace the oldest member. **The next
tournament draws from the population this step just mutated**, so the loss must be back
on the host before the loop can continue. There is no pipelining opportunity and no
queue depth to hide latency behind: every dispatch pays a full blocking round trip.

Consequence for the arithmetic: current forward-pass work is 2 403 971 x 69.5 us =
**167 work-seconds** (consistent with 62.01 s wall x 3.61 cpu/wall = 224 cpu-seconds at
75.8 %). Against that, a blocking round trip of 20-50 us — a typical figure for Windows
OpenCL with a reduction and a host sync, **assumed, not measured here** — costs 48-120
work-seconds. **Synchronization alone consumes 29-72 % of the very budget the port
exists to eliminate, before the GPU performs a single arithmetic operation.**

At an optimistic 5 us CUDA round trip the overhead falls to ~12 work-seconds (7 %), so
this blocker alone is *not* decisive at the best-case end of the assumed range. (b) and
(c) are, independently.

### (b) The dispatch is far too small to fill a GPU

1000 points is the entire parallel width available per call. A mid-range discrete GPU
has thousands of lanes; occupancy is single-digit to ~20 %. The CPU comparison point is
not idle either — `docs/60` §7.2 measured the existing OpenMP island parallelism at
**90-95 % efficiency at the 4-thread benchmark cap, against a 97 % island-granularity
ceiling**. The GPU would be replacing a nearly saturated CPU path with a nearly empty
GPU one.

### (c) Float64 + transcendentals is the GPU's weakest axis, and is exactly the hot spot

`docs/60` §7.6 measured per-operator cost of the production SoA tile kernels relative to
`Mul`: cheap ops 0.93-1.08, `Div`/`Inv` 3.0-3.3, `Sqrt` 9.7, `Tanh` 43.9, `Log` 141.3,
`Exp`/`Sin`/`Cos` 225-242, **`Pow` 450.2**. The cost is dominated by transcendentals, and
`docs/60` §7.6 also showed the natural experiment: rel_mass seeds 1 and 3 have
near-identical node visits (36.20 M vs 36.06 M) but seed 3 carries 1.84x the
transcendental density and took **2.01x the wall** (59.2 s vs 119.0 s).

Two vendor facts (assumed, not measured here) make this the wrong workload to move:

- **The fast transcendental hardware on GPUs is the FP32 special-function path.**
  IEEE double-precision `exp`/`log`/`sin` are software routines on GPUs, as on CPUs.
- **Consumer GPUs deliberately cripple FP64** — commonly 1/32 to 1/64 of FP32 on
  GeForce, weaker still or unavailable on Intel iGPU/Arc.

Dropping to Float32 to escape this is **not available**: `docs/36` measured Float32 at
**zero speedup on the Rtools/MinGW scalar evaluator and a 100-1000x worse loss floor**.
The precision the engine needs is precisely the precision consumer GPUs are worst at.

### (d) Batching across expressions is blocked by the algorithm, not by effort

The only structure in which a GPU wins is evaluating many expressions at once. Within an
island that is impossible by (a). Across islands one could in principle bundle the 31
concurrent children per step, but:

- 31 x 1000 = 31 000 points is still small for a GPU;
- it requires lock-stepping 31 islands that are today independent OpenMP work items,
  destroying the parallelism measured at 90-95 % efficiency in §7.2;
- it changes the concurrency structure the PySR-parity search trajectory is defined
  over, so the whole parity argument would have to be rebuilt.

Note also that **PySR / SymbolicRegression.jl ships no GPU backend**, so the parity rule
exerts no pull in this direction either.

## 3. The vendor question: Intel and AMD specifically

Answering the question as asked, independently of §2: a vendor-neutral port is
*possible*, but the only mechanism compatible with this project's mandatory toolchain
carries the highest maintenance cost of the options.

| mechanism | Intel | AMD | NVIDIA | verdict against Platform Constraints |
|---|---|---|---|---|
| CUDA | no | no | yes | **Disqualified on its own.** Windows `nvcc` requires MSVC `cl.exe`; R on Windows is built with **Rtools (MinGW/GCC + UCRT)**. CLAUDE.md: "fine on Windows via MSVC" does not mean usable from an R package |
| SYCL / oneAPI (DPC++) | yes | partial | partial | Needs its own clang or MSVC; not an Rtools/MinGW citizen |
| ROCm / HIP | no | yes | via HIP | Windows support for consumer AMD parts is thin; Linux-centric |
| Vulkan compute | yes | yes | yes | Workable, but SPIR-V toolchain + a graphics-API dependency for a numeric library |
| **OpenCL** | yes | yes | yes | **The only realistic route.** ICD loader can be `dlopen`ed, kernels compiled at runtime — no compile-time toolchain coupling, so MinGW-safe |

Even taking the OpenCL route, the standing cost is: a **second, permanently maintained
implementation of the hottest and most correctness-critical kernel**; a full CPU
fallback for every machine without a usable device (mandatory anyway — CLAUDE.md
requires parallel code to be correct when the accelerator is absent); runtime device
detection and dispatch in R, Python and WASM bindings; and a 3-vendor x 2-OS test
matrix. CLAUDE.md rates a dependency that raises Windows build or maintenance cost as a
**major architectural penalty**, and ranks Portability (#3) above Performance (#5).
This cost would be acceptable in exchange for a large measured win. It is not
acceptable in exchange for the ceiling computed in §2.

**Fallback if the dependency became unavailable** (required by Dependency Policy before
adoption): the CPU path, i.e. the status quo — which is itself an admission that the
feature carries no structural benefit.

## 4. Parity and numerics

- GPU `libm` differs in ULP, so a GPU arm is **not bit-identical**. That fails the hard
  gate every lever in `docs/60` was held to (§2 bar 1: 25/25 gate expressions identical).
  The WASM precedent (`docs/51`) shows a non-bit-identical backend *can* be accepted on
  quality parity — but that trade was paid for a capability (browser delivery) that
  nothing else provided. Here it would be paid for a speed win that §2 says does not
  exist.
- GPU fast-math defaults (`--use_fast_math`, `-cl-fast-relaxed-math`) are **unsafe here
  independently of determinism**, for the reason `docs/60` §6 rejected `-ffast-math`:
  the search uses IEEE NaN/Inf as control flow (`sse_current` returns `kInf` on
  non-finite, `clamp_finite` protects the normal equations, the HOF and `evolve_island`
  reject non-finite losses, and `docs/69`/`docs/77` make NaN — not 0 — the defined
  out-of-domain result). Folding `isfinite` to true disables all of it.

## 5. The exception condition, and why it is also closed

GPU offload becomes structurally sensible when a single expression evaluation is itself
large — order 1e5-1e6 rows, enough to fill the device and amortise the round trip. The
engine does support data at that scale, but **the large-data path deliberately shrinks
the evaluated point count**: PySR-parity batching (`docs/28` B5) evaluates evolution and
constant fitting over `batch_size` points (PySR default 50) and only the HOF /
early-stopping / finalize passes touch full data (`finalize_costs_and_merge`), and
`docs/59` established that for the web GUI the wall-clock wall arrives before the memory
wall. So on the path where a GPU could help, the default configuration has already
removed the work that would have justified it.

## 6. Decision

**NO-GO.** Not built, not prototyped, not scheduled. The ceiling is at or below zero
under measurements this project already owns, and the cheapest route to a vendor-neutral
implementation is simultaneously the most expensive to maintain against the mandatory
Rtools/MinGW + Ubuntu matrix.

**What would reopen this** (pre-registered, so it is not re-argued from intuition later):
all three of

1. a workload whose default configuration evaluates >= 1e5 points per forward pass —
   which today requires the batching default to change, i.e. a PySR-parity change;
2. evidence that the per-call round trip on the target stack is <= ~5 us **and** that
   FP64 transcendental throughput on the intended class of device beats a modern CPU
   core, both measured on this hardware rather than assumed from §2(b)-(c);
3. a batching structure that supplies many expressions per dispatch without altering the
   search trajectory — which §2(d) argues is blocked by the regularized-evolution
   dependency chain, and would have to be disproved rather than worked around.

## 7. What to do instead

Recorded because the screen turned up where the real headroom is, and it is not in the
hardware:

- **The measured win was a library swap, not a device.** `docs/67`/`docs/68` found the
  bottleneck was mingw-w64's own `libmingwex` transcendental implementations, not the
  CPU and not the compiler; redirecting to UCRT measured **4.4-6x**. The lesson
  generalises: the transcendental cost in §2(c) is a software-implementation cost, and
  software-implementation costs have historically been the tractable ones here.
- **One implementation lever remains unharvested:** `docs/60` Phase 3 (constant-subtree
  scalarisation in the SoA evaluator) — gate **PASSED but deferred by decision** (§7.6).
  Node-count weighting put it at 9.05 %, but the operator-cost table shows node count
  understates it, and it is bit-identical by construction. This outranks any GPU work.
- **Thread count is capped by island granularity** at `n_populations` = 31
  (`resolve_team_size`, `docs/37`), so more CPU parallelism is not the lever either.
