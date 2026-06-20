# 26. Feynman dev-set recovery: rsymbolic2 vs PySR, and the throughput root cause

Status: **measured reference point (1-seed, dev-scale), not a 30-run publication
result.** Same role as docs/15 for Nguyen: a fair head-to-head at the dev scale to
decide where effort belongs, not a final benchmark. Recovery thresholds and protocol
follow docs/04 and docs/19. All numbers below are a single seed per problem unless
stated; they are noisy and are used for direction, not for claims of superiority.

## 1. What was run

- **rsymbolic2**: `benchmarks/02_feynman_gate.R stage=1 runs=1` (HEAD 6798b89, the
  branch state *before* the init-parallelisation of §4). 25 dev equations, in-sample
  NMSE, per-problem 300 s timeout + fail-fast. Shipped defaults: `optimize_probability
  = 0.1`, `parsimony = 1e-3`, `adaptive_parsimony_scaling` on, SelfLM optimizer,
  `population_size = 500`, `n_populations = 4`, `generations = 500`, `max_nodes = 50`.
- **SR.jl (PySR engine)**: `benchmarks/05_feynman_pysr_comparison.jl seeds=1`, run as
  `julia -t 4`. PySR's *verified defaults* (read from `pysr/sr.py`, identical to
  docs/15): `niterations = 100`, `populations = 31`, `population_size = 27`,
  `ncycles_per_iteration = 380`, `tournament_selection_n = 15`, `parsimony = 0.0`,
  `maxsize = 30`, no `timeout_in_seconds`. PySR is only a thin wrapper over
  SymbolicRegression.jl, whose juliacall bridge is broken on this machine (docs/15), so
  the engine is driven directly. SR.jl version 1.11.0, Julia 1.12.6.

Both tools read **bit-identical** data (`benchmarks/data/feynman_*_train.csv`, exported
by `export_feynman_data.R`). The **operator set is the one shared problem input**, given
identically to both (CLAUDE.md): binary `[+, -, *, /, ^]`, unary `[exp, log, sin, cos,
sqrt, tanh, abs, square]` (`neg` omitted — expressible as `0 - x`; `pow` → `^`; `square`
→ SR.jl's built-in). The only equalisation is the **thread budget**, applied on
rsymbolic2's side (`n_populations = 4` ↔ `julia -t 4`), never by altering PySR.

Recovery criterion is identical to `benchmarks/utils.R`: `NMSE = SSE / ((n-1)·var(y))`,
recovered iff `NMSE < 1e-4`.

### Caveats (do not read past these)

- **1 seed only.** Several problems sit within a small factor of the threshold and
  would flip with a different seed or more compute; treat the totals as ±a few.
- **`maxsize` asymmetry, by design.** PySR default `maxsize = 30`; rsymbolic2
  `max_nodes = 50`. Each tool keeps its own complexity budget (defaults vs defaults);
  this is not equalised because it is an algorithm default, not the shared problem input.
- **`safe_pow` semantics.** SR.jl `^` and rsymbolic2 `safe_pow` differ for non-positive
  bases with non-integer exponents. The Feynman dev domains keep `^` bases positive, so
  this does not affect any result here; documented, not worked around (PySR unchanged).
- **Stopping criteria differ** (each tool at its own default early stop), as in docs/15.

## 2. Head-to-head (1 seed)

**rsymbolic2 12/25, SR.jl (PySR) 15/25.**

| key | rsym NMSE | rsym | SR NMSE | SR | note |
|-----|-----------|------|---------|----|------|
| gaussian       | 1.3e-26 | ✅ | 5.2e-09 | ✅ | both |
| rel_mass       | 2.9e-04 | ❌ TO | 8.6e-09 | ✅ | SR only (rsym timed out near threshold) |
| coulomb        | —       | ✅ | 4.3e-29 | ✅ | both |
| spring_pe      | —       | ✅ | 9.1e-30 | ✅ | both |
| lorentz_x      | 6.5e-04 | ❌ | 1.3e-04 | ❌ | neither (both close) |
| rel_mom        | 2.5e-03 | ❌ TO | 1.2e-07 | ✅ | SR only (rsym timed out) |
| harmonic_ke    | 1.2e-03 | ❌ TO | 1.4e-03 | ❌ | neither |
| interference   | 2.1e-02 | ❌ TO | 1.8e-03 | ❌ | neither |
| bohr_radius    | 3.9e-04 | ❌ TO | 2.7e-29 | ✅ | SR only (rsym timed out near threshold) |
| larmor_rad     | 1.6e-29 | ✅ | 8.9e-04 | ❌ | **rsym only** |
| planck         | 4.3e-02 | ❌ TO | 1.2e-02 | ❌ | neither |
| center_mass    | 4.2e-02 | ❌ TO | 1.4e-02 | ❌ | neither |
| torque         | —       | ✅ | 1.0e-29 | ✅ | both |
| larmor_freq    | —       | ✅ | 1.2e-29 | ✅ | both |
| doppler_rel    | 8.3e-08 | ✅ | 9.7e-17 | ✅ | both |
| einstein_smol  | —       | ✅ | 8.5e-30 | ✅ | both |
| driven_osc     | 1.4e-01 | ❌ TO | 6.9e-29 | ✅ | SR only |
| heat_conduct   | —       | ✅ | 1.0e-29 | ✅ | both |
| boltzmann_dist | 3.2e-02 | ❌ TO | 2.3e-03 | ❌ | neither |
| clausius_moss  | 9.8e-06 | ✅ TO | 3.1e-05 | ✅ | both |
| clausius_moss2 | 4.8e-07 | ✅ TO | 1.1e-04 | ❌ | **rsym only** |
| bohr_magneton  | —       | ✅ | 1.1e-29 | ✅ | both |
| bose_einstein  | 1.0e-03 | ❌ TO | 1.3e-04 | ❌ | neither |
| lens_eq        | 3.0e-03 | ❌ TO | 2.4e-29 | ✅ | SR only |
| newtons_grav   | 1.8e-02 | ❌ TO | 6.3e-04 | ❌ | neither |

"TO" = the rsymbolic2 run hit its 300 s budget. Source CSVs:
`benchmarks/results/feynman_gate_20260620.csv`,
`benchmarks/results/sr_comparison_feynman_20260620.csv`.

### Reading the result

- **Both recover 10; both fail 8.** The 8 that fail for *both* tools (lorentz_x,
  harmonic_ke, interference, planck, center_mass, boltzmann_dist, bose_einstein,
  newtons_grav) are the multi-variable rational / deceptive targets docs/23 flagged.
  PySR does not solve them either, so these are **intrinsically hard**, not an
  rsymbolic2-specific defect. The recovery gap should be read in that light.
- **rsym-only: 2** (larmor_rad, clausius_moss2). **SR-only: 5** (rel_mass, rel_mom,
  bohr_radius, driven_osc, lens_eq).
- **The net gap is concentrated in throughput-limited near-misses.** rel_mass, rel_mom
  and bohr_radius all **timed out** in rsymbolic2 at NMSE only a few× above threshold,
  while SR.jl solved each in < 100 s. This is the §3/§4 throughput story, now measured
  against the reference: rsymbolic2 runs out of its compute budget on problems the PySR
  engine finishes comfortably.
- `adaptive_parsimony_scaling` (commit 968e78b) did **not** move the hard problems at
  this seed — center_mass is still 4.2e-2, matching the docs/23 diagnosis that the
  parsimony rescale alone does not fix the deceptive low-variance targets.

## 3. Throughput investigation (measure-first; one hypothesis rejected)

The SR-only near-misses above are throughput-bound, so the question became: why does the
multi-island search leave compute on the table? The island health probe
(`benchmarks/diag_omp_check.R`) shows `n_populations=4` reaching only **cpu/wall ≈ 1.80**
— threads busy ~45% of the time. docs/23 §4 + docs/25 had already removed the per-fit
Eigen heap contention with SelfLM. The remaining loss had to be elsewhere.

Isolation probes (independent `std::thread`s, no OpenMP/migration — so any `cpu/wall < N`
is pure shared-resource contention):

| workload (4 threads) | cpu/wall | reading |
|----------------------|----------|---------|
| pure FP arithmetic (control) | 3.95 | hardware/scheduler healthy |
| residual/eval loop, reused stack (`bench_heap`) | 3.70 | eval scales |
| SelfLM fit (`bench_heap`) | 2.93 | fit scales acceptably |
| **non-fit child step** — mutate/crossover + sse + simplify + hof (`bench_evolve`, **new**) | **3.57** | **per-child path is NOT the bottleneck** |
| production island search (`diag_omp_check`) | 1.80 | the actual loss |

The hypothesis that the **per-child allocation path** (a fresh `Tree = std::vector<Node>`
per child, plus `initial_constants`, eval stack, `simplify`, `HallOfFame` copy) was
heap-lock-bound like fit() was — **rejected**: `bench_evolve` reproduces that exact
per-child sequence in independent threads and reaches cpu/wall 3.57, and removing
`simplify`/`hof` changes nothing. The non-fit path is memory/eval-bound, not lock-bound.

The real cause was found by varying the generation count (init-heavy vs evolution-heavy)
with `benchmarks/diag_init_split.R`: scaling **improved** as generations rose
(`gen=40` cpu/wall 1.86 → `gen=200` 2.27). That is the signature of a **fixed serial
cost** at the front of every run — **island initialisation ran in a serial loop**
(`run_evolution`, the island-build loop), outside the OpenMP region. With
`n_populations=4` it does 4× the `population_size` initial `fit()`s **serially**, adding
that wall to every multi-island run. `OMP_WAIT_POLICY=passive`/`GOMP_SPINCOUNT=0` made no
difference, confirming the inflated CPU is real init work, not barrier spin.

## 4. Fix: parallelise island initialisation

Island initialisation is the same share-nothing work as the evolution region — each
island touches only its own slot and reads the shared const dataset/options — so it is
wrapped in the same OpenMP parallel-for with the same team-size cap. Determinism is
unaffected: every island is seeded deterministically before its independent
initialisation, so the result is identical regardless of thread count or order.

Measured effect (center_mass, m=1000, after rebuild):

| metric | before | after |
|--------|--------|-------|
| `diag_omp_check` n_pop=4 cpu/wall | 1.80 | **2.40** (+33%) |
| `diag_init_split` gen=200 n_pop=4 wall | 28.8 s | **22.8 s** (−21%) |
| `diag_init_split` gen=200 cpu/wall | 2.27 | 2.66 |

All 16 standalone tests pass, including the thread-count determinism check
(`test_island_model`: 1-thread and 4-thread runs produce identical expressions).

### Residual loss and limits

The evolution region still reaches only cpu/wall ≈ 2.4 (not 4): each island consumes
~2.3× the single-island CPU when four run concurrently, the signature of the remaining
per-fit allocation traffic (Windows heap `CRITICAL_SECTION` spins under contention,
inflating CPU). SelfLM removed the Eigen share; the residue is `initial_constants` /
`make_least_squares_problem` closure captures per fit. Reducing it further is the next
lever but has diminishing returns.

**Honest bound on impact.** Throughput gains here are percent-level. The
throughput-limited near-misses are 3–25× above the recovery threshold; a few percent
more generations will not flip them on its own. Lifting those requires either an
algorithmic search improvement or materially more compute, not just better scaling.

### Instrumentation note: `bench_parallel` is not measuring thread scaling

`benchmarks/bench_parallel` sweeps `omp_set_num_threads({1,2,3,4,6})`, but
`run_evolution` hard-caps the team with `num_threads(min(n, omp_get_num_procs()))`
(the timeout fix in commit 2e1a026). `omp_get_num_procs()` is the hardware processor
count and ignores `omp_set_num_threads`, so every row in `bench_parallel` actually runs
the same team size — its efficiency column is an artifact. The valid multi-island probe
is `diag_omp_check.R` (vary `n_populations`, fixed per-island work). A caveat is recorded
in `bench_parallel.cpp`; a proper rewrite (scale `n_populations`, report medians) is a
separate task.

## 5. Reproduction

```
# rsymbolic2 1-seed dev gate
Rscript benchmarks/02_feynman_gate.R stage=1 runs=1

# SR.jl (PySR engine) 1-seed, 4 threads
julia -t 4 benchmarks/05_feynman_pysr_comparison.jl seeds=1 > benchmarks/results/_feynman_sr.log 2>&1

# throughput probes (standalone build)
./standalone/bench_heap 4 400         # fit / eval isolation
./standalone/bench_evolve 4 4000      # non-fit child-step isolation
Rscript benchmarks/diag_omp_check.R   # production island cpu/wall
Rscript benchmarks/diag_init_split.R  # init-vs-evolution split
```
