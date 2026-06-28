# 26. Feynman dev-set recovery: rsymbolic2 vs PySR, and the throughput root cause

Status: **measured reference point, dev-scale, not a 30-run publication result.**
Same role as docs/15 for Nguyen: a fair head-to-head at the dev scale to decide where
effort belongs, not a final benchmark. Recovery thresholds and protocol follow docs/04
and docs/19. §2–§4 are the original **1-seed** exploration (noisy, direction only).
**§5 is the authoritative multi-seed confirmation (rsymbolic2 5 seeds, SR.jl 3 seeds)
and supersedes §2's 1-seed totals; read it first for the head-to-head conclusion.**

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
  `maxsize = 30`, no `timeout_in_seconds`, `annealing = false` (`sr.py:981`),
  `precision = 32` i.e. Float32 data (`sr.py:1030,:1870`). PySR is only a thin wrapper
  over SymbolicRegression.jl, whose juliacall bridge is broken on this machine (docs/15),
  so the engine is driven directly. SR.jl version 1.11.0, Julia 1.12.6.
  **Faithfulness note:** the §2 1-seed and the original §5 multi-seed runs set neither
  `annealing` nor `precision`, so they used SR.jl's *Options* defaults (`annealing = true`,
  Float64 data) — which differ from PySR's documented defaults on exactly these two
  fields (full `Options` field dump compared 1:1 against `sr.py`; the only two
  divergences). Both inflate SR.jl in our favour (annealing aids escape from local
  optima; Float64 gives higher-precision constant fitting). §5 was therefore **re-run on
  2026-06-21 with both set to PySR's documented values** (this is faithfulness to PySR,
  not algorithm tuning — CLAUDE.md). The re-run result is below and supersedes the earlier
  non-faithful multi-seed numbers.

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

## 5. Multi-seed confirmation, faithful to PySR defaults (authoritative)

> **Update (2026-06-27, `docs/35`).** The "throughput-bound / 300 s TO" framing in this section
> is **superseded for the current code**. After the `docs/31` cadence fix nothing times out: every
> problem now completes the parity-fixed `generations = 2800` in 20–30 % of the budget, and
> rsymbolic2 reaches **18/25 = PySR 18/25** with *symmetric* membership (the strict-subset result
> below is gone) — the head-to-head now differs by a single 1-for-1 swap (rsymbolic2 wins
> `lorentz_x`, PySR wins `interference`). The far-misses are no longer throughput-bound; the
> residual gap is per-generation search quality (category-A). See `docs/35`.

Re-run **2026-06-21** with SR.jl made faithful to PySR's documented defaults
(`annealing = false`, `precision = 32` / Float32 data — see the §1 faithfulness note;
NMSE is still computed in Float64 so the recovery criterion is bit-for-bit the utils.R /
gate formula, unchanged). This supersedes both §2's 1-seed and the original 2026-06-20
multi-seed run, which used SR.jl's non-PySR Options defaults. Asymmetric protocol by
design: rsymbolic2 carries the per-problem 300 s timeout + fail-fast, so it was run at
the full 5 seeds; SR.jl has no timeout default, so its harness was hardened to
**per-problem incremental save** (`05_feynman_pysr_comparison.jl` flushes each search's
row as it completes — a stall on a later problem cannot lose completed work; PySR
settings unchanged, harness-side only) and run at 3 seeds.

- **rsymbolic2: 12/25** (5 seeds, majority ≥3/5). `feynman_gate_20260620.csv`. Unchanged
  from §2's 1-seed and the prior multi-seed — **stable**, not a lucky draw. (The rsym
  side was not re-run; only SR.jl changed.)
- **SR.jl (PySR engine), faithful: 18/25** (3 seeds, majority ≥2/3).
  `sr_comparison_feynman_20260621.csv`.

The faithfulness fix **did not move the headline total** (still 18/25) but **reshuffled
the membership**: making SR.jl faithful cost it `harmonic_ke` and `boltzmann_dist` (it now
fails both) and gained it `interference` and `lens_eq`. So the concern that SR.jl's 18 was
an artifact of annealing/Float64 is **resolved — the gap is real at PySR's documented
defaults.** It remains **6 problems and one-directional.**

| set | problems | n |
|-----|----------|---|
| both recover | gaussian, coulomb, spring_pe, bohr_radius, torque, larmor_freq, doppler_rel, einstein_smol, heat_conduct, clausius_moss, clausius_moss2, bohr_magneton | 12 |
| **SR.jl only** | rel_mass, rel_mom, larmor_rad, driven_osc, interference, lens_eq | 6 |
| rsymbolic2 only | — | 0 |
| both fail | lorentz_x, harmonic_ke, planck, center_mass, boltzmann_dist, bose_einstein, newtons_grav | 7 |

**rsymbolic2's recovered set is a strict subset of SR.jl's** (rsym-only = 0), as before.
Note the faithful set is *different* from the 2026-06-20 set even though both number 18:
`harmonic_ke` (SR.jl now 1/3) and `boltzmann_dist` (now 0/3) drop into both-fail, while
`interference` (now 2/3) and `lens_eq` (now 3/3) become SR-only. This matters for
targeting (below): two of the three problems previously cited as far-miss evidence of an
rsymbolic2 search-efficiency deficit are, under faithful PySR, problems **PySR cannot
solve either** — so they are intrinsically hard, not rsym-specific gaps.

### Algorithmic vs compute: the gap is search efficiency, not budget

On **every** one of the 6 faithful SR-only problems rsymbolic2 spent its **full 300 s**
budget (median time 300 s, timed out) and still failed, while SR.jl solved each in
**29–84 s median wall** on the same 4 threads — a fraction of rsymbolic2's budget:

| SR-only problem | rsym rec | rsym med NMSE | rsym med time | SR.jl rec | SR.jl med NMSE | SR.jl med time |
|-----------------|----------|---------------|---------------|-----------|----------------|----------------|
| rel_mass     | 1/5 | 2.7e-4 | 300 s TO | 3/3 | 3.0e-8  | 29 s |
| rel_mom      | 1/5 | 2.5e-3 | 300 s TO | 3/3 | 2.9e-9  | 70 s |
| larmor_rad   | 2/5 | 2.2e-3 | 300 s TO | 2/3 | 1.9e-14 | 46 s |
| lens_eq      | 1/5 | 1.2e-3 | 300 s TO | 3/3 | 1.4e-14 | 42 s |
| interference | 0/5 | 1.6e-2 | 300 s TO | 2/3 | 9.4e-5  | 84 s |
| driven_osc   | 0/5 | 1.5e-1 | 300 s TO | 3/3 | 4.1e-14 | 34 s |

Two sub-patterns:

- **Near-misses (rel_mass, rel_mom, larmor_rad, lens_eq):** rsymbolic2 *does* recover them
  on 1–2 of 5 seeds — the structure is reachable, the search just rarely converges inside
  budget. Partly compute/throughput-bound.
- **Far-misses (interference, driven_osc):** rsymbolic2 is 0/5 with NMSE one-to-two orders
  of magnitude above threshold, yet faithful SR.jl finds them in < 85 s. The search never
  approaches the structure. This is **algorithmic** (search sample efficiency), not a
  budget shortfall. (Note: `harmonic_ke` and `boltzmann_dist`, far-misses in the
  non-faithful run, drop out here because **faithful SR.jl also fails them** — they are
  intrinsically hard, not rsym-specific. `interference` enters as a new far-miss; faithful
  SR.jl recovers it 2/3.)

This sharpens §4's honest bound. SR.jl reaches all six in well under rsymbolic2's own
time limit, so "more compute" cannot be the primary lever: the §4 throughput fix is
percent-level (necessary hygiene) and would at best lift the four near-misses. The two
far-misses — and the strict-subset, one-directional shape of the gap — point at
**search efficiency** as the dominant deficit.

**Direction (confirm scope before building): invest in the search algorithm** — the
mutation/crossover operator mix, adaptive parsimony, and exploration of the complexity
ladder — benchmarked against faithful SR.jl on the far-miss problems (`interference`,
`driven_osc`), **not** in raw compute or further parallel scaling. The strict-subset result
means rsymbolic2 is currently strictly behind the PySR engine on recovery at dev scale;
closing it is a search-quality problem.

> **Correction (2026-06-23).** An earlier version of this paragraph claimed rsymbolic2
> "lacks `rotate_tree` / `insert_node` / `swap_operands`". That was already stale when
> written: those mutations (and the full PySR-default weight set) shipped in commit
> `aaead60`. A subsequent **faithfulness audit** of the structural mutations against SR.jl
> `MutationFunctions.jl` 1.11.0 (`docs/29` §A) found and fixed six mechanism divergences
> from the reference (missing `prepend_random_op` half of `add_node`; `append_random_op`
> reusing vs discarding the grown leaf; a fixed 0.5 unary/binary split instead of
> `nbin/(nuna+nbin)`; `insert_random_op` wrap side; binary-only `rotate_tree` vs the
> unary-aware reference; and the `delete_random_op` node-selection / leaf-randomize
> branch). The operators existed; the open work was making their *behaviour* faithful.

## 6. Reproduction

```
# rsymbolic2 dev gate — 1 seed (sanity) and 5 seeds (authoritative, §5)
Rscript benchmarks/02_feynman_gate.R stage=1 runs=1
Rscript benchmarks/02_feynman_gate.R stage=1 runs=5

# SR.jl (PySR engine), 4 threads — faithful to PySR defaults (annealing=false,
# precision=32). The harness now sets both, so seeds=3 reproduces the authoritative §5.
julia -t 4 benchmarks/05_feynman_pysr_comparison.jl seeds=1 > benchmarks/results/_feynman_sr.log 2>&1
julia -t 4 benchmarks/05_feynman_pysr_comparison.jl seeds=3 > benchmarks/results/_feynman_sr.log 2>&1

# throughput probes (standalone build)
./standalone/bench_heap 4 400         # fit / eval isolation
./standalone/bench_evolve 4 4000      # non-fit child-step isolation
Rscript benchmarks/diag_omp_check.R   # production island cpu/wall
Rscript benchmarks/diag_init_split.R  # init-vs-evolution split
```
