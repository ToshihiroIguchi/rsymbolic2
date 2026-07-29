# Memory reduction: where the bytes actually go, and what to do about it

Status: plan + implementation record.
Scope: the shared C++ core and the three bindings. Nothing here changes a default, a
setting, or a search trajectory — every item is "implementation method may differ" under
CLAUDE.md's PySR Default Parity rule, and every item is required to be **bit-identical**.

## 1. Where the memory goes

Read off the code, not guessed. With `m` rows, `p` features, `I` = `n_populations`
(default 31), `T` = resolved OpenMP team size, `k` = the largest constant count any tree on
an island has been fitted with (`maxsize=30` allows up to ~15):

| source | bytes | site |
|---|---|---|
| caller's array (R SEXP / numpy / WASM `Xflat`) | `8mp` | binding |
| binding's row-major `X_cpp` | `8mp + ~56m` | `rsymbolic2_r.cpp`, `_py.cpp`, `_wasm.cpp` |
| `Dataset::X` (row-major copy) | `8mp + ~56m` | `least_squares_problem.hpp` |
| `Dataset::Xcol` (column-major) | `8mp` | same |
| `y` × 3 | `24m` | |
| LM `rbuf_` + `trial_rbuf_`, **per island** | `16m·I` | `self_lm_optimizer.hpp` |
| LM `jbuf_` (m×k Jacobian), **per island** | `8mk·I` | same |

The `~56m` terms are the per-row `std::vector` overhead (24-byte header + allocator
overhead) of the two row-major `vector<vector<double>>` copies — `m` separate heap blocks
each.

Everything else is small and stays small: populations + halls of fame across 31 islands
total ~1.3 MB, and the SoA evaluation pools are `O(tree.size() × 256)`, independent of `m`.

**The dominant term is not the data — it is the per-island LM working set.** At
`m=100,000`, `p=5`, `I=31`, `k=8`: data ≈ 30 MB, LM scratch ≈ 248 MB.

### Relation to docs/59

docs/59 fitted `peak ≈ n·(24p + 80 + 16·n_populations)` to ten OK/OOM points under WASM.
The `16·n_populations` term is exactly `rbuf_ + trial_rbuf_` (two doubles per row per
island), which is a good check on the model above.

That fit has **no `k` term**, and `jbuf_` supplies one: `std::vector::resize` never releases
capacity, so once an island fits a tree with `k` constants its Jacobian buffer stays at
`8mk` bytes for the rest of the run. `web/app/js/data.js: maxRowsForBrowser()` inverts the
docs/59 formula, so the shipped browser row ceiling does not represent this term. After the
work below the term is gone, which is the clean resolution — no re-fit of the ceiling
formula is needed to make it sound, only to make it less conservative.

## 2. What is being changed

### A — LM scratch becomes per-worker instead of per-island

`SelfLMOptimizer` holds ten `mutable` buffers, and there is one optimizer per island, so
the `O(m)` ones are allocated `I` times even though only `T` of them are ever in use at
once.

None of that storage carries information between fits: `run_lm_from` starts with
`params_ = x0` and every other buffer is written before it is read. Only `rng_` (restart
perturbations) is stateful, and it must stay per island for determinism.

So: hoist the buffers into an `OptimizerScratch` owned by the caller, one per OpenMP
worker, passed into `optimize()`. `rng_` and `config_` stay on the per-island optimizer.

- Cuts the island term by `I/T` — a factor of ~4 on an 8-core machine, 1 on a 32-core one.
- Bit-identical trivially: the same buffers, the same arithmetic; only the address changes.
- Determinism under `schedule(dynamic)` is unaffected precisely because the scratch is
  stateless — which worker picks up which island cannot be observed.
- `thread_local` is **not** used: it is unreliable for libgomp workers inside a loaded DLL
  on Windows/MinGW (see the `Island::sse_pool` comment and the memory note on that trap).
  The scratch is an explicit vector indexed by `omp_get_thread_num()`, the same
  "owner passes its scratch down" shape already used for the SoA pools.

### D — stop making three row-major copies of the data

`Dataset::X` (row-major) has exactly one reader in the whole codebase: `make_batch`, which
copies whole rows out of it, and which only runs when the opt-in `batching` is on.
Prediction goes through the frozen expression string (docs/48 D2), not through the dataset.
So `Dataset::X` can be deleted outright and `make_batch` can gather from `Xcol`.

The bindings then still build a row-major `vector<vector<double>>` only to have
`Dataset`'s constructor transpose it. Adding a column-major entry point removes that too:

```cpp
struct FeatureColumns { std::vector<std::vector<double>> columns; };  // columns[j][i]
SearchResult run_evolution(FeatureColumns X, std::vector<double> y, const SearchOptions&);
SearchResult run_evolution(const std::vector<std::vector<double>>& X_rows, ...);  // transposes
```

`FeatureColumns` is a distinct type on purpose: row-major and column-major are the same
C++ type, so only a wrapper makes the compiler catch a layout mistake. The row-major
overload stays for the ~25 standalone tests and the benchmark suite, which build small
matrices where a transpose is free and readability is worth more.

Net: 4 full copies → 2 (R/Python: the caller's array plus `Xcol`; WASM: `Xcol` alone, since
`Xflat` is already released before the run). The `2 × 56m` per-row allocation overhead goes
to zero — `p` column allocations instead of `2m` row allocations. R matrices are already
column-major, so the R binding's transpose loop disappears entirely.

### B — accumulate JᵀJ and Jᵀr in row tiles, and stop materialising the m×k Jacobian

The LM never needs the Jacobian as a matrix. It needs `A = JᵀJ` (k×k) and `g = Jᵀr` (k),
both reductions over rows. The current code (`self_lm_optimizer.cpp`) builds the whole
`m×k` buffer and then reduces it:

```cpp
for (int a = 0; a < k; ++a) {
    double ga = 0.0;
    for (std::size_t i = 0; i < m; ++i) ga += jbuf_[i*ku + a] * rbuf_[i];
    ...
    for (int b = a; b < k; ++b) {
        double s = 0.0;
        for (std::size_t i = 0; i < m; ++i) s += jbuf_[i*ku + a] * jbuf_[i*ku + b];
```

**Every accumulator is a single scalar summed over `i` in strictly increasing order.** That
is the property that makes tiling bit-identical: keep `ata_[a][b]` and `g_[a]` as running
accumulators across tiles and each one still receives exactly the same products in exactly
the same order, so the floating-point sequence is unchanged. The tiles interleave
*different* accumulators, which is irrelevant — they are independent.

This requires the Jacobian closure to fill a row range rather than the whole matrix:

```cpp
using JacobianFunction = std::function<void(const std::vector<double>& params,
                                            std::size_t row_lo, std::size_t rows,
                                            std::vector<double>& jac_block)>;
```

`jbuf_` drops from `m·k` to `rows_per_block·k` (256×k, ~16 KB). The block size is carried on
`OptimizationProblem` so a test can vary it; `make_least_squares_problem` sets it to
`kStride`, which is also the existing stop-poll granularity, so the poll cadence is
preserved.

Inside `make_least_squares_problem` the closure's two loops swap order (`for block: for
tile:` becomes `for tile: for block:`). Each `(point, lane)` entry is computed independently
by `evaluate_soa_jacobian`, so no value depends on that order.

Accounting is held fixed on purpose: `njev` is still incremented **once per LM iteration**
(not once per tile), and `nfev` is untouched, so `n_lm_jac_evals`, `n_lm_resid_evals` and
the `max_evals` budget are unchanged.

Expected side effect: **faster**, not slower. Today `jbuf_` is 6.4 MB at `m=100k, k=8` and
is streamed `k(k+1)/2 + k = 44` times per LM iteration — ~280 MB of memory traffic. A
16 KB tile stays in L1.

The finite-difference fallback (`problem.jacobian == nullptr`) still needs the full matrix
and keeps it. That path is unreachable from the search — the least-squares problem always
supplies an analytic Jacobian — and is exercised only by two tests.

### F — per-fit SoA pools: considered, not done

`make_least_squares_problem` allocates a fresh `Model` per `fit()`, whose Jacobian pool is
`tree.size() × (1+8) × 256 × 8` ≈ 553 KB, allocated and freed once per fit.

It was planned and then dropped. The pool is `O(1)` in the row count, so it is worth ~5 MB
of peak at the default thread count — not a memory result. What is left is a *speed*
hypothesis about allocator traffic, and there is no measurement behind it; docs/23 §4
already removed the dominant per-fit allocation, and docs/60 found the forward path is
libm-bound. Under CLAUDE.md's ordering that makes it speculative performance work, so it
stays out until something measures it.

## 3. What is deliberately not being done

- **Packing `Node` from 24 to 16 bytes.** Populations plus halls of fame total ~1.3 MB
  across 31 islands. It buys nothing measurable and costs readability in the hottest
  switch in the codebase.
- **Float32 storage.** Breaks bit-identity, and was already measured and rejected
  (docs/36).
- **Lowering `n_populations`.** It is a PySR-parity setting. Out of bounds.
- **Lowering `kStride`.** It would shrink the SoA pools, but it is the stop-poll cadence
  (docs/22) and the vectorisation tile width (docs/30). Not a memory knob.
- **Zero-copy views over the caller's buffer.** R matrices are already column-major, so
  pointing `Xcol` at the SEXP would remove the last copy. It trades a lifetime/ownership
  invariant and integer-matrix coercion for ~`8mp` bytes, on top of a change that has
  already removed the large terms. Held back deliberately.
- **Tiling the residual buffers too** (the remaining `16m·T`). Feasible — `eval_sse` is
  also a strictly-ordered scalar reduction, and the SoA Jacobian's value column is
  bit-identical to the residual path, so `r` could come from the Jacobian tile at no extra
  cost. But it moves `rbuf_`/`trial_rbuf_` and touches the evaluation-accounting contract
  (`n_lm_resid_evals` feeds `max_evals`). Deferred until the measurement below says the
  remaining term matters.

## 4. Verification protocol

1. **Bit-identity** is the load-bearing claim, so it is tested directly: fixed-seed searches
   are digested (expression, loss, complexity, whole Pareto front, all four evaluation
   counters) before and after, and must match exactly.
2. A new gate proves the tiling argument rather than assuming it: the same LM fit run with
   several Jacobian block sizes (1, 7, 256, m) must return bit-identical constants and SSE.
3. Full standalone suite, R `testthat`, `pytest`, WASM build + its parity gate.
4. Windows first, then Ubuntu (WSL) at the milestone, per CLAUDE.md's verification cadence.
5. Peak RSS measured before and after with `standalone/benchmarks/bench_memory.cpp`.

## 5. Measured results

`standalone/benchmarks/bench_memory.cpp`, process peak working set (Windows
`PeakWorkingSetSize`; Linux `VmHWM`). Windows 11, Rtools45 g++ -O2, 100,000 rows ×
5 features, `n_populations=31`, `target_loss=-inf` so the full budget always runs.

### The two premises, checked before changing anything

| probe | baseline | reading |
|---|---|---|
| 8 threads vs 2 threads, 28 generations | 168.2 vs 167.2 MiB | the working set does not depend on the thread count — it is per **island**, as claimed |
| 28 vs 112 generations, 8 threads | 168.2 vs 225.9 MiB | +58 MiB purely from running longer: the `8mk·I` Jacobian term growing as islands meet trees with more constants. This is the term docs/59's formula has no slot for |

### Before / after

| case | before | after | factor |
|---|---|---|---|
| 8 threads, 28 generations | 168.2 MiB | 44.3 MiB | 3.8× |
| 8 threads, 112 generations | 225.9 MiB | 46.7 MiB | 4.8× |
| 2 threads, 28 generations | 167.2 MiB | 34.3 MiB | 4.9× |
| 8 threads, 28 gen, **binding path** (column-major in) | 168.2 MiB | 30.5 MiB | **5.5×** |

The first three rows run the row-major convenience entry point, so they still pay for the
transpose — that is the honest like-for-like number against the baseline, which had no
other path. The last row is what R, Python and WASM actually do now.

Two things worth reading off the table:

- The 28-vs-112-generation gap collapses from 57.7 MiB to 2.4 MiB. The Jacobian term is
  gone, so peak memory no longer creeps upward the longer a search runs.
- Memory now responds to the thread count (44.3 → 34.3 MiB at 2 threads) because the LM
  working set is per worker. Before, it did not.

Second size, after only (no baseline pair): 200,000 rows × 10 features, binding path,
31 populations, 8 threads — **57.4 MiB**.

Linux (Ubuntu 24.04, g++ -O2, WSL2) agrees with Windows: 32.3 MiB binding path,
38.2 MiB row-major path.

### Speed

Not a goal, but the tiled reduction was expected to help cache locality and it does. Same
runs, wall clock:

| case | before | after |
|---|---|---|
| 8 threads, 28 generations | 4.85 s | 3.39 s |
| 8 threads, 112 generations | 15.48 s | 12.68 s |

Roughly 20-30% on this workload. Consistent with the prediction: the old code streamed a
6.4 MB Jacobian buffer 44 times per LM iteration; the block is 16 KB and stays in L1.

## 6. Verification performed

- **Bit-identity vs the pre-change commit**: `diag_search_digest` over 33 fixed-seed
  searches (3 seeds × 11 configurations: default, `n_populations` 1/4/8, `n_threads` 1/3,
  batching, eval_cache, linear_scaling, weights, warmup, and a multi-feature problem),
  digesting the expression, the loss and every Pareto member's loss as `%a` hex floats,
  the complexities, and all four evaluation counters. **571 lines, byte-for-byte
  identical.**
- **Jacobian block-size invariance** (`test_self_lm_optimizer`): the same fit at block
  sizes 0/1/7/64/256/257/4096 over 257 rows returns bit-identical constants, loss, `nfev`
  and `njev`. This is the permanent gate on the accumulation-order argument in §2 B — if
  anyone converts the reduction to per-block partial sums, it fails.
- Standalone suite: 29/29 on Windows, 29/29 on Ubuntu.
- R `testthat`: 294 tests, 0 failures on both platforms.
- `pytest`: 65 passed (Windows), 56 passed / 9 skipped (Ubuntu).
- WASM: builds under the pinned emsdk; `web/wasm/test/parity_test.cjs` passes.

### Incidental finding: `expression_simplified` is not reproducible — **fixed, docs/66**

The first digest diff flagged two `expression_simplified` strings. Re-running the **same
unmodified binary** twice reproduced the difference, so it is pre-existing and unrelated:
`display_simplify()`'s e-graph runs under a 10 ms wall-clock budget (docs/54), which makes
its output depend on machine timing.

This affects only the display string. `expression` is the frozen round-trip source
(docs/48 D2), the search never reads the simplified form, and `predict()` does not use it —
so nothing computational is at risk. But a user re-running a fixed seed can see a different
"simplified" rendering of the identical model. Worth fixing separately, by budgeting the
e-graph in iterations/nodes rather than milliseconds.

**Done in docs/66**, by exactly that route: `max_millis` was deleted from `EGraphLimits`,
and the e-node cap tightened to bound the cost it had been hiding. `diag_search_digest` no
longer excludes the field — it digests it, and gained a `strong_simplify` arm.

## 7. Left undone, deliberately

- **The browser row ceiling is now conservative.** `web/app/js/data.js:
  maxRowsForBrowser()` inverts docs/59's fitted formula, which no longer describes this
  build — the `16·n_populations` term is now `16·n_threads`, the unmodelled `8mk·I` term is
  gone, and the input is held once instead of three times. Raising the ceiling needs its
  own WASM OOM sweep, because over-estimating it aborts the module rather than degrading
  (docs/59 §3). Not attempted here; the current limit is safe, just pessimistic.

  **Resolved in docs/66 §6 — the ceiling stays where it is, as a decision rather than a
  pending task.** The browser's binding constraint is time, not memory, and the present
  ceiling already sits at that wall; raising it would only permit runs nobody waits for.
- **The residual buffers** (`16m` per worker) are the largest remaining `O(m)` term. See
  §3 for why tiling them was deferred.
