# Making `expression_simplified` reproducible: budget the e-graph in counts, not milliseconds

Status: implementation record.
Scope: the display-only simplifier (`display_simplify` / `egraph`) and the three bindings.
The search is **bit-identical** before and after — proven, not asserted (§5).

Follows up the loose end recorded in `docs/65` §6 and §7.

## 1. The defect

`docs/65` §6 found, while digesting fixed-seed searches for the memory work, that
`expression_simplified` differed between two runs of the **same unmodified binary**. The
field was excluded from that digest so it would not manufacture false differences, with a
note that it should be fixed separately.

The cause is a single field. `EGraphLimits` bounded Layer-2 equality saturation with three
caps — iterations, e-nodes, and `max_millis = 10.0`. The first two are functions of the
input tree; the third is a reading of `steady_clock`, consulted in three places in
`egraph_simplify` (the iteration head, the rule-matching loop, and the write loop). When it
binds, *where* saturation stops depends on how busy the machine was, so the same tree
renders differently on two runs.

The header called the wall-clock cap "a safety net that is not expected to bind at these
sizes". That expectation was wrong, and nothing measured it. It binds routinely: at the
default `maxsize=30` roughly 1-2% of random trees exceed 10 ms, and the whole per-call
distribution was visibly clipped at the cap.

### Blast radius: display only, and the opt-in search path was already safe

The first thing checked was whether opt-in `strong_simplify` (docs/55), which runs the same
simplifier **inside** the evolution loop, was also affected — that would have made this a
reproducibility bug in a search-affecting feature rather than a display one.

It was not. `kSearchStrongSimplifyLimits` was `{4, 1000, 1.0e9}`: an 11.6-day sentinel
deliberately neutralising the wall-clock stop, with a comment saying exactly why ("a
wall-clock stop would make the search non-reproducible"). So the search-side budget was
already counts-only. The defect was confined to the display default.

`expression_simplified` is display-only: `expression` is the frozen `predict()` round-trip
source (docs/48 D2), the search never reads the simplified form. So nothing computational
was ever at risk — but a user re-running a fixed seed could see a different rendering of an
identical model, which is a reproducibility hole regardless.

## 2. The fix

`max_millis` is **deleted from `EGraphLimits`**, not defaulted to infinity. Leaving the
field would leave a non-deterministic stop condition reachable through the API and force
the search site to keep its unnatural sentinel. With the field gone, both remaining caps
are counts, `egraph_simplify` no longer includes `<chrono>`, and the search budget reads
plainly as `{4, 1000}`.

The header now carries the rule rather than the expectation: do not reintroduce a clock
reading, a deadline, or any other stop a second run could evaluate differently.

## 3. Which counts, and why 2000

Removing the time cap means the counts alone must bound the cost, so they were measured
rather than assumed. `standalone/benchmarks/bench_simplify.cpp` (new) generates random
trees of an exact node count over the full operator set, times `display_simplify` per call,
and reports the distribution plus Layer-2 adoption rate and mean output size.

Quality is measured as **Layer-2 adoption**, not "did the tree shrink": Layer 1 alone
shrinks 97-100% of random trees, so a raw shrink rate cannot discriminate between budgets.

2000 trees per cell, Windows 11, Rtools45 g++ -O2. Per-call milliseconds:

| caps `{iters, enodes}` | 30 nodes max | 60 nodes max | 120 nodes max | adopt @30 | adopt @60 | adopt @120 |
|---|---|---|---|---|---|---|
| `{10, 10000}` (old) | 45.3 | **615.5** | 73.0 | 8.6% | 14.6% | 28.2% |
| **`{10, 2000}` (new)** | **9.5** | **18.4** | **22.9** | **8.6%** | **14.6%** | 27.6% |
| `{6, 2000}` | 3.9 | 11.0 | 7.0 | 7.8% | 13.4% | 25.6% |
| `{4, 1000}` (search-side) | 0.9 | 2.6 | 3.0 | 6.3% | 10.9% | 22.3% |

Two things fall out of this table.

**The old e-node cap had a heavy tail the wall-clock net was hiding.** 615 ms on the worst
of 2000 60-node trees. E-matching is superlinear in the class count, and 10000 e-nodes is
far past where that bites. Since display simplification runs once per Pareto front member
at finalisation, the ceiling that matters is (front size) × this — up to ~19 s at
`maxsize=60`, on the WASM build's single UI thread. Removing the time cap without
tightening the counts would have converted a reproducibility bug into a hang.

**The two caps are not interchangeable.** Tightening `max_enodes` to 2000 bounds the tail
27× at a cost of 0.6 percentage points of adoption at 120 nodes and *nothing* at 30 or 60.
Tightening `max_iterations` to 6 instead buys less and costs ~2.6 points. The tail is
driven by matching over the class count, not by the iteration count, so `max_enodes` is the
correct lever. `{10, 2000}` therefore dominates `{6, 2000}` outright, and was chosen.

The search-side `{4, 1000}` is left alone: it runs per population member inside the
evolution loop rather than once at finalisation, so it is budgeted for a different call
frequency (docs/55).

### Trade-off taken, stated plainly

Tightening `max_enodes` is not free. Decomposed on the 36-case search digest (§5), by
operator-node count of the rendered strings:

| change | strings changed | operator nodes | direction |
|---|---|---|---|
| drop `max_millis`, keep `{10, 10000}` | 3 of 36 | 199 → 198 | 1 smaller, 0 larger |
| also `max_enodes` 10000 → 2000 | 6 of 36 | 199 → 201 | 0 smaller, 3 larger |

Dropping the time cap can only improve the rendering — more saturation means the extraction
minimises over a superset — and it does. The e-node tightening costs **+3 operator nodes
out of 198 (1.5%), on 3 of 36 cases**, in exchange for the 27× tail bound. Taken
deliberately: a display string one node longer is a far smaller defect than a browser tab
frozen for seconds, and CLAUDE.md ranks correctness above the cosmetics of the display
layer.

## 4. Reproducing the defect

Back-to-back re-simplification on an idle machine shows **zero** mismatches even pre-fix —
timing is too stable for the cap to land differently. The defect only surfaces under
contention, which is exactly why docs/65 hit it during a long digest run and why a naive
gate would have missed it.

`bench_simplify` re-simplifies every tree and compares renderings. Twelve concurrent
processes, 1500 trees × 120 nodes each:

| | mismatches per process |
|---|---|
| before | 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 7 |
| after | 0 × 12 |

## 5. Verification

- **Search bit-identity vs the parent commit** (`c29d0f8`): `diag_search_digest`, 36
  fixed-seed searches, every float as `%a`. The digest binary was rebuilt from the parent
  commit's core with the *current* digest source so the two runs are comparable. Excluding
  the display-only `simplified` line, the 656-line reports are **byte-for-byte identical**.
- The digest gained a `strong_simplify = true` arm, because that is the only option that
  runs the e-graph inside the evolution loop and therefore the only trajectory a change to
  `EGraphLimits` could move. It is identical too.
- `expression_simplified` is **restored to the digest**. It was excluded only because of
  this bug; it is now a pure function of the tree and belongs in the golden. Six concurrent
  digest runs produce one distinct hash.
- New permanent gate in `test_display_simplify.cpp`: `test_rendering_is_reproducible`
  re-simplifies 40 trees at each of 30/60/120 nodes four times and requires identical
  renderings. Sized deliberately at and beyond the default `maxsize` — cheap trees never
  approach a cap and would pass under the old code too.
- New user-facing gates: `test-display-simplify.R` and `test_display_simplify.py` each run
  the same fixed-seed search twice and assert every simplified field matches.
- Standalone suite 29/29; R `testthat` 327 passed / 0 failed; `pytest` 66 passed; WASM
  builds under the pinned emsdk and `parity_test.cjs` passes. Windows and Ubuntu (WSL).

## 6. The browser row ceiling: closed, not deferred

`docs/65` §7 left `web/app/js/data.js: maxRowsForBrowser()` as an open item — its fitted
formula no longer describes the engine, so the shipped ceiling is conservative. Examined
here and **deliberately left at its current value**; it is not a pending task.

Reading the current code, the drift is in the *shape*, not just the size:

| model term (docs/59) | today |
|---|---|
| `24p + 80` — three row-major copies | one column-major copy, moved not copied; the `+80` per-row overhead is gone |
| `16·n_populations` — per-island LM scratch | per-**worker**, and this build is single-threaded: `16·1`. No physical dependence on `n_populations` remains |
| — | the dominant term is now unmodelled: the intake transpose holds `Xflat` and the column copy at once (`~16pn`), peaking *before* the search starts |

That puts true capacity roughly 5-19x above what the function returns (~80 B/row at `p=5`
against 696). **It is still not raised**, because docs/59 §1 already settled the question:
the binding constraint on browser row count is time, not memory, and the current ceiling
sits at that wall. A default-budget run at the present `p=5` limit (~96,000 rows) takes
~5.7 min batched and ~2 h unbatched; the modelled ceiling of ~839,000 rows corresponds to
~48 min batched. Raising it would let users start runs nobody waits for, while costing a
fresh WASM OOM sweep — over-estimating aborts the module rather than degrading (docs/59
§3). Negative benefit at real risk, so the work done here is limited to correcting the
stale explanation at the call site and in docs/59.

Accepted consequence: the now-spurious `n_populations` term still shrinks the ceiling when
the user raises the population count, forcing sampling the engine does not require. It
costs rows in the sample, never a failed run, and removing it would mean raising a ceiling
— which needs the sweep this section declines.

## 7. Left undone

- **The residual buffers** (`docs/65` §3) are still the largest remaining `O(m)` term.
