# 42. `warmup_maxsize_by` — the last PySR mechanism gap closed

> **Attribution.** PySR and SymbolicRegression.jl are Apache-2.0, (C) Miles Cranmer.
> rsymbolic2 is an independent re-implementation whose defaults and mechanisms are
> matched to theirs; see the repository `NOTICE`. Not affiliated.

Before this change, `warmup_maxsize_by` was the single *true* mechanism gap in the
PySR difference catalog (`docs/29` ★F / B#5): the default value (`0.0`, off) matched,
so default behaviour was identical, but rsymbolic2 had **no ramp mechanism at all** —
setting the parameter could not change anything because the size cap was fixed at
`max_nodes`. This document records the implementation that closes it.

## A. What PySR does (authoritative)

PySR exposes `warmup_maxsize_by` (sr.py, default `0.0`); SymbolicRegression.jl 1.11.0
drives it through `get_cur_maxsize` (`SearchUtils.jl`):

```julia
function get_cur_maxsize(; options, total_cycles, cycles_remaining)
    cycles_elapsed   = total_cycles - cycles_remaining
    fraction_elapsed = cycles_elapsed / total_cycles
    in_warmup_period = fraction_elapsed <= options.warmup_maxsize_by
    if options.warmup_maxsize_by > 0 && in_warmup_period
        return 3 + floor(Int, (options.maxsize - 3) * fraction_elapsed / options.warmup_maxsize_by)
    else
        return options.maxsize
    end
end
```

So the **size cap used during evolution** (`curmaxsize`) starts at **3** and grows
linearly to `maxsize` over the first `warmup_maxsize_by` fraction of the run, then
stays at `maxsize`. (The `Options.jl` docstring says "from 5"; the **code says 3** —
the code is authoritative, per CLAUDE.md "read defaults from the installed source".)

Where SR.jl consumes `curmaxsize` (all in the evolution inner loop):
1. `condition_mutation_weights!` — if `complexity >= curmaxsize`, zero the `add_node`
   and `insert_node` weights (don't grow past the cap).
2. `check_constraints(tree, options, curmaxsize)` — reject a mutated tree over the cap.
3. `randomize_tree(tree, curmaxsize, …)` — draw the random size from `1:curmaxsize`.
4. crossover — `check_constraints` on both children against `curmaxsize`.

The frequency histogram (`AdaptiveParsimony`) is always sized by `maxsize`, and the
**initial `Population` is built before any cycle**, so it uses `maxsize`, not the
warmed-up size. Both are independent of `curmaxsize`.

## B. Mapping to rsymbolic2

### B1. A single per-epoch `cur_maxsize` (the `n` cancels)
SR.jl computes `fraction_elapsed = cycles_elapsed / total_cycles` with
`total_cycles = niterations * populations`, decrementing a shared counter as each
population finishes an iteration. rsymbolic2 evolves all islands in **lockstep** (every
island runs `interval` generations per epoch), so after `epoch` epochs every one of the
`n` islands has completed `epoch` iterations:

```
fraction_elapsed = (epoch * n) / (niterations * n) = epoch / niterations = done / generations
```

The population count cancels, so a **single** `fraction_elapsed = done / generations`
computed once per epoch is the faithful synchronous analog of SR.jl's per-population
fraction. It is computed at the **epoch start** (using `done`), so the first epoch sees
`fraction = 0 → cur_maxsize = 3` exactly as SR.jl's first iteration does. The function
is pure and unit-tested:

```cpp
int get_cur_maxsize(int max_nodes, double fraction_elapsed, double warmup_maxsize_by);
```

It clamps the result to `[1, max_nodes]` so a `max_nodes < 3` configuration cannot
exceed the absolute cap (SR.jl's start value of 3 would otherwise overshoot).

### B2. One override point: a local `SearchSpace`
Inside `evolve_island`, `opts.space` is read in exactly two places — the `mutate()` call
and the `subtree_crossover_pair()` size limit. rsymbolic2 builds a per-epoch local copy
`cur_space = opts.space; cur_space.max_nodes = cur_maxsize;` and uses it at both. This is
sufficient and faithful because rsymbolic2's `mutate()` already uses a
**feasibility-filtered menu**: `room = max_nodes - size` decides `can_add`/`can_insert`,
and an infeasible kind is dropped from the weighted menu (its weight redistributed) —
which is exactly SR.jl's `condition_mutation_weights!` zeroing of `add_node`/`insert_node`
at `complexity >= curmaxsize`. So overriding `max_nodes` simultaneously and correctly
covers all four SR.jl consumption points: growth conditioning (1), the post-mutation size
check (2, via the same `room` guard inside the structural mutations), the `randomize_tree`
size draw (3), and the crossover size cap (4).

**Deliberately left at full `max_nodes`** (matching SR.jl): the frequency histogram
(`isl.stats`, initialised once at island setup) and the initial population
(`random_tree.cpp`, generated before the epoch loop). Neither is inside `evolve_island`.

### B3. Default-off is byte-for-byte unchanged
At `warmup_maxsize_by = 0.0` (PySR default), `get_cur_maxsize` returns `max_nodes`
every epoch, so `cur_space == opts.space` and the evolution path is identical to the
pre-warmup code. Verified: the standalone `warmup_maxsize` test and the R/Python option
tests assert that a `warmup_maxsize_by = 0` run produces the same expression and loss as
leaving the field unset, and the island-model thread-determinism test is unchanged.

## C. Surface and validation

| Surface | Parameter | Default | Validation |
|---|---|---|---|
| C++ `SearchOptions` | `warmup_maxsize_by` (double) | `0.0` | clamp in `get_cur_maxsize` |
| R `symbolic_regression()` | `warmup_maxsize_by` | `0.0` | single finite number `>= 0` |
| Python `symbolic_regression()` | `warmup_maxsize_by` | `0.0` | finite, `>= 0` |

The numeric/Float32-vs-double difference in `fraction_elapsed` is an allowed
implementation difference (CLAUDE.md); the `floor` makes the cap integer either way.

## D. Tests

- **standalone `test_warmup_maxsize`**: `get_cur_maxsize` formula across the ramp /
  plateau / off / small-maxsize clamp (Gate 1), `warmup_maxsize_by = 0` reproduces the
  fixed-maxsize result (Gate 2), warmup-on run is valid + deterministic + within
  `max_nodes` (Gate 3).
- **R `test-new-options.R`**: default-off equivalence, warmup recovery + determinism +
  complexity bound, negative/`NA` rejection.
- **Python `test_rsymbolic2.py`**: default-off equivalence + warmup recovery/determinism,
  negative rejection, default surfaced in the signature.

## E. Status

Implemented and verified on Windows (standalone tests pass; R testthat 60/60 with
`NOT_CRAN`; pytest 13/13). Ubuntu (WSL) verification at the milestone per CLAUDE.md.
With this, `docs/29` category C/★F has **no remaining true mechanism gap**; the other
catalog entries are PySR-default-OFF *features* (annealing, custom loss, constraints,
units, graph mode, …), not behaviours that diverge at the matched defaults.
