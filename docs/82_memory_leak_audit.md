# 82 — Memory-leak audit of the shipped code

Whether rsymbolic2 can leak memory, answered by reading the ownership model and then
proving it dynamically with AddressSanitizer / LeakSanitizer on the C++ core, the R
package, and the Python extension.

The short answer is **no leak was found anywhere**, and the interesting content of this
document is not that verdict but the evidence standard behind it: a clean sanitizer run
is worthless unless you first prove the sanitizer would have spoken up. Three of the runs
below were clean for the wrong reason before they were clean for the right one.

## 1. The static picture: there is nothing to leak

A search over the shipped C++ (`r-package/rsymbolic2/src`, `python/src`, `web/wasm`) for
`new`, `delete`, `malloc`, `free`, `calloc`, `realloc` and `unique_ptr::release()` returns
**zero real occurrences** — every hit is the English word "new" in a comment. Heap
ownership is therefore entirely in RAII types:

| Owner | Where | Note |
|---|---|---|
| `std::vector` | everywhere | `using Tree = std::vector<Node>` (`tree.hpp:21`) — the expression tree is a flat array, not a pointer graph, so the classic tree-teardown leak cannot be expressed |
| `std::unique_ptr` | `OptimizerFactory::create`, `PNode` in `mutation.cpp` | |
| `std::shared_ptr` | `Dataset`, the abort flag | one-directional: `Dataset` holds no back-reference, so no reference cycle is constructible |

The single pointer-linked structure is `PNode` in `mutation.cpp`, the temporary tree
`rotate_subtree` works on. It is `unique_ptr`-owned and torn down iteratively by
`destroy_pnode()` (`mutation.cpp:496`) — that iteration exists to stop a deep tree
overflowing the stack in the recursive destructor, not to prevent a leak.

Nothing grows without bound over a long run either: the hall of fame is one slot per
complexity, the opt-in `eval_cache` is direct-mapped with a fixed slot count
(`kEvalCacheSlots`), and the population size is fixed. The one allocation that lives for
the process lifetime is `egraph.cpp:407`, a function-local `static const std::vector<Rule>`
built once — still-reachable by construction, not a leak.

The binding layers add no allocation of their own: cpp11 and pybind11 handles are
refcounted RAII. `cpp11::stop()` throws rather than `longjmp`s out of C++ frames, which is
precisely the property the Rcpp→cpp11 migration (docs/41) bought.

## 2. The dynamic picture

Environment: WSL Ubuntu 24.04, g++ 13.3.0, 12 cores.
Flags: `-fsanitize=address -fno-omit-frame-pointer -g -O1`, `ASAN_OPTIONS=detect_leaks=1`.

| Layer | What was run | Leak report |
|---|---|---|
| C++ core | `ctest` 30/30, `OMP_NUM_THREADS=1` | none |
| C++ core | `ctest` 30/30, 12 threads | none |
| C++ core | `diag_search_digest` (real `run_evolution`: islands, migration, batching, strong-simplify) — 656 lines, matching the recorded Linux golden | none |
| **R** | `testthat` **FAIL 0 / WARN 0 / SKIP 0 / PASS 409** | **1 byte in 1 allocation** |
| **Python** | `pytest` **109 passed, 7 skipped** | see §2.2 |

### 2.1 R

The R number is the cleanest result in the audit because the baseline is so quiet:

| Process | Leaked |
|---|---|
| plain R, package not loaded | 1 byte / 1 allocation |
| R with `library(rsymbolic2)` | 1 byte / 1 allocation |
| R running the whole testthat suite | 1 byte / 1 allocation |

The same single byte in all three — R's own constant — and **zero** leak stacks naming
`rsymbolic2.so`. Loading the package costs nothing and running 409 tests through it costs
nothing.

### 2.2 Python: why the byte total is not the evidence

The Python process leaks ~966 KB in ~878 allocations at exit. That number is CPython,
numpy and pytest, and comparing it against a smaller baseline proves nothing because the
two processes import different amounts. The decisive measurement is instead **whether the
total responds to the number of searches**, everything else in the script held identical:

| `fit()` calls | Leaked |
|---|---|
| 0 | 966,632 bytes / 878 allocations |
| 2 | 966,632 bytes / 878 allocations |
| 8 | 966,632 bytes / 878 allocations |

Byte-for-byte identical. A leak inside the search leaks once per run and would rise
linearly; this does not move at all. Independently, **no leak record names
`rsymbolic2/_core`** in any run.

## 3. What nearly produced a false verdict

Five defects in the measurement, each of which would have yielded a confident and wrong
answer. They are recorded because the next person to run this will hit the same ones.

1. **A clean run with the detector disarmed.** The first positive control — a deliberate
   `malloc` never freed — did not fire, which looked like LeakSanitizer being unavailable
   under WSL. It was not: the probe was compiled at `-O1`, and **GCC deletes a `malloc`
   whose result is unused**. At `-O0`, writing through a `volatile` pointer, it fired
   immediately. Every subsequent run was gated on a positive control, and the control for
   R and Python leaks from C with the address never returned to the interpreter — an R
   numeric or Python int holding the pointer value makes the block *still reachable* and
   silences the report.
2. **`export LD_PRELOAD` put the whole toolchain under test.** Exporting it meant `gcc`,
   `cc1`, `tail` and `bash` were leak-checked too; the first "Python" report was in fact
   **the GCC compiler's** internal allocations. Preload per invocation, never export.
3. **Preloading libasan alone kills the process on the first C++ exception.** ASan
   initialises before libstdc++ is mapped, so its `__cxa_throw` interceptor never resolves
   the real symbol and the process dies with
   `CHECK failed: "((__interception::real___cxa_throw)) != (0)"`. The first pytest run
   stopped at test 34 — at the invalid-input tests, which throw by design (docs/80). Fix:
   preload `libstdc++.so.6` alongside `libasan.so`. This is an ASan/`LD_PRELOAD`
   limitation and says nothing about the extension.
4. **`grep _core.cpython` matched the wrong project.** It reported 59 leak frames "in our
   extension"; they were all in
   `scipy/optimize/_highspy/_core.cpython-312-x86_64-linux-gnu.so`, which happens to share
   the filename. Match on `rsymbolic2/_core`.
5. **An A/B where the B arm never ran.** The first fit-vs-no-fit comparison passed
   `niterations=`/`random_state=` (PySR's spellings, not ours — ours are `generations=`
   and `seed=`), so it raised `TypeError` and measured a traceback instead of a search.

## 4. Reproducing it

```bash
# --- C++ core -------------------------------------------------------------------
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS='-fsanitize=address -fno-omit-frame-pointer -g -O1' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address'
cmake --build build-asan -j "$(nproc)"
cd build-asan && ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure
```

For R and Python the sanitizer cannot be linked into the executable (the executable is the
interpreter), so it is preloaded — with the four rules from §3:

```bash
PRE="$(gcc -print-file-name=libasan.so) $(g++ -print-file-name=libstdc++.so.6)"
OPT="detect_leaks=1:abort_on_error=0:detect_odr_violation=0:log_path=$OUT/phase"

# R: flags via R_MAKEVARS_USER so no global ~/.R/Makevars is created; --no-test-load
# because the final load check spawns a plain R with no preload and always fails there;
# -l so the ordinary site-library install is untouched.
R_MAKEVARS_USER=Makevars.asan R CMD INSTALL --no-test-load -l "$LIB" r-package/rsymbolic2
LD_PRELOAD="$PRE" ASAN_OPTIONS="$OPT" R_LIBS_USER="$LIB" Rscript suite.R

# Python: rebuild the extension with the flags, and reinstall the normal build afterwards
# (the ASan build cannot be imported without the preload).
pip install ./python --force-reinstall --no-deps --no-build-isolation \
  --config-settings=cmake.args="-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer -g -O1;-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address"
LD_PRELOAD="$PRE" ASAN_OPTIONS="$OPT" python -m pytest python/tests -q
```

`log_path` matters: without it the sanitizer writes to stderr and buries pytest's and
testthat's own summaries, which is how the truncated first run went unnoticed.

## 5. What this does not cover, and the failure modes that remain

The audit covers leaks. It does not cover the memory-related failures that are actually
reachable in this codebase, both already documented:

- **`std::bad_alloc` inside an OpenMP region is fatal, not a leak.** `evolutionary_search.cpp`
  has no `try`/`catch`, and an exception cannot propagate out of an OpenMP structured
  block, so it reaches `std::terminate` (docs/74 hit exactly this with `std::length_error`).
  Memory exhaustion therefore appears as the R session dying, not as growth.
- **The WASM heap is fixed at 128 MB with growth disabled** (docs/51), so exhaustion there
  is a hard failure, and emscripten's allocator does not return freed memory to the OS —
  resident size alone reads like a leak when it is not (docs/79 measured 814 MB and
  concluded exactly that).
