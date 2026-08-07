# 86. The core source list was written out four times

**Date:** 2026-08-06
**Status:** implemented; verified on Windows and Ubuntu.
**Change:** `cmake/CoreSources.cmake` (new), `standalone/CMakeLists.txt`,
`python/CMakeLists.txt`, `web/wasm/CMakeLists.txt`, `.github/workflows/deploy-pages.yml`.
**Follows:** `docs/85` §6, which recorded this as a known residual.

## 1. The defect

The twelve `.cpp` files that make up the shared core were spelled out by hand in four
places: `python/CMakeLists.txt` (`CORE_CPP`), `standalone/CMakeLists.txt` (inside
`add_library`), and `web/wasm/CMakeLists.txt` **twice** (the web target and the Node
parity target, which differ only by the embind bridge in front of the same list).

The lists agreed. The problem is what happens when a thirteenth file is added, and it is
made worse by an asymmetry:

- The R package needs **no edit at all**. `src/Makevars` sets flags only; R compiles every
  `.cpp` in `src/`, so a new core file is picked up automatically.
- The other three builds need four edits.

So the build a developer is most likely to try first is the one build that cannot detect
the omission. The other three fail later — at link time if the new translation unit is
referenced, and not at all if it only registers or overrides something, which is the
worse case because it produces a working binary with a piece of the engine missing.

`CORE_CPP` also drives the sdist staging (`docs/85`): the same list decides which files
are copied into `python/_core_src/`. A miss there ships an sdist whose engine does not
link — the failure `docs/85` had just finished fixing.

## 2. What was implemented

`cmake/CoreSources.cmake` holds the list once and is included by all three CMake
projects. It exports:

- `RSYMBOLIC2_CORE_CPP` — the twelve file names, in the order they were always compiled
  in. Order is preserved deliberately so object and link order are unchanged from before
  the change; the check below compares sets, so the order stays a human choice.
- `RSYMBOLIC2_R_ONLY_CPP` — `cpp11.cpp` and `rsymbolic2_r.cpp`. They live in the same
  directory because a CRAN package must be self-contained, but only the R build may
  compile them. The exclusion belongs to the check, not to the list.
- `rsymbolic2_core_sources(<dir> <outvar>)` — verifies the list against the `.cpp` files
  actually in `<dir>`, then sets `<outvar>` to the list expanded to full paths.

Verification and expansion are **one call** on purpose: there is no way to obtain the
source list while skipping the check, so a build added later cannot quietly omit it. On a
mismatch the configure step stops with the offending file names in both directions ("in
src/ but not listed" / "listed but not in src/") and a pointer to this file.

Three details are load bearing:

- **`CONFIGURE_DEPENDS` on the check's glob.** It makes the build system re-run the glob
  before each build and re-configure when the result changed, so a core file added after
  a tree was configured fails at the next `cmake --build`, not at link time. The usual
  objection to globbing does not apply: the glob only feeds the check; the source list
  stays explicit. `.o`/`.dll` files left in `src/` by a local R build do not match
  `*.cpp`, so they cause no re-configure churn.
- **The sdist carries the list next to the sources it names.** An unpacked sdist has no
  `cmake/` above it, so the staging block copies `CoreSources.cmake` into `_core_src/`
  alongside the twelve `.cpp` files; `sdist.include = ["_core_src/**"]` already picks it
  up, so `pyproject.toml` did not change. `python/CMakeLists.txt` selects the list from
  the same branch that selects the sources — a checkout never reads a staged list. The
  check then runs against `_core_src/` too, where it doubles as a completeness check on
  the tarball.
- **`cmake/**` was added to `deploy-pages.yml`'s path filter.** The WASM build is the one
  that runs in CI, so it is where a missing entry surfaces without a developer running
  anything locally; a change confined to the list must therefore rebuild it.

`web/wasm/CMakeLists.txt`'s two targets now share one expanded `CORE_SOURCES` plus a
named `WASM_BRIDGE`, which puts the real difference between them — link options only —
into the structure. Merging them into one OBJECT library would halve WASM compile time
and was deliberately left alone: it changes build semantics for a speed gain nobody has
asked for.

## 3. What this does not fix

The asymmetry itself is unchanged: R still needs no edit, and a developer who only ever
builds the R package still sees nothing. What changed is the other three builds' failure
mode — from "silently stale, or a link error naming a symbol" to "configure stops and
names the file and the list to edit". CI narrows it further: any core `.cpp` added under
`r-package/rsymbolic2/src/` triggers the Pages workflow, whose WASM configure runs the
check.

`RSYMBOLIC2_R_ONLY_CPP` is a hand-maintained exclusion. If another binding ever puts a
non-core `.cpp` in `src/`, it has to be named there. Detecting R-only sources by content
(say, an `#include <cpp11.hpp>` scan) was rejected as too clever for a two-entry list.

## 4. Verification

Windows 11 (R 4.6.0 / Rtools45 / Python 3.13) and Ubuntu 24.04 (WSL, Python 3.12).

| | check | result |
|---|---|---|
| standalone | configure, build, `ctest` | 30/30 passed on both platforms |
| Python (checkout) | `pip install --no-build-isolation ./python`, `pytest` | 116 passed; `python/_core_src` not created |
| Python sdist | contents | 58 files; staged 43 = 12 `.cpp` + 30 `.hpp` + `CoreSources.cmake` (was 42) |
| Python sdist | install in a clean venv, isolation on, from a tarball outside the repo | 86 passed, 24 skipped — both platforms, same as `docs/85` |
| WASM | `emcmake` build + `node web/wasm/test/parity_test.cjs` | passed; the rebuilt `web/app/vendor/` artefacts were **byte-identical** to the committed ones |
| R package | `R CMD INSTALL` + `testthat` | 351 passed, 0 failed, 29 skipped (`On CRAN` gates) |

The negative controls matter as much as the passes — a check that never fires is worse
than no check, because it is believed:

| | control | result |
|---|---|---|
| N1 | add `src/zzz_probe.cpp`, re-configure standalone / Python / WASM | all three stop, naming `zzz_probe.cpp` |
| N2 | with a **configured and built** tree, add the probe and run only `cmake --build` | re-configures and stops — `CONFIGURE_DEPENDS` does what the design depends on |
| N3 | add a `nosuch.cpp` entry to the list | stops with "listed but not in src/: nosuch.cpp" |
| U3 | N2 repeated on Ubuntu | stops, and recovers once the probe is removed |

No C++ source and no compilation order changed, so the search is bit-identical by
construction; the identical WASM artefacts are incidental evidence of that.

### An unrelated observation, not acted on

The sdist built **in WSL against the Windows checkout** contained one extra file: the
tarball a previous Windows run had left in `python/dist/`. Rebuilt on Windows with the
same directory present, it was excluded, so the cause is environment-specific rather than
a property of the packaging config, and it was not chased further. Practical form: clean
`python/dist/` before building an sdist, and prefer the Windows-built tarball, until
someone pins down why the two disagree. Nothing published today is affected — there is no
PyPI release yet.

> **Corrected in `docs/88`.** It was not environment-specific, and the packaging config
> was exactly where the cause was: scikit-build-core collects the sdist by walking from
> `python/` and reads only the `.gitignore` files at or below it, so the root file's
> `python/dist/` rule was never in scope on *either* platform. The Windows rebuild looked
> clean because the documented procedure deletes `python/dist/` first. Fixed with
> `sdist.exclude` in `pyproject.toml`; the "prefer the Windows-built tarball" advice above
> is obsolete.
