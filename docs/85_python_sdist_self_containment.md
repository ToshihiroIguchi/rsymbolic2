# 85. The Python sdist could not build itself

**Date:** 2026-08-05
**Status:** implemented; verified on Windows and Ubuntu.
**Change:** `python/CMakeLists.txt`, `python/pyproject.toml`, `.gitignore`.
**Found by:** the licensing audit (`docs/84`), while checking that the sdist carried
`LICENSE` / `NOTICE` / `THIRD_PARTY_NOTICES.txt`. It did. It carried almost nothing else.

## 1. The defect

`python -m build --sdist` produced a 15-file tarball:

```
CMakeLists.txt  conftest.py  LICENSE  NOTICE  pyproject.toml  README.md
rsymbolic2/__init__.py  src/rsymbolic2_py.cpp  tests/*.py
THIRD_PARTY_NOTICES.txt  PKG-INFO
```

`src/rsymbolic2_py.cpp` is the pybind11 bridge — 12 lines of glue over an engine that
was not in the box. `python/CMakeLists.txt` reaches the engine through
`${CMAKE_CURRENT_SOURCE_DIR}/../r-package/rsymbolic2/src`, and an unpacked sdist has no
`../r-package`. Installing that sdist stops at the `FATAL_ERROR` the file raises when the
core is missing.

Nothing was broken for anyone yet: `pip install ./python` from a checkout works, which is
every path anyone has used, and neither PyPI nor CRAN has a submission. But a source
install is the only route for a platform without a prebuilt wheel, and it is the route
distro and conda-forge packagers take by policy. Publishing the package with this in
place would have meant "works only where we happened to build a wheel".

## 2. Why it is structural, and where the choice actually is

Two constraints that cannot both be satisfied by a repository layout:

- **A CRAN package must be self-contained.** So the core has to live inside
  `r-package/rsymbolic2/src/`. That is why it is there, and it is not negotiable.
- **A PEP 517 sdist is built from the project directory** — the one holding
  `pyproject.toml` — and cannot reach above it.

So the core must be **physically present in two distribution trees**. There is no layout
that avoids the duplication. The only real question is *when the copy happens*:

| When | What it costs |
|---|---|
| At commit time (duplicate in git) | Every core change touches two trees; every diff shows the engine twice; drift is caught only by a checker nobody reads until it fires |
| At sdist-build time (generated) | One generated directory, gitignored; the mechanism has to be understood by whoever next edits the build |

The second is the one that keeps the invariant three `CMakeLists.txt` files already state
("that directory is the single source of truth ... we do NOT duplicate any core sources").

## 3. Alternatives measured or rejected

- **`sdist.include = ["../r-package/rsymbolic2/src/**"]`** — the obvious two-line fix.
  **Tried; it does not work.** scikit-build-core accepts the setting without complaint
  and the sdist stays at 15 files. Silent, which is the worst kind: a reader of
  `pyproject.toml` would reasonably believe the core was covered. (`sdist.include`
  documents itself as "files to include ... even if skipped by default" — the file walk
  never leaves the project directory, so a parent path matches nothing.)
- **Move `pyproject.toml` to the repository root.** Works, needs no new mechanism, and
  costs the most elsewhere: the sdist would carry the R package, the web GUI (WASM
  binary, KaTeX, fonts) and `docs/` unless each is excluded by hand, and
  `pip install ./python` — the command in the README, the tutorial, both CI paths and
  `docs/58` — becomes `pip install .` at the root of a repository that is not primarily a
  Python project.
- **Commit a synced copy under `python/`.** Simple and needs no build magic; it is also
  ~1,600 lines of engine duplicated in git, touched twice on every core change. The
  licensing work in `docs/84` had just finished adding a CI job to keep *nine one-page
  text files* in sync; extending that discipline to the engine itself is the wrong
  direction.
- **Publish wheels only, no sdist.** Legitimate, and common for C++ extension packages.
  Rejected because it converts a build problem into a portability problem, and
  portability outranks simplicity in this project's priorities.

## 4. What was implemented

`sdist.cmake = true` makes scikit-build-core run the CMake configure step *before* it
collects the sdist's files. `python/CMakeLists.txt` uses that step to stage the core into
`python/_core_src/`, and `sdist.include = ["_core_src/**"]` puts it in the tarball.

Three details that are load bearing:

- **Gated on `SKBUILD_STATE`.** Verified to be exactly `sdist` during that configure
  run, so an ordinary `pip install ./python` writes nothing into the source tree.
  Confirmed by building a wheel from a checkout and checking `python/_core_src` was not
  created.
- **The copy is explicit, not a glob.** The first attempt copied the directory whole and
  produced an sdist containing `cpp11.cpp`, `rsymbolic2_r.cpp`, `Makevars`, and the `.o`
  files left in `src/` by a local R build. The `.cpp` list is now named once and consumed
  twice — by the staging loop and by `pybind11_add_module` — so the set that ships and the
  set that compiles cannot drift. (It was a local `CORE_CPP` here; `docs/86` moved it to
  `RSYMBOLIC2_CORE_CPP` in `cmake/CoreSources.cmake`, shared with the other builds.)
- **The checkout wins when both exist.** A developer always compiles
  `r-package/rsymbolic2/src`, never a stale staged copy; `_core_src` is used only where
  there is no `../r-package`, which is precisely the unpacked sdist.

## 5. Verification

The test that matters is a clean virtual environment with **build isolation on**, from a
tarball sitting outside the repository — if any path leaked back to the checkout, the
result would be a false pass.

| | Windows 11 / Python 3.13 | Ubuntu 24.04 (WSL) / Python 3.12 |
|---|---|---|
| sdist built on that platform | 57 files, 192 KB | 42 staged files, same set |
| staged set | 12 `.cpp` + 30 `.hpp`; no `.o`/`.so`, no `Makevars`, no `cpp11.cpp`/`rsymbolic2_r.cpp` | identical (42 = 12 + 30) |
| `pip install <sdist>` in a fresh venv, isolation on | built and installed | built and installed |
| test suite against it | **86 passed, 24 skipped** | **86 passed, 24 skipped** |
| `pip wheel ./python` from a checkout | unchanged; `_core_src` not created | — |

The skips are `pandas` and `matplotlib` absent from a bare venv, which is the documented
behaviour of those tests.

## 6. Residual

- ~~**The sdist's own `tests/` are shipped but not what was run.** Both runs exercised the
  repository's `python/tests` against the sdist-installed package, which is the same test
  code; nothing runs the tarball's copy in isolation.~~
  **Closed in `docs/88`.** The tarball's own `tests/` have now been run on both platforms
  against a venv install made from a tarball outside the repository — 86 passed, 24
  skipped on each, with `rsymbolic2.__file__` confirmed to resolve inside the venv's
  `site-packages`. Running them on *every* change is CI work and is not claimed here.
- ~~**Nothing checks that `CORE_CPP` still matches `standalone/CMakeLists.txt`.**~~
  **Fixed in `docs/86`.** The list (four hand-written copies, counting the two in
  `web/wasm/`) is now stated once in `cmake/CoreSources.cmake` and checked against the
  `.cpp` files actually in `src/`, so a missing entry stops the configure step instead of
  failing to link. `python/CMakeLists.txt` consumes that list for both the staging copy
  and the extension target, and the sdist carries a copy of it beside the staged sources.

  The R package is *not* a third list: `Makevars` sets flags only and R compiles every
  `.cpp` in `src/`, so a new core file is picked up there automatically. That asymmetry
  is the trap — the build that needs no edit is the one most likely to be tested first,
  which is how a missing entry in the other two would get past a developer. `docs/86`
  does not remove the asymmetry, only the silence.
