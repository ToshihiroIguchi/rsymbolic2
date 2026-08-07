# 89 — The checks stop depending on one machine

Phase 2 of the pre-release work: move the verification that had been done by hand, at
milestones, on one developer's machine, into something that runs on every push on both
mandatory platforms.

## 1. Why this is a defect and not an improvement

`CLAUDE.md` says a change is not done until it builds and its tests pass on Windows 11
**and** Ubuntu LTS. That bar was real — the local loop genuinely ran both — but nothing
measured it. Before this change the repository had two workflows: the licensing gate and
the web GUI deploy. **The C++ core, the R package and the Python package had never been
built by anything except one laptop and one WSL instance.**

For iterating, that is fine. For releasing it is not, and the reason is the one this
whole phase turns on: a bar nothing measures is a bar nobody can be shown to have
cleared. "Concerns are zero" cannot rest on a person's memory of having run something.

## 2. What runs

`.github/workflows/ci.yml`, four jobs on `ubuntu-latest` and `windows-latest` both:

| job | what it does | evidence in the log |
|---|---|---|
| `core` | configure, build and `ctest` the standalone harness over the shared C++ core | 30/30 |
| `python` | `pip install ./python` from a clone, then `pytest` | 116 passed |
| `sdist` | build the sdist, install the tarball from outside the repository into a clean venv, run **the tarball's own** tests | 58 members, 86 passed / 24 skipped |
| `r-package` | `R CMD check --as-cran` | `Status: OK`, `[ FAIL 0 | WARN 0 | SKIP 0 | PASS 409 ]` |

Deliberately not here: the WebAssembly build and its parity test (in `deploy-pages.yml`,
which owns the pinned emsdk and publishes the artefact being tested), the licensing gate
(`license-sync.yml`, `docs/87`), and the benchmarks, which take hours and are evidence
for decisions rather than a merge gate.

Two things the jobs do that are worth stating, because they are the difference between a
check and a green tick:

- **The `sdist` job plants leftovers before it builds** — a fake tarball in
  `python/dist`, decoys in `python/build` and `python/_skbuild`. Without them it would
  build a clean sdist on a clean runner and prove nothing about the exclusion added in
  `docs/88`; it would pass because there was nothing to exclude, which is exactly how
  that defect stayed hidden on Windows in the first place.
- **The R job runs with `NOT_CRAN=true`**, so the searches guarded by `skip_on_cran()`
  execute. Those are the slow tests and also the only ones that exercise the engine end
  to end. CI therefore runs a **superset** of what CRAN will run: 409 passing assertions
  against the 351 the local loop sees with the skips in place.

## 3. What bring-up found

Four runs to green. Every failure was a real defect, not workflow plumbing, which is the
argument for having done this before releasing rather than after.

### 3.1 Two MSVC-only compile errors in the C++ (both fixed)

The local Windows loop uses Rtools/MinGW and never MSVC, so nothing had ever compiled the
standalone harness with the Microsoft toolchain.

- `windows.h` defines `min` and `max` as macros unless `NOMINMAX` is set.
  `bench_evolve.cpp`, `bench_memory.cpp` and `bench_profile.cpp` include it without
  setting it, so every subsequent `std::min` and `std::numeric_limits<T>::max` **inside
  the core headers** became a syntax error (`C2589`). MinGW happened not to trip over it.
- `test_dimensional_analysis.cpp` wrote `1.0 / 0.0` for an infinity. Division by zero is
  undefined behaviour and a compiler may reject it in a constant expression; MSVC does
  (`C2124`). `std::numeric_limits<double>::infinity()` is what was meant.

Neither touches shipped code — three development-only benchmarks and one test — and the
MinGW build and `ctest` 30/30 are unchanged. But the Python package *is* built with MSVC
on Windows (`docs/58`), so the core headers being MSVC-clean is not academic.

### 3.2 Two defects in the workflow's own assertions

Both are the same species as the bug `docs/87`'s negative controls caught: a check that
passes for the wrong reason.

- The `sdist` job addressed the tarball as `rsymbolic2-*.tar.gz`, which also matched the
  decoy it plants on purpose. It read the decoy and died on "not in gzip format". It now
  reads the version out of `pyproject.toml` and names the file exactly.
- The job asserts that the tarball's tests import the *installed* extension rather than
  the extracted source tree — and that assertion was written as `python -c` run from
  inside the extracted tree. `python -c` puts the working directory on `sys.path[0]`, so
  the probe imported precisely the thing it existed to rule out. `-I` fixes it. `pytest`
  is deliberately left without the flag: the tarball's own `conftest.py` is what strips
  that entry there, and that mechanism is part of what the job is testing.

### 3.3 CI silently verifying less than the local machine

The first all-green run reported **108 passed** where the local loop reports **116**.
Nothing was red. Nothing was reported as skipped. `sympy` was missing from the runner and
`test_sympy_export.py` guards with a **module-level** `pytest.importorskip`, so the entire
file vanished from collection rather than appearing as skips — the pattern `docs/81`
already recorded once as hiding verification.

`scikit-learn` was the same story for the estimator-contract tests, visible only as one
unexplained skip. With `pandas`, `matplotlib`, `sympy` and `scikit-learn` installed the
job now collects and passes the same 116 tests as a local run.

Both pytest invocations gained `-rs`, so every skip prints its reason. A skip that appears
only as a number is a hole nobody reads.

The general point, and the reason this section exists rather than just the fix: **a CI
that quietly verifies less than the developer's own machine is not a safety net. It is a
second opinion that agrees because it looked at less.** Equality with the local run is
the property being maintained, not any particular count.

The `sdist` job is the deliberate exception — its venv stays bare, because it is the only
place that checks the package works with none of its optional extras, which is what a
plain `pip install rsymbolic2` gives a user. The 24 tests that skip there are run with
their extras by the `python` job on the same platform.

## 4. What this does not do

- It does not build wheels. Whether to publish them at all is decision **B**, still open;
  `docs/58` §4 has the Windows runtime-DLL problem that would have to be solved first.
- It does not run the benchmarks, so a change that preserves every test while wrecking
  recovery rate would still pass. That has always been true and is why Feynman runs are a
  separate, manual, evidence-gathering exercise.
- It does not check R-devel. That is `docs/83` §4, and is the remaining item before the
  CRAN question can be answered — a check service, not a submission.
