# 90 — The release gate: what "no concerns left" is allowed to mean

rsymbolic2 is not published anywhere. The decision was that it should be published only
once there are no concerns left. This document is what makes that decision checkable,
because "no concerns left" cannot be judged from memory, and read naively it can never be
reached at all.

It is the closing document of a four-phase pass: `docs/87` (licensing gate), `docs/88`
(sdist), `docs/89` (CI), `docs/83` §4 (R-devel). Those did the work. This one says what
the work adds up to and what is still missing.

## 1. The definition, and why the naive one is a trap

The residual sections across `docs/` contain items that will **never** be closed, because
closing them was considered and rejected on evidence:

- the class-B e-graph rules can flip finiteness in the display-only
  `expression_simplified` string (`docs/73`) — kept, because they are the size-reduction
  driver and no search decision reads them;
- the browser row ceiling is conservative by 5–19x (`docs/66` §6) — kept, because the
  binding constraint is time, not memory, and raising it would only permit runs nobody
  waits for;
- the LM residual buffers are the largest remaining `O(m)` term (`docs/65` §3) — kept
  until a measurement says the term matters;
- `std::bad_alloc` inside an OpenMP region reaches `std::terminate` (`docs/82` §5);
- the WASM heap is fixed at 128 MB with growth disabled (`docs/51`);
- `symbolic_regression.formula()`'s first parameter is not the generic's `X` (`docs/61`);
- the operator guards are transcribed rather than re-derived (`docs/84` §7).

Count those as concerns and the count never reaches zero, so the release never happens.
They are not unfinished work; they are **finished decisions**, each with its reasoning
recorded at the point where someone would go looking.

So the definition this gate uses:

> **No concerns left = (a) no concern that is unrecorded or undecided, and (b) nothing
> load-bearing that works only because a person remembered to do it.**

Clause (a) is a documentation property and was largely true before this pass. **Clause (b)
was not true at all**, and is what the four phases were about. At the start, the licensing
gate could not see an omission, the sdist shipped whatever the last local build left
behind, the tarball's own tests had never run, and the C++ core, the R package and the
Python package had never been built by anything except one laptop and one WSL instance.

## 2. The gate

Each row is a condition, its current state, and where the evidence is. A row is green only
if something other than a person's memory produces the evidence.

| # | Condition | Mechanism | State |
|---|---|---|---|
| G1 | Core builds and its tests pass on Windows and Ubuntu | `ci.yml` `core`, every push | **automated** — ctest 30/30 |
| G2 | Python package builds and its tests pass on both | `ci.yml` `python` | **automated** — 116 passed, 0 skipped |
| G3 | The Python sdist is self-contained, ships no local build output, and its own tests pass | `ci.yml` `sdist` | **automated** — 58 members, 86 passed / 24 skipped |
| G4 | R package passes `R CMD check --as-cran` on both, on release R | `ci.yml` `r-package` | **automated** — `Status: OK`, PASS 409 |
| G5 | R package passes `R CMD check --as-cran` on **R-devel**, on both | `r-devel.yml`, weekly + on demand | **automated** — `Status: OK`, 0 NOTEs both |
| G6 | Every distribution carries verbatim license and attribution files | `scripts/check_license_files.sh` in two workflows | **automated** |
| G7 | No third-party file is redistributed without its notice | same script, check 3 | **automated** |
| G8 | The published web GUI matches the core and recovers a known answer | `deploy-pages.yml` WASM build + parity test | **automated** |
| G9 | Recovery rate and search quality have not regressed | Feynman / Nguyen / Keijzer runs | **manual** — see §4 |
| G10 | CRAN incoming feasibility | — | **not runnable** outside CRAN; the `New submission` NOTE is expected |

G1–G8 hold on `master` as of commit `041aa3d`, with all four workflows green.

The negative controls matter as much as the passes, and are recorded with the mechanisms
rather than here: ten for the licensing gate (`docs/87` §3), four for the core source list
(`docs/86` §4), and for G3 the planted leftovers that make the sdist check capable of
failing at all (`docs/89` §2).

## 3. What each phase actually found

Listed because a gate whose construction found nothing would be a gate worth distrusting.

| phase | defect found | would it have shipped? |
|---|---|---|
| 1 (`docs/87`) | the licensing gate's directory rule made every file placed directly in `vendor/` pass — the exact case it was written to catch | yes, silently |
| 1 (`docs/88`) | the sdist swallowed `python/dist`, `python/build`, `python/_skbuild`; not environment-specific as `docs/86` had concluded, but never excluded on any platform | yes, in the first PyPI sdist |
| 1 (`docs/88`) | the sdist's own `tests/` had never executed on either platform | — (a hole, not a defect) |
| 2 (`docs/89`) | two MSVC-only compile errors in the C++: `windows.h`'s `min`/`max` macros unguarded by `NOMINMAX`, and `1.0 / 0.0` in a constant expression | not in shipped code, but the core headers are compiled by MSVC for the Windows Python package |
| 2 (`docs/89`) | CI verified 108 tests where the local machine verified 116, and reported neither a failure nor a skip | yes — as a green tick meaning less than it appeared to |
| 4 (this sweep) | `docs/68` §10 and `docs/77` §6 were stale: both had already been closed by later work and were still being carried as open | — (false open items) |

## 4. What the gate does not cover

Stated rather than implied, because a checklist that hides its own edges is worse than none.

- **G9 is not automated and will not be.** Benchmarks take hours; they are evidence for
  decisions, not a merge gate (`docs/89` §4). A change that keeps every test green while
  wrecking recovery rate passes everything above. The mitigation is that recovery-affecting
  changes are exactly the ones that go through a `docs/` screen with measurements, and
  `diag_search_digest` detects an unintended change in the search trajectory cheaply.
- **G10 cannot be run by anyone but CRAN.**
- **The web GUI is exempt from PySR default parity** by design (`CLAUDE.md`), so its
  `model_selection` default differing from `best` is not a gate failure.
- **No wheels are built anywhere.** See §5, decision B.
- **The permanent decisions in §1 are not gate rows.** If one of them is ever reopened it
  becomes ordinary work with its own document; it does not block a release today.

## 5. The two decisions still open

### A — CRAN submission

Everything the checklist can establish is established: `Status: OK` on release R and on
R-devel, on both platforms, with the full test suite including the searches
`skip_on_cran()` guards (`docs/83`). What remains is not verification but the act of
submitting, plus two mechanical points:

1. **The tarball must be rebuilt from the current `HEAD`.** The one prepared at `0dbdb95`
   is stale by many commits.
2. **win-builder is optional now.** It would add a data point from CRAN's own Windows
   host; the GitHub runner uses the CRAN-built R-devel binary with Rtools, close but not
   the identical machine. Natural to do while preparing the submission, no longer
   load-bearing (`docs/83` §4).

**Not to be done without an explicit instruction.**

### B — PyPI

Undecided, and the gate does not decide it. What the last four phases changed is that the
sdist is now genuinely publishable: self-contained, verified from outside the repository on
both platforms, and with its own tests running on every push. What is still unanswered is
whether to publish **wheels**, and that carries a question no amount of gate-passing
settles — the Windows wheel would be an MSVC build needing `VCOMP140`/`MSVCP140` bundled
via `delvewheel` (`docs/58` §2.1), and macOS remains a platform this project does not
target and cannot verify, where the package would silently lose its parallelism.

The recommendation stands from before: CRAN first, wheels for Linux and Windows only if at
all, macOS sdist-only, and publication gated behind a tag with manual approval.

## 6. Using this document

At the moment of release, the check is not "does it feel ready". It is:

1. All four workflows green on the commit being released (G1–G8).
2. §5's mechanical points done for whichever distribution is being published.
3. Nothing has been added to `docs/` residual sections since the last sweep that is a
   *defect* rather than a recorded decision — §1 is the test for which it is.

If a new distribution target or a new vendored component appears, G6 and G7 extend
themselves; that is the property `docs/87` was built for. If a new binding appears, it
needs a row here and a job in `ci.yml`, and adding those is part of adding the binding.
