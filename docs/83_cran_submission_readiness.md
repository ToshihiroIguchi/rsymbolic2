# 83 — CRAN submission readiness of the R package

Where `r-package/rsymbolic2` stands against CRAN's rules as of 2026-08-05, what was
changed to get there, and the one pre-submission step that is deliberately **not** done
yet. The package has **never been submitted and never been released**, so nothing here
is a change to a published version.

## 1. Status

| | Result |
|---|---|
| Windows 11, R 4.6.0, Rtools45 (GCC 14.3.0, UCRT) | `R CMD check --as-cran` → **Status: OK** (no NOTEs) |
| Ubuntu 24.04 (WSL2), R 4.3.3 (apt build) | `R CMD check --as-cran` → **3 NOTEs**, 0 ERRORs, 0 WARNINGs |
| **R-devel, Windows Server 2022 (GitHub Actions)** | `R CMD check --as-cran` → **Status: OK** (no NOTEs), see §4 |
| **R-devel, Ubuntu 24.04 (GitHub Actions)** | `R CMD check --as-cran` → **Status: OK** (no NOTEs), see §4 |

Both local runs disable the CRAN incoming feasibility check
(`_R_CHECK_CRAN_INCOMING_=false`), because neither machine can reach CRAN. On CRAN's own
machines that check is expected to emit the ordinary `New submission` NOTE.

The three Ubuntu NOTEs are properties of that machine, not of the package, and each is
explained in `cran-comments.md`: the `-mno-omit-leaf-frame-pointer` flag injected by
Debian's packaging of R, an installed size of 7.9Mb (7.7Mb of it the compiled OpenMP
search engine), and `unable to verify current time` on a box with no route to the time
server.

## 2. What was changed to reach that state

Four things a reviewer would have written back about, fixed before submission rather
than after:

- **Two-core policy.** CRAN's repository policy allows a check no more than two
  simultaneous threads. The search is OpenMP-parallel and defaults to every core
  (`omp_get_max_threads()`), so the checks now cap it explicitly: every example passes
  `n_threads = 2L`, and `tests/testthat.R` sets `OMP_NUM_THREADS = 2` before the package
  is loaded — early enough that no individual test has to know about it. The island model
  is bit-deterministic across thread counts (`docs/37`), so the cap changes how fast a
  check runs and nothing else.
- **`\donttest{}` removed.** All ten examples were wrapped in it and none qualified:
  measured with `--run-donttest` they ran in 0.08–0.75s each. `\donttest` is for examples
  that genuinely cannot run, and asking for exactly this unwrapping is a standard review
  comment. The examples that draw are guarded by
  `requireNamespace("ggplot2", quietly = TRUE)`, ggplot2 being a suggested package.
- **A stray `tests/testthat/Rplots.pdf`** — git-ignored, therefore invisible in every
  diff, but `R CMD build` ships what is on disk, and it was inside the tarball being
  checked. Deleted, with the name added to `.Rbuildignore` so a later local run cannot
  reintroduce it.
- **Description reference format** — now `Cranmer (2023) <doi:10.48550/arXiv.2305.01582>`,
  the form CRAN asks for.

`cran-comments.md` states the two-core measures, the example timings, that compiled-code
diagnostics go through `REprintf()` (never `printf`/`std::cout`), and that the package
writes no files and changes no `options()`, `par()` or working directory.

## 3. Things that were checked and needed no change

- `License: Apache License 2.0` is canonical (`tools:::analyze_license()` reports both
  standardizable and canonical); attribution to PySR / SymbolicRegression.jl lives in
  `inst/NOTICE` per Apache-2.0 §4.

  **Superseded in part by `docs/84`.** The licensing audit added
  `inst/APACHE-LICENSE-2.0.txt` (the license text itself, which the tarball did not
  previously carry) and `inst/THIRD_PARTY_NOTICES.txt` (cpp11's MIT notice, which the
  installed shared library needs), rewrote `inst/NOTICE`, and extended the licensing
  section of `cran-comments.md`. The `License:` field is unchanged, and `--as-cran`
  still reports the same NOTEs as it did before the audit.
- Title is in title case; the Description field is 546 characters and does not open with
  the package name.
- `\value` is present on every exported function's help page, and every page now carries
  a `\seealso` (docs commit `ea07088`).
- `CXX_STD = CXX17` is set in both `Makevars` files and is still required, because
  `Depends: R (>= 4.2.0)` includes versions whose default is older than C++17.
- OpenMP is requested through `SHLIB_OPENMP_CXXFLAGS`, the R-official mechanism, and the
  serial fallback is exercised on platforms without it.
- Test suite: `[ FAIL 0 | WARN 0 | SKIP 29 | PASS 351 ]` in ~6s; the 29 skips are
  `skip_on_cran()` on the long-running searches.
- NEWS.md is a single `0.1.0` section (docs commit `f1b0a54`), which R's own news parser
  reads correctly. The version stays `0.1.0` — a `.9000` development suffix marks changes
  since a *released* version, and there is none.

## 4. The deferred step: R-devel — **done (2026-08-07)**

Both local environments run a **release** R. CRAN expects a package to check on
**R-devel** as well, and this was the one known-open item on the checklist; everything
else in this document was already measured.

It was deferred on 2026-08-05 because the usual route is
`devtools::check_win_devel()`, which uploads the tarball to an external service and
returns its result by email — an action belonging to the moment of submission rather
than to hardening. That reasoning applied to win-builder specifically, not to the
question. Once `ci.yml` existed (`docs/89`), a better route was available:
`.github/workflows/r-devel.yml` runs `R CMD check --as-cran` against R-devel on both
mandatory platforms, in infrastructure already trusted, with the result readable
directly instead of arriving by mail. It uploads nothing to anyone.

| environment | result |
|---|---|
| Windows Server 2022 x64, R-devel (2026-08-06 r90366 ucrt) | **Status: OK** — 0 NOTEs |
| Ubuntu 24.04, R-devel (2026-06-21 r90185) | **Status: OK** — 0 NOTEs |

Both with `_R_CHECK_CRAN_INCOMING_=false` (no route to CRAN from a runner) and
`NOT_CRAN=true`, so the searches `skip_on_cran()` guards ran too:
`[ FAIL 0 | WARN 0 | SKIP 0 | PASS 409 ]` on each.

Two things worth noting rather than glossing:

- The result is **cleaner than release R on Ubuntu**, which `docs/83` §1 recorded with
  three NOTEs. Those were properties of Debian's packaging of R (an injected
  `-mno-omit-leaf-frame-pointer`, the installed size, and no route to a time server),
  not of the package, and the runner's R-devel has none of them. Under R-devel the
  installed-size check reports `INFO` rather than `NOTE`.
- The workflow runs **weekly** as well as on demand, so a breaking change in R-devel is
  found while there is time to react, not at the moment of submission.

§1's "Status: OK" can now be read as clean on release R *and* on R-devel, on both
platforms. What it still does not include is CRAN's own incoming-feasibility check,
which no environment outside CRAN can run and which is expected to emit the ordinary
`New submission` NOTE.

### win-builder: still not run, and no longer load-bearing

A win-builder upload would add a fourth data point from CRAN's own Windows machine. The
GitHub runner uses the CRAN-built R-devel binary with Rtools, so it is close but not the
identical host. It remains sensible to do **as part of preparing the actual submission**,
where the emailed result is a natural artefact — it is no longer the thing standing
between here and knowing whether the package survives R-devel.
