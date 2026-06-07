# Plan: Close `R CMD check --as-cran`

**Date:** 2026-06-07
**Status:** executed on Windows (2026-06-07) — see §9 Outcome. Ubuntu re-check pending.
**Owner:** maintainer
**Prereq context:** `docs/05` immediate priority #2; the PySR comparison (priority #1) is done (`docs/15`).

This plan takes the package from "`R CMD check --no-manual` is clean (Status: OK,
2026-06-06)" to "`R CMD check --as-cran` is clean on Windows (Rtools/MinGW/UCRT),
with only the unavoidable *New submission* NOTE remaining." Ubuntu LTS re-check is a
follow-up step (Platform Constraints require both), scoped at the end.

---

## 1. Definition of done

- `R CMD check --as-cran` on Windows 11 (R 4.6.0, Rtools45) reports **no ERROR, no
  WARNING**.
- The **only** acceptable NOTE is `checking CRAN incoming feasibility ... NOTE` whose
  body is limited to `New submission` (and, if offline, the skipped network sub-checks).
  Any other NOTE is either fixed or explicitly justified in `cran-comments.md`.
- The PDF reference manual builds (requires LaTeX — see §3).
- The result is recorded: full `00check.log` archived and a one-line status added to
  `docs/05` (replacing the stale "in progress" note for the CRAN-readiness item).

Note (priorities): `--as-cran` is a *release-quality* gate, not a correctness gate.
Per CLAUDE.md the package is already correct and portable; this work raises it to CRAN
hygiene. Keep changes minimal and reversible.

---

## 2. What `--as-cran` adds over the passing `--no-manual` run

The 2026-06-06 log passed every structural check. `--as-cran` adds, roughly:

1. **PDF reference manual build** — runs `pdflatex` over the `.Rd` → manual. This is
   the one hard environment gap (see §3).
2. **CRAN incoming feasibility** (network) — new-submission NOTE; Title/Description
   prose checks; **URL reachability** of `URL:` and `BugReports:` (both GitHub, expected
   reachable); license normalization (`MIT + file LICENSE` is canonical → OK).
3. **Future file-timestamp check** (network) — harmless; downgrades to a skipped
   sub-check when offline.
4. **Stricter source-hygiene** on the *built* tarball — see §4; several build artifacts
   currently sit inside the package source tree and must be excluded so they are not
   shipped.

Everything else in the prior log (namespace, S3 consistency, compiled code, examples,
tests) already passes and is not expected to regress.

---

## 3. Environment gap and the one decision

Probed on 2026-06-07 (R 4.6.0 at `C:\Program Files\R\R-4.6.0`):

| Tool | Status | Needed for |
|------|--------|-----------|
| `qpdf` | present (`c:\rtools45\usr\bin\qpdf.exe`) | PDF size check — OK |
| `pdflatex` | **missing** | building the PDF manual |
| `tinytex` / `spelling` / `rcmdcheck` R pkgs | not installed | optional helpers |

**Decision — how to satisfy the manual build (pick one):**

- **(A, recommended) Install TinyTeX** via `tinytex::install_tinytex()`. R-official,
  ~100 MB, no admin rights, isolated from system TeX, and it auto-installs any missing
  `.sty` on demand. One-time network download. This makes `--as-cran` build the manual
  exactly as CRAN does, so the gate is meaningful.
- **(B) Defer the manual** — run `--as-cran` but keep `--no-manual`. Closes the
  incoming-feasibility / hygiene parts now without LaTeX. Weaker: the manual is what CRAN
  itself builds, so "done" would be partial. Acceptable only as an interim if network is
  unavailable.

This is the only choice in the plan that touches the machine; everything else is
in-repo edits. Recommendation: **A**, because the goal is a *true* CRAN-equivalent pass.

---

## 4. In-repo hygiene work (before running the check)

These are edits inside `r-package/rsymbolic2/`. None change library behaviour.

4.1 **Add `.Rbuildignore`** — the source dir currently contains build/check artifacts
that must not be bundled into the tarball:
- `rsymbolic2.Rcheck/` (a whole prior check tree)
- `rsymbolic2_0.1.0.tar.gz` (a prior build)
- `src/*.o`, `src/*.dll` (compiled objects; `R CMD build` cleans most, but ignore
  defensively and confirm the built tarball is clean)
- editor/OS cruft if any
Verify after building that `R CMD build` produced a tarball with **no** `.Rcheck`,
no nested tarball, and a clean `src/` (only sources + the `rsymbolic/` headers).

4.2 **Add `cran-comments.md`** — standard submission notes: platform/R version tested,
the single expected *New submission* NOTE with a one-line justification, confirmation
that there are no reverse dependencies.

4.3 **Add `NEWS.md`** — a `0.1.0` initial-release entry. Not strictly required to pass,
but expected for a clean submission and cheap to add.

4.4 **Spot-check DESCRIPTION prose** — Title is Title Case (OK); Description is a
full sentence ending with a period (OK). No software-name quoting issues expected.
Confirm `RoxygenNote: 8.0.0` matches the installed roxygen2 (regenerate man/ only if
roxygen flags a mismatch — do not churn the `.Rd` otherwise).

4.5 **LICENSE** — `MIT + file LICENSE` with a `YEAR`/`COPYRIGHT HOLDER` `LICENSE` file
is already the canonical form; no change expected.

---

## 5. Execution steps (Windows)

Run from `r-package/` using the project's R 4.6.0 + Rtools45 toolchain.

1. **(if decision A)** install LaTeX:
   `Rscript -e "install.packages('tinytex'); tinytex::install_tinytex()"`,
   then confirm `Sys.which('pdflatex')` is non-empty in a fresh R session.
2. Apply §4 edits (`.Rbuildignore`, `cran-comments.md`, `NEWS.md`).
3. Build a clean tarball: `R CMD build rsymbolic2`.
4. Inspect the tarball contents (no `.Rcheck`, no nested tarball, clean `src/`).
5. Run the gate:
   `R CMD check --as-cran rsymbolic2_0.1.0.tar.gz`
   (drop `--no-manual`; if decision B, keep it).
6. Read `rsymbolic2.Rcheck/00check.log`. Triage every WARNING/NOTE per §6.
7. Iterate §4/§5 until §1 is met.
8. Archive the final log and update `docs/05` status line. Commit (English message);
   do not push until the user approves.

---

## 6. Anticipated findings and the response to each

| Finding | Expected? | Response |
|---------|-----------|----------|
| `New submission` NOTE | yes | Accept; document in `cran-comments.md`. |
| Bundled `.Rcheck` / nested `.tar.gz` NOTE ("left-over files" / non-standard) | yes if §4.1 skipped | Fixed by `.Rbuildignore`. |
| `pdflatex` not found WARNING | yes if decision B | Resolved by decision A; under B, note it as the known deferred item. |
| URL not reachable NOTE (offline) | only if offline | Re-run with network; GitHub URLs are valid. |
| Title/Description prose NOTE | unlikely | Reword minimally if raised. |
| Any ERROR | not expected | Stop and investigate; this would be a real regression vs the 06-06 run. |

---

## 7. Cross-platform close-out (Ubuntu LTS)

Per Platform Constraints a change is not done until both OSes pass. The Ubuntu build
was last verified in `docs/09`. After the Windows `--as-cran` is clean:
- Re-run `R CMD check --as-cran` on Ubuntu LTS (LaTeX there via `tinytex` or the system
  `texlive` already used in CI/dev).
- Record both logs. Only then mark the CRAN-readiness item DONE in `docs/05`.

This step is in-scope for "done" but can be a separate session; the Windows pass is the
immediate deliverable.

---

## 8. Out of scope

- Actually submitting to CRAN / r-universe (separate decision; needs the user).
- The Feynman benchmark (next roadmap item; planned separately).
- Any library behaviour change. If the check demands one, stop and raise it — that is a
  finding, not a formatting fix.

---

## 9. Outcome (Windows, 2026-06-07)

Final: `R CMD check --as-cran` on R 4.6.0 + Rtools45 → **no ERROR, no WARNING, 2 NOTEs**
(both benign and CRAN-clean). Archived log:
`benchmarks/results/r_cmd_check_as_cran_windows_20260607.log` (gitignored; result
recorded here and in `docs/05`).

Findings, in the order they surfaced, and how each was resolved:

1. **ERROR — Suggests not available (`testthat`, `ggplot2`).** `--as-cran` requires
   Suggests for a complete check; they were not installed. Fixed by installing both
   from CRAN. (They are genuinely used: `testthat` for tests, `ggplot2` for `plot()`.)
2. **ERROR/WARNING — PDF manual LaTeX failure (`pcrr8t`, Courier).** TinyTeX is minimal
   and lacked the Courier font metrics that R uses to typeset the DESCRIPTION URL in
   typewriter. Fixed with `tinytex::tlmgr_install("courier")`. (Does not occur on CRAN's
   full TeX Live.)
3. **WARNING + leftover-`.tex` NOTE — "with index" manual build.** TinyTeX lacked
   `makeindex`. Fixed with `tinytex::tlmgr_install("makeindex")`; the manual then builds
   cleanly and leaves no detritus.
4. **NOTE — no visible binding for `complexity` / `loss`.** `plot.rsymbolic2()` uses
   bare column names inside `ggplot2::aes()` (NSE). Fixed *without a new dependency* by
   declaring them via `utils::globalVariables(c("complexity", "loss", "expression"))` in
   `R/rsymbolic2-package.R` (chosen over importing the `rlang` `.data` pronoun, per the
   Dependency Policy).

**Remaining NOTEs (accepted, documented in `cran-comments.md`):**

- `New submission` — expected for a first submission.
- `pandoc` not installed → `NEWS.md` cannot be checked. Environment-specific only; CRAN's
  check machines provide pandoc. An attempt to install a portable pandoc locally was
  blocked by the environment's autonomy guard (out-of-scope external binary), so the NOTE
  is documented rather than cleared. `NEWS.md` is valid CommonMark.

**Dev-environment changes made (not package changes):** installed TinyTeX, the TeX
packages `courier` and `makeindex`, and the R packages `testthat` + `ggplot2`. None of
these are runtime/library dependencies.
