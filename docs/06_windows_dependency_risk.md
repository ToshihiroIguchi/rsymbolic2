# Windows / Maintenance Risk Assessment: Ceres, TBB, OpenMP, Self-implemented LM

**Date:** 2026-06-03
**Motivation:** The earlier decision to demote Ceres and TBB cited "heavy
dependency" as the reason. This document replaces that with concrete, primary-source
evidence of **Windows operational risk and maintenance cost**, because the project's
overriding concern is maintainability and portability (CLAUDE.md priorities), not
performance. The user's top priority is **Windows 11 + R package distribution**.

All claims below are grounded in primary sources (project docs, GitHub issues, CRAN
manuals). Where evidence is weak or absent, this is stated explicitly.

---

## Authoritative Conclusion (calibrated, 2026-06-03)

This is the decision of record. It calibrates the more detailed (and in places
overstated) analysis that follows; where they differ, this section governs.

- **Ceres has a sufficient track record on Windows itself.** The concern is narrower:
  in an **R-package environment built with Rtools/MinGW**, its maintenance cost *may*
  be high. This is a possibility supported by evidence (toolchain mismatch, no CRAN
  precedent), not a certainty.
- **Ceres 2.2+'s Abseil dependency is a long-term maintenance risk factor**, because
  Abseil makes no ABI promise and follows "Live at Head."
- **TBB is not "dangerous."** Via RcppParallel it has a solid CRAN operational record.
  The reason to avoid it is weaker than risk: if OpenMP meets the performance
  requirement, there is simply little reason to add TBB.
- **OpenMP is R's official support mechanism** (`SHLIB_OPENMP_CXXFLAGS`) and is a
  sound first choice for rsymbolic2.
- **The constant-optimization problem here is low-dimensional.** The real question is
  therefore *not* "how good is Ceres's performance?" but "what is the maintenance cost
  of implementing a small LM ourselves?"
- **Decision:** In Phase 2, implement a self-contained LM using **Eigen only**, and
  **measure both convergence quality and execution speed.** Re-evaluate Ceres as an
  *optional* dependency **only if** the self-implemented LM is empirically shown to be
  insufficient.

---

## 0. The decisive fact: R on Windows does NOT use MSVC

This single fact reframes the entire assessment and is easy to miss.

- R packages on Windows are compiled with **Rtools**, a toolchain based on
  **MSYS2 + MinGW (GCC)** — not Microsoft Visual C++.
  Source: CRAN, *Rtools45 for Windows* (https://cran.r-project.org/bin/windows/Rtools/).
- Since R 4.2, the Windows runtime is **UCRT**, and libraries built for the older
  **MSVCRT are incompatible** — linking them produces undefined-reference errors
  (e.g. `__imp___iob_func`, `__ms_vsnprintf`, `_setjmp`).
  Source: Uwe Korn, "Building R Arrow on Windows: A tale of two compilers"
  (https://uwekorn.com/2020/06/14/r-arrow-for-conda-windows.html); CRAN R-for-Windows FAQ.
- Consequence: a library that is "well supported on Windows via MSVC / vcpkg" is
  **not automatically usable from an R package**, because the R package must be built
  with MinGW/GCC and linked against UCRT. MSVC-precompiled binaries cannot simply be
  linked in.

This is why "does it build on Windows?" must be re-asked as "does it build under
**Rtools/MinGW + UCRT**, and can it be bundled into an R package?"

---

## 1. Ceres Solver — Windows + R risk

### 1.1 Officially supported Windows toolchain is MSVC only
- The Ceres installation documentation lists Windows support as **"Visual Studio
  2019 and newer."** MinGW/GCC is **not** in the officially supported platform list.
  Source: Ceres Solver, *Installation* (ceres-solver.org / readthedocs).
- Since R on Windows uses MinGW, **Ceres's officially supported Windows path does not
  match the toolchain R actually uses.** MinGW builds are possible (the issue tracker
  contains MinGW fixes) but are not a first-class, tested configuration.

### 1.2 Heavy, ABI-volatile required dependency: Abseil
- **Ceres 2.2+ requires Abseil** (`>= 20240116`) as a *required* dependency, and
  requires a fully C++17 compiler.
  Source: Ceres *Installation*; CMakeLists.txt.
- **Abseil explicitly does not promise ABI compatibility** and follows a
  "Live at Head" philosophy.
  Source: Abseil FAQ (chromium.googlesource.com/.../abseil-cpp/+/HEAD/FAQ.md);
  *Software Engineering at Google*, ch. 21.
- Real, documented build pain from this transition:
  - "cmake error when building packages with ceres due to absl" — Ceres issue **#1089**.
  - "CMake build with locally installed abseil-cpp not accepted" — Ceres issue **#1098**.
- This is the opposite of what a low-maintenance dependency looks like: a required,
  ABI-unstable, large transitive dependency tracking a moving LTS.

### 1.3 Documented Windows build failures (historical, glog/gflags era)
- "/MT vs /MD runtime conflict between glog.lib and libcpmt.lib" — issue **#683**.
- "Windows build gflags error" — issue **#420**.
- "Linking Ceres to GFlags unresolved symbol" — issue **#519**.
- "Problem when using ceres in Debug model on Windows 10" — issue **#743**.
  Source: ceres-solver/ceres-solver GitHub issues.
- These are MSVC-era issues. The common workaround is **vcpkg (`vcpkg install
  ceres:x64-windows`)** — which produces **MSVC/UCRT-or-MSVCRT binaries unsuitable for
  linking into a MinGW-built R package** (see §0).

### 1.4 R packaging precedent
- **No prominent CRAN package was found that bundles or links Ceres Solver.**
  This is absence of evidence, not proof of impossibility — but the lack of any
  well-trodden path is itself a maintenance risk: there is no reference recipe to
  copy, and the builder would be solving Abseil-under-MinGW-UCRT essentially alone.

### 1.5 Assessment (calibrated)
Ceres is a strong, well-supported solver on Windows **in general** (its MSVC build is
mature and widely used). The concern is **specific to the R-package environment**:
because R on Windows builds with Rtools/MinGW + UCRT, Ceres's officially supported
MSVC path does not match the R toolchain, the required Abseil dependency is
ABI-unstable, and no CRAN precedent exists to copy. In that narrow context the
maintenance cost **may** be high. This is an evidence-supported possibility on the
**portability/maintainability** axis — not a claim that Ceres is generally risky, and
not a performance objection.

---

## 2. Intel TBB (oneTBB) — Windows + R risk

### 2.1 There IS a working CRAN precedent: RcppParallel
- **RcppParallel bundles TBB and is distributed on CRAN, including Windows.**
  Source: CRAN RcppParallel; RcppParallel NEWS.
- Windows support has been actively maintained:
  - 4.3.6 — "Support for TBB on Windows" (milestone).
  - 5.0.0 — "Fixed issue when compiling RcppParallel with Rtools40."
  - 5.1.5 — compatibility with the **R 4.2.0 UCRT toolchain** (patches adapted from
    Tomas Kalibera).
  - 5.1.9 — stopped passing `-rpath` to fix builds with the **LLVM linker on Windows**.
  - 5.1.1 — can use an **external TBB** via `TBB_LIB` / `TBB_INC`.
  Source: RcppParallel NEWS.
- Interpretation: TBB on CRAN+Windows is **demonstrably feasible** — but the long
  trail of Windows-specific fixes shows it required a **dedicated, expert team's
  sustained maintenance** to keep working across Rtools/UCRT/linker changes.

### 2.2 Runtime characteristics
- TBB is **not header-only**; it ships a runtime (`tbb.dll`) that must be locatable
  (PATH / co-located DLL). Source: Intel oneTBB Windows guidance.
- This adds a DLL-bundling and loading concern absent from header-only or
  compiler-builtin alternatives.

### 2.3 Assessment
TBB is **medium risk** for Windows + R — *if* it is consumed through RcppParallel
(`LinkingTo: RcppParallel`), inheriting that project's solved build problems. Vendoring
TBB ourselves means **inheriting the maintenance trail** RcppParallel has been paying
down for years. Given that the island model's load is roughly uniform (TBB's
work-stealing advantage is largely unused here), taking on that maintenance is not
justified unless OpenMP is measured to be inadequate.

---

## 3. OpenMP — Windows + R risk

### 3.1 Officially supported by R, with a defined mechanism
- R provides first-class make macros for OpenMP: **`SHLIB_OPENMP_CXXFLAGS`** (and
  C/F variants), to be used in `Makevars` / `Makevars.win`.
  Source: CRAN, *Writing R Extensions* (§ on OpenMP).
- On **Windows**, Rtools is GCC-based and ships **libgomp**, so OpenMP works when the
  package is built from source. This directly serves the user's top priority.

### 3.2 The real caveat: macOS and CRAN binaries
- "The binary version of packages available at CRAN are **not compiled to use
  OpenMP**." So OpenMP features only exist when users build from source (or on
  platforms where CRAN enables it). Source: *Writing R Extensions*.
- On **macOS**, Apple's Xcode clang **does not ship libomp**, so CRAN macOS builds
  have OpenMP effectively disabled by default; the R macOS project provides
  `libomp.dylib` separately for source builds.
  Source: mac.r-project.org/openmp/; *Writing R Extensions*.
- **Implication, not a blocker:** the library **must be correct and complete when
  OpenMP is absent** (serial fallback). This is a code-design requirement (guard with
  `#ifdef _OPENMP`), not a distribution failure. It does **not** affect the user's
  Windows priority, where Rtools provides OpenMP.

### 3.3 Is OpenMP really more maintainable than TBB?
Yes, for this project, on the maintainability/portability axis:
- **No external library, no DLL to bundle, no version/ABI matrix.** It is a compiler
  feature available in GCC (Rtools), Clang, and MSVC (`/openmp`).
- The official R mechanism (`SHLIB_OPENMP_*`) is a **single, documented, stable** path
  versus TBB's multi-year stream of Windows-specific fixes.
- Trade-off accepted: OpenMP gives less control over scheduling than TBB and is
  disabled in CRAN macOS *binaries*. For a coarse-grained, roughly uniform island
  model, this is an acceptable trade — and serial correctness is required regardless.

### 3.4 Assessment
OpenMP is **low risk** for Windows + R, with one explicit, manageable requirement:
**serial-correct fallback when OpenMP is unavailable.**

---

## 4. Eigen / Self-implemented LM — the baseline

- **Eigen is trivially distributable to R via RcppEigen**: header-only, included in
  the package, **no system dependency**, consumed with `LinkingTo: RcppEigen`.
  Source: CRAN RcppEigen; package docs ("users do not need to install Eigen itself").
- A self-implemented Levenberg-Marquardt on Eigen therefore introduces **zero new
  external/system dependencies** beyond a header-only library that already has a
  proven CRAN distribution path on Windows.
- Cost is shifted from *distribution/maintenance* to *implementation + verification*
  (writing and testing the solver). That is a one-time, in-repo, fully controllable
  cost — the kind the priority order prefers over recurring external-toolchain risk.
- **Uncertainty:** whether a compact LM matches Ceres's robustness on degenerate
  Jacobians is **not yet known** and must be measured. If it proves inadequate, Ceres
  becomes a justified, evidence-backed fallback — accepting its Windows/R cost
  knowingly.

---

## 5. Comparison Table

Scale: ✓✓✓ = strongly favorable, ✓✓ = favorable, ✓ = acceptable/with caveats,
✗ = unfavorable. Ratings reflect the **maintainability/portability** axis the project
prioritizes, **for the Windows 11 + R distribution target specifically**.

| Criterion (weight for this project)        | Self-implemented LM        | Ceres Solver                          | OpenMP                          | Intel TBB                                |
|--------------------------------------------|----------------------------|---------------------------------------|---------------------------------|------------------------------------------|
| **Performance**                            | ✓ adequate for k≤~5 (unverified) | ✓✓✓ best-in-class NNLS, robust   | ✓✓ good for coarse islands      | ✓✓✓ work-stealing (advantage unused here)|
| **Implementation effort**                  | ✗ must write + verify LM (~hundreds of LOC) | ✓✓✓ mature API, drop-in    | ✓✓✓ pragmas only                | ✓✓ moderate integration                  |
| **Windows maintenance cost**               | ✓✓✓ header-only Eigen only | ✗ MSVC-only official; Abseil ABI-volatile; MinGW/UCRT mismatch | ✓✓ libgomp in Rtools; stable | ✓ feasible but multi-year Windows fix trail |
| **R package distribution ease**            | ✓✓✓ LinkingTo RcppEigen, no system dep | ✗ no CRAN precedent; bundle Abseil+Ceres under MinGW/UCRT | ✓✓ official SHLIB_OPENMP macros (binaries macOS-off) | ✓✓ via LinkingTo RcppParallel; ✗ if self-vendored |
| **Future Python distribution ease**        | ✓✓✓ vendor Eigen in wheel  | ✓✓ wheels feasible (MSVC + delvewheel/manylinux) | ✓✓ MSVC /openmp, libgomp/libomp | ✓✓ tbb wheels exist, DLL bundling        |
| **CI/CD operability**                      | ✓✓✓ no system deps         | ✓ MSVC CI ok via vcpkg; ✗ R/MinGW CI is the hard part | ✓✓✓ no system deps        | ✓✓ workable (RcppParallel demonstrates)  |

### Reading the table for this project
- **Constant optimization:** Self-implemented LM wins on every axis the project ranks
  above performance. Ceres wins only on performance and raw implementation effort —
  the two lowest-priority axes — while scoring worst exactly on Windows maintenance
  and R distribution, the user's top concerns. **Default = self-implemented LM;
  Ceres = evidence-gated fallback** is therefore well supported by primary evidence,
  not just "it's heavy."
- **Parallelism:** OpenMP wins on maintenance, R distribution, and CI/CD. TBB's only
  clear win (work-stealing performance) is largely irrelevant to a uniform island
  model. **Default = OpenMP (with mandatory serial fallback); TBB = evidence-gated
  fallback, and if ever needed, via `LinkingTo: RcppParallel` rather than
  self-vendoring.**

---

## 6. Conclusion

The revised defaults (self-implemented LM + OpenMP) are supported by **specific,
primary-source Windows/R evidence**, not merely by dependency weight:

1. R's Windows toolchain is MinGW/UCRT, which Ceres does not officially target and
   into which MSVC/vcpkg binaries cannot be linked.
2. Ceres 2.2+ requires Abseil, which does not promise ABI stability and follows
   "Live at Head" — a recurring maintenance liability with documented build breakage.
3. No CRAN precedent exists for shipping Ceres, so there is no proven recipe.
4. TBB is feasible on CRAN/Windows (RcppParallel proves it) but only at the cost of a
   sustained Windows-fix maintenance burden; its performance advantage is unused by a
   uniform island model.
5. OpenMP is the R-official, dependency-free parallelism path and works on Windows via
   Rtools' libgomp; its only material caveat (CRAN binaries, macOS) is handled by a
   mandatory serial fallback and does not affect the Windows target.
6. Eigen (hence self-implemented LM) has a proven, system-dependency-free CRAN
   distribution path via RcppEigen.

**Counter-evidence acknowledged:** TBB *is* shippable on CRAN+Windows (RcppParallel),
so "TBB cannot be distributed" would be false. The accurate statement is that doing so
carries a maintenance burden that is unjustified while OpenMP suffices. For Ceres, no
comparable positive R precedent was found.

**Remaining uncertainty:** the self-implemented LM's numerical robustness versus Ceres
is unverified and must be measured (Phase 2 gate). The fallbacks remain available and
are to be adopted only on such evidence.

---

## Sources
- CRAN, *Rtools45 for Windows*: https://cran.r-project.org/bin/windows/Rtools/
- CRAN, *Writing R Extensions* (OpenMP, SystemRequirements): https://cran.r-project.org/doc/manuals/r-release/R-exts.html
- Uwe Korn, "Building R Arrow on Windows: A tale of two compilers": https://uwekorn.com/2020/06/14/r-arrow-for-conda-windows.html
- Ceres Solver, *Installation*: http://ceres-solver.org/installation.html
- Ceres Solver issues #420, #519, #683, #743 (MSVC/glog/gflags); #1089, #1098 (Abseil): https://github.com/ceres-solver/ceres-solver/issues
- Abseil FAQ (ABI / Live at Head): https://chromium.googlesource.com/external/github.com/abseil/abseil-cpp/+/HEAD/FAQ.md
- CRAN RcppParallel + NEWS (TBB on Windows/UCRT history): https://cran.r-project.org/web/packages/RcppParallel/news/news.html
- Intel oneTBB Windows guidance (tbb.dll / PATH): https://www.intel.com/content/www/us/en/developer/articles/technical/build-intel-oneapi-rendering-toolkit-windows.html
- CRAN RcppEigen (header-only, LinkingTo, no system dep): https://cran.r-project.org/package=RcppEigen
- mac.r-project.org/openmp/ (Apple clang lacks libomp)
