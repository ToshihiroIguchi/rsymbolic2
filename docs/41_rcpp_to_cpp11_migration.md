# R bridge migrated from Rcpp to cpp11

**Date:** 2026-06-28

## Why

The project was relicensed wholesale to **Apache-2.0** to minimise legal risk
(`docs/`, NOTICE). The R package's only remaining tension with that goal was its
binding layer: **Rcpp is `GPL (>= 2)`**. Rcpp's headers contain substantial inline
template code that is compiled into the package's shared object, so an
"Apache-2.0" binary that incorporates Rcpp sits in a known copyleft grey area
(Apache-2.0 is one-way incompatible with GPL-2.0).

**cpp11** (r-lib/cpp11) is:

- **MIT-licensed** — compatible with Apache-2.0, removing the GPL linkage.
- **header-only, `LinkingTo:` only** — it is *not* a runtime `Imports`, so the
  package now has one fewer runtime dependency than before.
- **clean on the Rtools/MinGW/UCRT toolchain** R uses on Windows.

This is also a return to the original design: `docs/02_dependency_analysis.md`
(2026-06-03) listed cpp11 — not Rcpp — as the intended bridge.

This is purely an **interface swap**. No search behaviour, no PySR-parity
default, and no C++ core code changed. Rcpp lived only at the R↔C++ call
boundary (one-time argument marshalling and result construction), never in the
generation loop, so there is no performance impact.

## What changed

- **`src/rsymbolic2_r.cpp`** — rewritten with cpp11 types/idioms
  (`cpp11::doubles_matrix<>`, `cpp11::doubles`, `cpp11::strings`,
  `cpp11::writable::{integers,doubles,strings,data_frame,list}`, `cpp11::stop`,
  `[[cpp11::register]]`). Function logic is unchanged. Named-vector reads use
  `r_vector::names()` (nil-safe → empty for unnamed vectors, preserving the old
  "must be a named vector" guard). The core uses its own seeded RNG, so the loss
  of Rcpp's `RNGScope` does not change results.
- **Binding glue** — deleted `src/RcppExports.cpp` and `R/RcppExports.R`;
  regenerated `src/cpp11.cpp` and `R/cpp11.R` via `cpp11::cpp_register()`
  (dev-time tool; requires the `decor` package, which is not a package
  dependency).
- **`DESCRIPTION`** — removed `Imports: Rcpp`; `LinkingTo: Rcpp` → `LinkingTo: cpp11`.
- **`R/rsymbolic2-package.R` / `NAMESPACE`** — dropped `@importFrom Rcpp evalCpp`;
  `useDynLib(rsymbolic2, .registration = TRUE)` retained; NAMESPACE regenerated
  with roxygen2.
- **Build files** — `src/Makevars` and `src/Makevars.win` unchanged
  (`PKG_CPPFLAGS = -I.`, `CXX_STD = CXX17`, `SHLIB_OPENMP_CXXFLAGS`); cpp11 needs
  no special flags.
- **Attribution** — `NOTICE`, `inst/NOTICE`, and `cran-comments.md` updated to
  list `cpp11 (MIT)` instead of `Rcpp (GPL >= 2)`.

## Verification

- **Windows (Rtools45, R 4.6.0):** clean `R CMD INSTALL --preclean`, full testthat
  suite, `R CMD check --as-cran` (same benign NOTEs as before, no new
  ERROR/WARNING).
- **Ubuntu / WSL:** `R CMD INSTALL` + testthat (portability milestone).
- **Python (pybind11):** unaffected by this change.
