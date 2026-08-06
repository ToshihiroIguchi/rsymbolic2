## Test environments

- Windows 11 x64 (build 26200), R 4.6.0, Rtools45 (GCC 14.3.0, UCRT)
- Ubuntu 24.04 (WSL2), R 4.3.3 (Ubuntu apt build)

## R CMD check results

`R CMD check --as-cran` produced no ERRORs and no WARNINGs on either platform.
Tests and examples (including `--run-donttest`) pass on both.

Both local runs were made with the CRAN incoming feasibility check disabled,
since this machine cannot reach CRAN; on CRAN's machines that check is expected
to produce the usual first-submission NOTE:

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Toshihiro Iguchi <toshihiro.iguchi.mail@gmail.com>'
  New submission

  Expected for a first submission.

Every other NOTE seen locally is a property of the test machine rather than of
the package, and none is expected on CRAN's build machines (which use
R-project.org builds, have network access, and have the checking tools
installed).

Both platforms report one NOTE from the manual check, for the same reason in two
guises -- the tool that check wants is absent, so it skips a step and says so:

* checking HTML version of manual ... NOTE
  Skipping checking math rendering: package 'V8' unavailable       (Windows)
  Skipping checking HTML validation: no command 'tidy' found       (Ubuntu)

  Nothing about the manual is reported as wrong; the checker is reporting what
  it could not run. Windows is otherwise clean: 1 NOTE in total.

On Ubuntu 24.04 with the apt-packaged R 4.3.3, three further NOTEs appear
(4 in total):

* checking compilation flags used ... NOTE
  Compilation used the following non-portable flag(s):
    '-mno-omit-leaf-frame-pointer'

  This flag is injected into CXXFLAGS by the Ubuntu 24.04 Debian packaging of
  R 4.3.3 itself (present in /usr/lib/R/etc/Makeconf for all C/C++ standards).
  It is not set by this package's Makevars. It does not appear when building
  with R-project.org's Ubuntu binaries (which CRAN uses).

* checking installed package size ... NOTE
  installed size is 7.9Mb
    libs 7.7Mb

  The size is the compiled C++ object code of the OpenMP island-model search
  engine. The engine depends only on the C++ standard library; no third-party
  C++ library and no large data files are included.

* checking for future file timestamps ... NOTE
  unable to verify current time

  The test machine has no access to the time server this check queries.

## Policy points a reviewer usually asks about

* **Two cores.** The search is OpenMP-parallel and defaults to every core, so the
  checks cap it explicitly: every example passes `n_threads = 2L`, and
  `tests/testthat.R` sets `OMP_NUM_THREADS = 2` before the package is loaded.
  The island model is bit-deterministic across thread counts, so the cap changes
  only how fast a check runs, never its result.

* **Examples are executable.** No example uses `\donttest{}` or `\dontrun{}`.
  Each runs in well under a second (0.08-0.75s elapsed on the Windows test
  machine, with `--run-donttest` no longer applicable). The examples that draw
  use ggplot2, a suggested package, and are guarded by
  `requireNamespace("ggplot2", quietly = TRUE)`.

* **Console output from compiled code** goes through `REprintf()`, never
  `printf`/`std::cout`, and is one line per epoch at the default `verbosity = 1`
  (which matches the reference implementation's default). `verbosity = 0`
  silences it.

* The package writes no files, and changes no `options()`, `par()` or working
  directory. Its only side effect is the plot a `plot()` call draws.

## Licensing and attribution

The package is released under the Apache License 2.0. Its default settings and
search behaviour are an independent re-implementation matched to the documented
defaults of 'PySR' and 'SymbolicRegression.jl' (both Apache-2.0, copyright 2020
Miles Cranmer); attribution is provided in `inst/NOTICE` per Apache License 2.0
Section 4, which also records the extent of the derivation (Section 4(b)) and
the position on the upstream names (Section 6: nominative use only). The
package is not affiliated with or endorsed by those projects.

`License: Apache License 2.0` is the standardizable form and needs no
`| file LICENSE` clause, since R ships the license text in `share/licenses`.
The full text is nevertheless installed as `inst/APACHE-LICENSE-2.0.txt`, so a
recipient of the tarball alone has the license this package's derivation depends
on. It is deliberately not named `LICENSE`: that name is reserved by convention
for terms *additional* to the declared license, and this file is an unmodified
copy of Apache-2.0, not an addition to it.

The C++ engine depends only on the C++ standard library; `LinkingTo: cpp11`
(MIT-licensed, header-only) is the only build-time dependency. cpp11's headers
are not redistributed in this tarball, but they are compiled into the installed
shared library, so cpp11's MIT notice is reproduced in
`inst/THIRD_PARTY_NOTICES.txt`.

## Reverse dependencies

None (new package).
