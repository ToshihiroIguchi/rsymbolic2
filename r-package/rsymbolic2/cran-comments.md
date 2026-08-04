## Test environments

- Windows 11 x64 (build 26200), R 4.6.0, Rtools45 (GCC 14.3.0, UCRT)
- Ubuntu 24.04 (WSL2), R 4.3.3 (Ubuntu apt build)

## R CMD check results

`R CMD check --as-cran` produced no ERRORs and no WARNINGs on either platform.
Tests and examples (including `--run-donttest`) pass on both.

On Windows the check is clean with no NOTEs. Both local runs were made with the
CRAN incoming feasibility check disabled, since this machine cannot reach CRAN;
on CRAN's machines that check is expected to produce the usual first-submission
NOTE:

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Toshihiro Iguchi <toshihiro.iguchi.mail@gmail.com>'
  New submission

  Expected for a first submission.

On Ubuntu 24.04 with the apt-packaged R 4.3.3, three NOTEs appear. All three are
properties of that test machine rather than of the package, and none is expected
on CRAN's build machines (which use R-project.org builds and have network
access):

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

## Licensing and attribution

The package is released under the Apache License 2.0. Its default settings and
search behaviour are an independent re-implementation matched to the documented
defaults of 'PySR' and 'SymbolicRegression.jl' (both Apache-2.0); attribution to
those projects (copyright Miles Cranmer) is provided in `inst/NOTICE` per Apache
License 2.0 Section 4. The package is not affiliated with or endorsed by them.
The C++ engine depends only on the C++ standard library; `LinkingTo: cpp11`
(MIT-licensed, header-only) is the only build-time dependency.

## Reverse dependencies

None (new package).
