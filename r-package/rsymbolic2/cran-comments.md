## Test environments

- Windows 11 x64 (build 26200), R 4.6.0, Rtools45 (GCC 14.3.0, UCRT)
- Ubuntu LTS, R release (planned re-check before submission)

## R CMD check results

`R CMD check --as-cran` produced no ERRORs and no WARNINGs.

Two NOTEs remain, both benign:

* checking CRAN incoming feasibility ... NOTE
  Maintainer: 'Toshihiro Iguchi <toshihiro.iguchi.mail@gmail.com>'
  New submission

  Expected for a first submission.

* checking top-level files ... NOTE
  Files 'README.md' or 'NEWS.md' cannot be checked without 'pandoc' being installed.

  Environment-specific: 'pandoc' is not installed on the local Windows test
  machine. This NOTE does not occur on the CRAN check machines, which provide
  pandoc. 'NEWS.md' is valid CommonMark.

Additional NOTEs observed only on Ubuntu 24.04 with the apt-packaged R 4.3.3
(not on CRAN's build machines, which use R-project.org builds):

* checking compilation flags used ... NOTE
  Compilation used the following non-portable flag(s):
    '-mno-omit-leaf-frame-pointer'

  This flag is injected into CXXFLAGS by the Ubuntu 24.04 Debian packaging of
  R 4.3.3 itself (present in /usr/lib/R/etc/Makeconf for all C/C++ standards).
  It is not set by this package's Makevars. It does not appear when building
  with R-project.org's Ubuntu binaries (which CRAN uses).

* checking installed package size ... NOTE
  installed size is 17.9Mb

  The compiled library is large due to the Eigen-based C++ core and the
  OpenMP island-model search. No large data files are included.

## Reverse dependencies

None (new package).
