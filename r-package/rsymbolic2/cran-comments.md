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

## Reverse dependencies

None (new package).
