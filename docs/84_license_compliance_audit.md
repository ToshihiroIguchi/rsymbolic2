# 84. License compliance audit: PySR derivation, and what we redistribute

**Date:** 2026-08-05
**Status:** audited; the defects found are fixed in this commit. Verified on Windows:
`R CMD check --as-cran` back to its pre-audit `Status: 1 NOTE`, the Python wheel built
and confirmed to carry all three files in `dist-info/licenses/`, and the web GUI loaded
in a browser with KaTeX and Chart.js still rendering, no console errors, and the footer
links resolving at both desktop and 375 px width. No legal advice — this is an
engineering audit of what the project distributes and what the licenses on that material
require.
**Change:** `NOTICE` and its two distribution copies; `THIRD_PARTY_NOTICES.txt` (new,
plus three copies); `r-package/rsymbolic2/inst/APACHE-LICENSE-2.0.txt` and
`web/app/LICENSE.txt` (new copies of the Apache text); an MIT banner restored to
`web/app/vendor/katex.min.{js,css}`; `web/app/index.html` footer and
`web/app/css/style.css`; `python/pyproject.toml`; `README.md`, `web/README.md`,
`r-package/rsymbolic2/cran-comments.md`, `docs/83`;
`.github/workflows/license-sync.yml` (new) and `.github/workflows/deploy-pages.yml`.
**No code change. Search behaviour is untouched on every platform.**

## 1. Why this audit, and what it is actually about

The worry that prompted it was "are we in trouble with PySR?". That turns out to be the
*least* exposed part of the project, and the audit's main value is elsewhere: the things
that were actually out of compliance are third-party components we ship in the browser
build, which nobody had been thinking about as redistribution at all.

Two questions have to be kept apart, because they have different answers:

1. **May rsymbolic2 exist at all, given how closely it follows PySR?** — Yes, and not
   marginally.
2. **Are we discharging the obligations that come with that, and with everything else we
   ship?** — Mostly, but there were five gaps (§4), four of them substantive.

## 2. Prior art: how permissive-license disputes actually go wrong

Collected to calibrate where the risk really sits. The pattern across all of them is that
**permissive licenses are almost never lost on the "may I use this?" question; they are
lost on attribution, on trademarks, and on binaries that quietly dropped a notice.**

### 2.1 Re-implementation and API/behaviour copying

| Case | What happened | What it tells us |
|---|---|---|
| **Google v. Oracle** (US Sup. Ct. 2021) | Google reimplemented the Java SE API and copied ~11,500 lines of *declaring* code. Held: fair use. | Copying an *interface* — names, signatures, the shape of a configuration surface — to allow programmers to carry their knowledge across is the paradigm case for fair use. Reproducing a documented **default value** is even further from the line: a number chosen for a parameter is not expressive authorship. |
| **SCO v. IBM / Linux** (2003–2021) | Sweeping claims that Linux contained copied UNIX code. Collapsed; almost nothing survived scrutiny. | Allegations of copying are cheap; proving substantial similarity of *protected expression* is not. A project that documents its provenance honestly is in a far better position than one that has to reconstruct it years later. |
| **Compaq's BIOS clean room** (1980s) | Two teams, one reading the spec, one writing the code, never in contact. | The gold standard — and **rsymbolic2 is deliberately not this** (§3.2). That is fine, because unlike Compaq we have a *license*. Clean-room procedure is what you need when you have no permission; it is not required when you do. |
| **Wine / ReactOS "tainted developer" episodes** | Contributors with access to leaked or disassembled Microsoft code were excluded to protect the project's provenance. | Same lesson inverted: the risk is having read a source you had **no right** to read. Reading Apache-2.0 source is exactly what the license invites. |

### 2.2 Apache-2.0 §4 mechanics — the common failure

Section 4 is where permissive licensing actually bites, and the failures are boring and
repetitive:

- **NOTICE files silently dropped.** Apache-2.0 §4(d) makes propagating an upstream
  NOTICE mandatory, and the ASF's own guidance repeatedly has to restate it. Vendoring,
  re-bundling and "we rewrote the packaging" are the usual ways it disappears.
- **§4(b) "state your changes" ignored.** Derivative works are supposed to carry
  prominent notices in the modified files. Almost nobody does it; it costs one comment.
- **§4(a) "give recipients a copy of this License" forgotten in binary artifacts.** The
  repository has a LICENSE file; the wheel, the tarball, the container image and the
  website do not.

### 2.3 MIT/BSD attribution stripped by the build

The single most common real-world violation in web projects: a minifier or bundler drops
the copyright banner, and the shipped `bundle.js` contains MIT code with no notice —
which is precisely the one thing MIT asks for. Removing a copyright notice can also be
argued as removal of copyright management information (DMCA §1202) independently of the
license breach, which is why "it's only MIT, nobody minds" is the wrong frame. **This is
the category rsymbolic2 was actually in** (§4.1).

### 2.4 Trademarks are not licensed by the code license

- **Elastic v. Amazon** (settled 2022) — a dispute over the *name* "Elasticsearch"
  applied to a service, not over the code, which was openly licensed.
- **Mozilla / Debian → Iceweasel** — freely licensed code, but the trademark policy
  forced a rename for a modified build.

Apache-2.0 §6 grants no trademark rights at all. Using an upstream name **descriptively**
("our defaults match PySR's"; "PySR calls this `niterations`") is nominative use and is
what documentation is for. Using it **as identity** — naming a package `PySR-cpp`, or
letting a page read as though it were an official PySR product — is where projects get a
letter. rsymbolic2 is on the safe side of that line, but the line was nowhere written
down before this audit.

### 2.5 Downstream compatibility of *our* choice of license

- **Apache-2.0 is incompatible with GPL-2.0-only** (the patent and indemnity terms are
  additional restrictions). It is compatible one-way with GPLv3.
- This is not hypothetical for us: it is exactly why the R bindings moved from Rcpp
  (GPL-2/3) to cpp11 (MIT) in `docs/41`. That migration is retroactively confirmed here
  as a licensing necessity, not merely a dependency reduction — an Apache-2.0 package
  linking a GPL-2-or-later-only binding would have had to relicense.
- **Enforceability** is settled: *Artifex v. Hancom* (N.D. Cal. 2017) held open-source
  license terms enforceable as contract, and the BusyBox/GPL settlements before it made
  the same point about redistribution obligations. "It's open source" is not a defence
  against failing the conditions.

### 2.6 Upstream relicensing (MongoDB→SSPL, HashiCorp→BUSL, Elastic→SSPL)

A recurring modern shock: a permissively licensed upstream changes license, and
downstreams discover they were depending on the license, not just the code. Worth stating
for completeness: **the Apache-2.0 grant on the PySR / SymbolicRegression.jl versions
studied here is irrevocable for those versions** (§2 and §3 of the license). If upstream
relicensed tomorrow, rsymbolic2's existing position would be unaffected; only the ability
to study *future* versions on the same terms would change.

## 3. rsymbolic2's position on the PySR question

### 3.1 The licenses (verified, not remembered)

Both `MilesCranmer/PySR/LICENSE` and `MilesCranmer/SymbolicRegression.jl/LICENSE` are the
**Apache License, Version 2.0**, appendix filled in as **"Copyright 2020 Miles Cranmer"**.
Neither repository has a NOTICE file, so §4(d) attaches nothing further — the obligations
that apply are §4(a) (ship the license), §4(b) (mark modified files) and §4(c) (retain
attribution notices).

rsymbolic2 is itself Apache-2.0. Same license both directions is the easiest possible
configuration: there is no compatibility question to answer, only a bookkeeping one.

### 3.2 What is actually derived, stated plainly

The uncomfortable part, and the one worth being precise about rather than optimistic:

- **Defaults and search behaviour are matched on purpose.** `CLAUDE.md` makes PySR's
  documented defaults the specification, and `docs/28` tabulates them. Per §2.1 this is
  the weakest possible copyright claim — parameter values are facts about a
  configuration, and reproducing an interface for user portability is the *Google v.
  Oracle* fact pattern.
- **This is not a clean room.** The upstream sources were read while the C++ was written.
  That is not a defect: we have a license that permits derivative works. It only matters
  that we do not *claim* independent origin.
- **A few passages are transcriptions, not re-derivations.** Chiefly the protected
  operator domain guards from `SymbolicRegression.jl/src/Operators.jl` — `safe_log`,
  `safe_sqrt`, `safe_pow` — where `docs/69` and `docs/77` record that folding the branch
  table into "equivalent" logic is exactly how two bugs got in, so the branch structure is
  transcribed verbatim and kept that way deliberately.

  Under copyright these are short, purely functional, and dictated by the behaviour they
  must produce (merger / *scènes à faire* territory); expressed in C++ against a Julia
  original they are also not literal copies of anything. The honest answer is that they
  are covered by the Apache grant regardless, so the analysis never has to reach that
  question — **provided §4 is satisfied.** That is the only real obligation, and it is
  cheap.

### 3.3 Verdict

The PySR relationship was the *initial* worry and is the *smallest* risk in the project:
same license, permitted derivation, attribution already present. What it needed was
precision, not repair — §5.2 and §5.3.

## 4. Defects found

### 4.1 KaTeX redistributed with no copyright notice anywhere — **the real one**

`web/app/vendor/` ships `katex.min.js`, `katex.min.css` and 19 `KaTeX_*.woff2` fonts.
Fixed two ways: the notice is reproduced in `THIRD_PARTY_NOTICES.txt` (the load-bearing
part), and a one-line MIT banner is prepended to the two minified files so a copy taken
out of this directory still travels with its notice. The banner is the belt-and-braces
half and will be lost on the next KaTeX upgrade unless re-applied — `web/README.md`
says so.

The upstream minified artifacts **carry no license banner** (unlike `chart.umd.js`, which
keeps its `/*! Chart.js v4.4.4 ... MIT */` header), and nothing else in `web/app/` named
KaTeX either. The site is public at
`https://toshihiroiguchi.github.io/rsymbolic2/`, so every visitor was being handed MIT
code stripped of the one condition MIT imposes. Textbook §2.3.

### 4.2 The published site carried no license or notice at all

`deploy-pages.yml` publishes `path: web/app`, and `web/app` contained no `LICENSE`, no
`NOTICE`, and no mention of either in the page. So the site also failed Apache-2.0 §4(a)
for **our own** Apache-2.0 WebAssembly build, and omitted the Emscripten runtime (MIT /
NCSA) and its bundled musl libc that `rsymbolic2.js` / `rsymbolic2.wasm` embed. A link to
the GitHub repository would not have discharged this: the recipient of the site does not
have the repository.

### 4.3 Binding libraries acknowledged by name only

`NOTICE` named cpp11 (MIT) and pybind11 (BSD-3) with a URL and no license text. Both are
header-only and get **compiled into** the artifacts we distribute — the installed R shared
library and the Python extension module. pybind11's BSD-3 clause 2 is explicit that a
binary redistribution must reproduce the notice and disclaimer; naming the project is not
that.

### 4.4 The R package shipped no license text, and the attribution had drifted

`r-package/rsymbolic2/inst/` held only `NOTICE`. R itself ships the Apache-2.0 text in
`share/licenses/`, so `License: Apache License 2.0` is standardizable and CRAN-valid
(verified with `tools:::analyze_license`) — but a tarball recipient still received a
derivative work of an Apache-2.0 project with no copy of the license, which is what §4(a)
asks for on its own terms.

Separately, `inst/NOTICE` had already **drifted** from the root `NOTICE`: it had lost the
nominative-use sentence about the PySR name. Exactly the failure mode of §2.2 — a copy
that looks authoritative and is stale. Nothing enforced the two staying equal.

### 4.5 Minor: the SymbolicRegression.jl copyright line was paraphrased

`NOTICE` read "Copyright Miles Cranmer and the SymbolicRegression.jl contributors". The
upstream line is "Copyright 2020 Miles Cranmer". §4(c) is about *retaining* notices, so
they should be reproduced as written, not improved.

## 5. What was changed

### 5.1 `THIRD_PARTY_NOTICES.txt` (new, repository root)

Full verbatim license texts for cpp11 0.5.5 (MIT), pybind11 3.0.4 (BSD-3), Emscripten
6.0.2 (MIT/NCSA, which also covers the bundled musl libc), KaTeX 0.16.11 (MIT) and
Chart.js 4.4.4 (MIT). Each entry names **which distribution it is present in**, so one
file can ship everywhere without claiming, say, that the R package contains pybind11.
One file beats per-distribution variants: variants drift (§4.4), and over-inclusion of a
notice harms nobody.

### 5.2 `NOTICE` rewritten

Upstream copyright lines corrected to match upstream exactly (§4.5); a new paragraph
states the **extent of the derivation** in the terms of §3.2 above — that this is not a
clean room, which passages are transcribed, and that the per-file notices in the C++ are
how §4(b) is satisfied; and a new **Trademarks** section records the §6 position and the
nominative-use limit that §2.4 says needs to be written down.

Stating "not a clean room" in the NOTICE is deliberate. It is the accurate description,
it costs nothing under a license that permits derivative works, and the alternative — a
silence that could later read as a claim of independent origin — is the position §2.1
shows is expensive to defend.

### 5.3 Every distribution now carries LICENSE + NOTICE + THIRD_PARTY_NOTICES

| Distribution | Files | How they get there |
|---|---|---|
| Repository | `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.txt` | source of truth |
| R package | `inst/APACHE-LICENSE-2.0.txt`, `inst/NOTICE`, `inst/THIRD_PARTY_NOTICES.txt` | committed copies; `inst/` is installed |
| Python wheel | `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.txt` | `wheel.license-files` in `pyproject.toml` |
| Web GUI | `LICENSE.txt`, `NOTICE.txt`, `THIRD_PARTY_NOTICES.txt` | committed copies in `web/app/`, linked from the footer |

The Apache text is placed in `inst/` rather than the package root so it does not need a
`| file LICENSE` clause in `DESCRIPTION`; the License field stays the standardizable
`Apache License 2.0` that CRAN prefers, and the text still reaches the user.

It is named `APACHE-LICENSE-2.0.txt`, not `LICENSE`, and that was not a style choice.
`inst/` installs at the package top level, and the first `R CMD check --as-cran` run
after adding it returned a new NOTE:

```
* checking top-level files ... NOTE
File
  inst/LICENSE
will install at top-level and is not mentioned in the DESCRIPTION file.
```

R special-cases the *name*: `inst/NOTICE` and `inst/THIRD_PARTY_NOTICES.txt` sat beside
it without complaint. Declaring `| file LICENSE` would also have silenced it, but that
clause conventionally signals terms *additional* to the declared license, and there are
none — so the file was renamed instead. Re-checked afterwards: `Status: 1 NOTE`, that
one being `checking HTML version of manual` / "Skipping checking math rendering: package
'V8' unavailable", which is a property of this machine and predates the audit. The
licensing files add no NOTE of their own.

### 5.4 A legal footer on the web GUI

`web/app/index.html` gains a footer linking License / Notice / Third-party licenses /
Source, plus the non-affiliation sentence. The page names PySR several dozen times, so it
has to be able to answer on its own what that relationship is (§2.4). Hidden in the print
report, which has its own layout.

### 5.5 Two CI gates, because copies drift

- **`license-sync.yml`** (new) byte-compares all nine distribution copies against the
  root on every push and pull request, and checks that the footer's links resolve to files
  that exist.
- **`deploy-pages.yml`** repeats the three `web/app` comparisons **inside the publish
  job**. This duplication is intentional: workflows fail independently, so a red
  `license-sync` run would not stop the deploy — only a check inside that job can.

Note the limit of both: a byte comparison catches **divergence**, never **omission**. If
a new library is vendored into `web/app/vendor/` and its license text is not added to
`THIRD_PARTY_NOTICES.txt`, CI stays green. `web/README.md` says so where a person adding
a vendored library will read it.

## 6. What was checked and found clean

- **No upstream source files are redistributed.** No vendored copy of PySR or
  SymbolicRegression.jl exists in the tree. `benchmarks/*.jl` are our own scripts that
  *call* SymbolicRegression.jl in a development environment; they ship in no distribution,
  and CLAUDE.md already forbids Julia in anything shipped.
- **Benchmark data is generated, not copied.** `benchmarks/feynman_datasets.R` and
  `nguyen_datasets.R` construct data from the published equations. Physical formulae are
  facts, not expressible works; no dataset file with its own license terms is vendored.
- **Short Julia quotations are attributed where they appear.** `docs/69`, `docs/77` and
  the comment above `pow()` in `expression/dual.hpp` quote a few lines of
  `Operators.jl` to show what is being matched — and the `dual.hpp` one ships inside the
  R and Python packages. Apache-2.0 permits reproduction in source form on the §4
  conditions, which are now met; each quotation already names the upstream file inline,
  and `NOTICE` carries the copyright. No change needed, but it is the kind of thing that
  would look bad if found rather than declared.
- **Package naming does not appropriate the mark.** `rsymbolic2` on PyPI/CRAN, `PySR`
  used only in prose and parameter documentation.
- **The academic citation is present** — `DESCRIPTION` cites Cranmer (2023),
  arXiv:2305.01582. Not a license condition, but the norm the upstream `CITATION.md` asks
  for, and cheap goodwill.
- **SPDX headers** are already on the C++, R, Python, workflow and web sources.
- **Chart.js keeps its upstream banner** in `chart.umd.js`; it was the counter-example
  that made KaTeX's missing one visible.

## 7. Residual risk, stated rather than hidden

- **The transcribed operator guards (§3.2).** Assessed as covered by the Apache grant and
  as unprotectable functional expression in any case. If a stronger position were ever
  wanted, the branch tables could be re-derived from the *documented* semantics — but
  `docs/69` §4.1 records that doing this by reasoning rather than transcription is exactly
  how the last bug survived, so behaviour-first transcription is kept.
- **Omission-blindness in CI (§5.5).** Adding a vendored library without its notice passes
  every automated check. Convention and the `web/README.md` note are the only guard.
- **Nothing here is legal advice.** It is an engineering audit against the text of the
  licenses. The facts it rests on — that both upstreams are Apache-2.0 with the copyright
  line quoted in §3.1, and which files ship where — are verifiable from the repository and
  were verified rather than recalled.
