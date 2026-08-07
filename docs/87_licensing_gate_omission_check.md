# 87 — The licensing gate stops being blind to omission

`docs/84` built the licensing position and put two CI workflows behind it, then recorded
its own residual risk in §7: *"Adding a vendored library without its notice passes every
automated check. Convention and the `web/README.md` note are the only guard."* This
change closes that, and folds in the second convention-only rule that lived next to it —
re-applying KaTeX's MIT banner after an upgrade.

## 1. What was actually wrong

`license-sync.yml` held nine `cmp` lines written out by hand:

```
cmp LICENSE              python/LICENSE
cmp NOTICE               python/NOTICE
...
```

Byte comparison is the right *specification* for a copy. A distribution copy must be
verbatim, or it is a notice that reads as authoritative while naming the wrong thing.
The defect was never the comparison; it was that **the set being compared was a list**.

A list can only detect divergence among the pairs someone remembered to write down. It
is structurally blind in the other direction:

- A new distribution tree arrives with no comparison of its own, and nothing notices that
  the list did not grow.
- A new third-party component is vendored into `web/app/vendor/` and redistributed to
  every visitor of the published site, with no entry in `THIRD_PARTY_NOTICES.txt` — the
  case `docs/84` §5.5 named explicitly. All nine `cmp` lines still pass.

Both failures share a shape: **the check covers less than reality, and covering less is
silent.** That is worse than having no check, because a green run is read as a statement
about the whole repository when it is a statement about nine paths.

## 2. The fix, and why it takes this form

`cmake/CoreSources.cmake` already solved the same problem for the core `.cpp` list
(`docs/86`): state the thing once, then **check it against what is actually on disk**, so
the check cannot quietly fall behind. The licensing gate now works the same way.

The checks moved out of the workflow YAML into `scripts/check_license_files.sh`. Two
reasons, both practical: a developer can run it before pushing, and `deploy-pages.yml`
can run the *same* checks inside its publish job. A check that exists only as inline YAML
in one workflow cannot be reused by the workflow where failing actually blocks a release.

Five checks, in the order they run:

| # | Check | Driven by |
|---|---|---|
| 1 | Every license-file copy in the repository byte-matches its root original | `git ls-files` — **discovered, not listed** |
| 2 | Each declared distribution carries the complete set | a list of three distributions |
| 3 | Every file under `web/app/vendor/` is named in `THIRD_PARTY_NOTICES.txt` | `git ls-files` + the notices file itself |
| 4 | The KaTeX banner is present, self-consistent, and its version is the one the notices record | the vendored files |
| 5 | The web GUI footer's links resolve | `web/app/index.html` |

Check 1 is what closes the first omission: a new distribution directory carrying a copy
is compared from the moment it is committed, with no edit to the script and none to any
workflow. Check 3 closes the second, and needs no second list — `THIRD_PARTY_NOTICES.txt`
already names the concrete paths each component occupies in its `Present in:` paragraph,
so the notices file *is* the manifest.

Check 4 is the KaTeX item. Upstream's minified KaTeX ships with no copyright header, so a
one-line MIT banner was added here; `web/README.md` asked whoever upgrades to re-apply it,
which is an instruction with nothing behind it. It is now enforced, together with the
version agreeing between the two vendored files and with the notices.

### What is still hand-maintained, and why that is the right place for it

Check 2's list of distributions. "Which directories are a distribution" is not visible in
the filesystem, so this one cannot be discovered. It is deliberately the smallest manual
surface available: the entries are **whole distributions**, added roughly never, rather
than individual file pairs added every time anything moves. And forgetting it degrades
gracefully — a new distribution's copies are still compared by check 1; they just are not
*required* to be complete.

This is the same honest asymmetry `docs/86` records for `RSYMBOLIC2_R_ONLY_CPP`: one
short hand-written exclusion, named as such, rather than a clever inference that would be
wrong in a way nobody could see.

`THIRD_PARTY_OWNED_LICENSE_FILES` in check 1 is the other one, and is empty today. A
third-party file carrying *someone else's* license text would fail check 1, correctly and
loudly, and has to be named there — so the exemption is a recorded decision rather than a
hole. Nothing qualifies now: every third-party notice we redistribute lives inside
`THIRD_PARTY_NOTICES.txt` rather than as a separate file.

## 3. Verification

`sh scripts/check_license_files.sh` on a clean tree: passes, having examined 9 copies,
3 distributions, 25 vendored files, 2 KaTeX banners and 3 footer links.

The negative controls matter more than the pass, for exactly the reason this document
exists — a check that never fires is worse than no check, because it is believed. Each
breaks one thing, asserts the failure names it, and restores:

| | control | result |
|---|---|---|
| N1 | a new distribution copy whose contents differ from the root | fires, naming the file and the root it should match |
| N1b | a new distribution copy that *does* match | passes — compared automatically, no edit to the script |
| N2 | a declared distribution loses one of its three files | fires, naming the distribution and the missing file |
| N3 | a vendored file directly in `vendor/` with no entry in the notices | fires |
| N3b | a vendored file in a **new subdirectory** with no entry | fires |
| N3c | a new file added to the covered `vendor/fonts/` directory | passes — no false alarm |
| N4 | the KaTeX banner stripped, as a careless upgrade would | fires |
| N5 | banner version bumped, `THIRD_PARTY_NOTICES.txt` not updated | fires |
| N5b | the `.js` and `.css` banners disagree with each other | fires |
| N6 | the footer links to a file that is not there | fires |

**N3 failed on the first attempt, and that is the finding worth recording.** The original
rule let a vendored file be covered either by its own path or by the path of its
directory. But `web/app/vendor/` occurs in the notices as the prefix of *every* path
listed there, so "covered by its directory" was satisfied for anything placed directly in
`vendor/` — the check passed on precisely the case it was written to catch. The rule now
accepts a directory only when it is **deeper than `vendor/` itself**, which is what the
KaTeX fonts entry needs and nothing more. N3c is the control that keeps that from being
tightened into a false alarm.

No shipped file, build input or compiled source changed, so nothing about the search or
any binary is affected; this is entirely a CI and packaging-hygiene change.

## 4. Documents corrected

- `docs/84` §5.5 and §7 said omission passes every automated check. Both now point here;
  the §7 bullet is struck through rather than deleted, since the reasoning that led to it
  is still the reason the check exists.
- `web/README.md` said "no CI check can see it" and asked the reader to re-apply the
  KaTeX banner from memory. Both statements were true when written and are now false.
