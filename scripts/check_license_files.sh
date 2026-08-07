#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
#
# Licensing gate for every distribution rsymbolic2 ships from (docs/84).
#
# The predecessor of this script was nine `cmp` lines written out by hand in
# license-sync.yml. Byte comparison is the right specification for a copy -- a
# distribution copy must be verbatim, or it is a notice that looks authoritative while
# naming the wrong thing. But a hand-written comparison list can only ever detect
# DIVERGENCE between the pairs someone remembered to list. It is structurally blind to
# OMISSION: a new distribution tree, or a new vendored third-party component, arrives
# with no comparison of its own, and nothing anywhere notices that the list did not
# grow. docs/84 §5.5 recorded that as the residual risk, with "convention and a note in
# web/README.md" as the only guard. This script is what replaces that convention.
#
# The fix is the pattern cmake/CoreSources.cmake already uses for the core source list
# (docs/86): state a thing once, then check it against what is actually on disk, so the
# check cannot quietly cover less than reality. Concretely, the checks below are driven
# by `git ls-files` and by the contents of THIRD_PARTY_NOTICES.txt -- not by a list of
# pairs -- so the set of things examined grows by itself when the repository does.
#
# Run it from anywhere inside a checkout:
#
#     sh scripts/check_license_files.sh
#
# It is called by .github/workflows/license-sync.yml on every push, and again by
# deploy-pages.yml inside the publish job, where a failure can actually stop the web
# GUI from being deployed (workflows fail independently, so a red run in the first one
# would not block the second).

set -eu

cd "$(git rev-parse --show-toplevel)"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT INT TERM

fail=0
err() {
    printf 'FAIL: %s\n' "$1" >&2
    fail=1
}
note() { printf '  %s\n' "$1"; }

# The three originals at the repository root. Every copy anywhere else is a verbatim
# copy of one of these; nothing is generated or templated.
ROOT_LICENSE=LICENSE
ROOT_NOTICE=NOTICE
ROOT_TPN=THIRD_PARTY_NOTICES.txt

for f in "$ROOT_LICENSE" "$ROOT_NOTICE" "$ROOT_TPN"; do
    [ -f "$f" ] || err "repository root is missing $f"
done
[ "$fail" -eq 0 ] || exit 1

# ---------------------------------------------------------------------------------
# 1. Every license file in the repository is compared -- the set is discovered, not listed
# ---------------------------------------------------------------------------------
#
# This is the check that closes the omission gap. It walks the tracked files, picks out
# everything whose name marks it as one of our license/attribution files, and byte-
# compares it against the root original it is a copy of. A new distribution directory
# that carries a copy is therefore compared from the moment it is committed, with no
# edit to this script and no edit to any workflow.
#
# The R copy is named APACHE-LICENSE-2.0.txt, not LICENSE: inst/ installs at the package
# top level, and `R CMD check --as-cran` NOTEs a top-level LICENSE that DESCRIPTION does
# not reference (docs/84 §5.3). The explicit name also cannot be mistaken for the CRAN
# convention where a LICENSE file carries terms ADDITIONAL to the declared license --
# there are none here.
#
# A third-party file that legitimately carries SOMEONE ELSE'S license text would fail
# this check, correctly and loudly: it must be named here so the exemption is a recorded
# decision rather than a silent hole. Nothing qualifies today -- the third-party notices
# we redistribute all live inside THIRD_PARTY_NOTICES.txt rather than as separate files.
THIRD_PARTY_OWNED_LICENSE_FILES=""

printf 'Checking discovered license-file copies against the repository root...\n'
git ls-files >"$tmp/tracked"
found_copies=0
while IFS= read -r f; do
    case "${f##*/}" in
        LICENSE|LICENSE.txt|APACHE-LICENSE-2.0.txt) root=$ROOT_LICENSE ;;
        NOTICE|NOTICE.txt)                          root=$ROOT_NOTICE ;;
        THIRD_PARTY_NOTICES.txt)                    root=$ROOT_TPN ;;
        *) continue ;;
    esac
    [ "$f" = "$root" ] && continue

    for exempt in $THIRD_PARTY_OWNED_LICENSE_FILES; do
        [ "$f" = "$exempt" ] && continue 2
    done

    found_copies=$((found_copies + 1))
    if cmp -s "$root" "$f"; then
        note "ok  $f == $root"
    else
        err "$f differs from $root"
        note "If this file is a THIRD PARTY's own license text rather than a copy of"
        note "ours, add it to THIRD_PARTY_OWNED_LICENSE_FILES in this script."
    fi
done <"$tmp/tracked"

# A discovery-driven check that discovers nothing passes silently, which is the failure
# mode this whole script exists to prevent. There are copies in three distributions.
if [ "$found_copies" -lt 3 ]; then
    err "only $found_copies license-file copies were discovered; the matching rule above is broken"
fi

# ---------------------------------------------------------------------------------
# 2. Each distribution carries the COMPLETE set
# ---------------------------------------------------------------------------------
#
# Check 1 runs in one direction only: it compares the copies that exist. It cannot see a
# copy that is missing, and a distribution carrying LICENSE but not NOTICE would pass it.
# This is the inverse direction, and unlike check 1 the list is hand-written, because
# "which directories are a distribution" is not visible in the filesystem. That is the
# one manual step left, and it is deliberately the smallest one: the entries are whole
# distributions, which are added roughly never, rather than individual file pairs.
#
# Adding a distribution here is what a reviewer should be looking for when a new
# packaging target appears. Adding one and forgetting this list leaves that
# distribution's copies still compared by check 1 -- they just are not required to be
# complete.
#
# rsymbolic2 ships from four places. The repository root is the original, so the other
# three are listed. The Apache-2.0 text goes everywhere a recipient could otherwise end
# up without it: the R package (installed from a tarball that would otherwise contain no
# license text) and the web GUI (a website; nobody there has the repository). The Python
# wheel gets LICENSE through pyproject.toml's wheel.license-files, from the file checked
# here.
DISTRIBUTIONS="
python:LICENSE,NOTICE,THIRD_PARTY_NOTICES.txt
r-package/rsymbolic2/inst:APACHE-LICENSE-2.0.txt,NOTICE,THIRD_PARTY_NOTICES.txt
web/app:LICENSE.txt,NOTICE.txt,THIRD_PARTY_NOTICES.txt
"

printf 'Checking each distribution carries the complete set...\n'
for entry in $DISTRIBUTIONS; do
    dir=${entry%%:*}
    names=${entry#*:}
    if [ ! -d "$dir" ]; then
        err "declared distribution directory $dir does not exist"
        continue
    fi
    IFS=,
    for name in $names; do
        if [ -f "$dir/$name" ]; then
            note "ok  $dir/$name present"
        else
            err "$dir is a declared distribution but does not carry $name"
        fi
    done
    unset IFS
done

# ---------------------------------------------------------------------------------
# 3. Every vendored file is attributed in THIRD_PARTY_NOTICES.txt
# ---------------------------------------------------------------------------------
#
# This is the omission docs/84 §5.5 called out by name: "adding a vendored library
# without its notice passes every automated check". web/app/vendor/ is the only place in
# the repository where third-party code is physically redistributed, and the web GUI is
# a genuine redistribution -- a visitor receives KaTeX, Chart.js and the Emscripten
# runtime, and has no access to the repository.
#
# The check needs no second list, because THIRD_PARTY_NOTICES.txt already names the
# concrete paths each component occupies in its "Present in:" paragraph. So the rule is:
# a vendored file must be mentioned there, either by its own path or -- only for a file
# in a SUBDIRECTORY of vendor/ -- by the path of the directory holding it. The KaTeX web
# fonts are covered that way as a unit; naming twenty-one .woff2 files individually would
# add nothing.
#
# The subdirectory restriction is load-bearing, and a negative control is what found
# that out: "web/app/vendor/" occurs in the notices as the prefix of every path listed
# there, so allowing a file to be covered by its own directory would have made every
# file placed DIRECTLY in vendor/ pass -- which is precisely the omission being checked
# for. A directory only counts when it is deeper than vendor/ itself.
#
# Our own build outputs are not exempt, and do not need to be: rsymbolic2.js and
# rsymbolic2.wasm are named under the Emscripten entry, because they embed the
# Emscripten runtime and its bundled musl-derived libc.
VENDOR_DIR=web/app/vendor

printf 'Checking every vendored file is attributed in %s...\n' "$ROOT_TPN"
git ls-files "$VENDOR_DIR" >"$tmp/vendor"
if [ ! -s "$tmp/vendor" ]; then
    err "no tracked files under $VENDOR_DIR; this check would pass without examining anything"
fi
while IFS= read -r f; do
    # Empty unless the file sits in a subdirectory of vendor/ -- see the note above on
    # why vendor/ itself must never serve as the covering directory.
    subdir=${f%/*}/
    [ "$subdir" = "$VENDOR_DIR/" ] && subdir=""

    if grep -Fq -- "$f" "$ROOT_TPN"; then
        note "ok  $f named in $ROOT_TPN"
    elif [ -n "$subdir" ] && grep -Fq -- "$subdir" "$ROOT_TPN"; then
        note "ok  $f covered by $subdir in $ROOT_TPN"
    else
        err "$f is redistributed but is not mentioned in $ROOT_TPN"
        note "Add the component's license text and a 'Present in:' paragraph naming"
        note "this path (or its directory). See docs/84."
    fi
done <"$tmp/vendor"

# ---------------------------------------------------------------------------------
# 4. The KaTeX banner survives an upgrade
# ---------------------------------------------------------------------------------
#
# KaTeX's minified files as published upstream carry no copyright header at all, unlike
# Chart.js which keeps its own. A one-line MIT banner was added here so that a file
# copied out of vendor/ does not travel with no notice whatsoever. web/README.md asks
# whoever upgrades KaTeX to re-apply it -- an instruction with nothing behind it, which
# is the same shape of defect as the one this script's check 1 fixes: correct only while
# someone remembers.
#
# The version in the banner is checked against the version THIRD_PARTY_NOTICES.txt
# records, so an upgrade cannot leave the two disagreeing either.
KATEX_FILES="$VENDOR_DIR/katex.min.js $VENDOR_DIR/katex.min.css"

printf 'Checking the KaTeX notice banner...\n'
katex_ver=""
for f in $KATEX_FILES; do
    if [ ! -f "$f" ]; then
        err "$f is missing; if KaTeX was removed, drop this check and its entry in $ROOT_TPN"
        continue
    fi
    banner=$(head -n 1 "$f")
    case "$banner" in
        *"KaTeX v"*"MIT License"*"Copyright"*)
            v=$(printf '%s' "$banner" | sed -n 's/.*KaTeX v\([0-9][0-9A-Za-z.-]*\).*/\1/p')
            if [ -z "$v" ]; then
                err "$f has a banner with no readable version"
            elif [ -z "$katex_ver" ]; then
                katex_ver=$v
                note "ok  $f banner: KaTeX v$v"
            elif [ "$v" != "$katex_ver" ]; then
                err "$f says KaTeX v$v but the other vendored KaTeX file says v$katex_ver"
            else
                note "ok  $f banner: KaTeX v$v"
            fi
            ;;
        *)
            err "$f has lost its MIT banner -- re-apply it after a KaTeX upgrade (web/README.md)"
            ;;
    esac
done
if [ -n "$katex_ver" ]; then
    if grep -q "KaTeX.*$katex_ver" "$ROOT_TPN"; then
        note "ok  $ROOT_TPN records KaTeX $katex_ver"
    else
        err "the vendored KaTeX is v$katex_ver but $ROOT_TPN does not record that version"
    fi
fi

# ---------------------------------------------------------------------------------
# 5. The web GUI footer links to files that exist
# ---------------------------------------------------------------------------------
#
# The site is served as files, so the footer's links are only as good as the paths in
# them. A renamed copy would 404 silently on a page whose whole job here is to hand the
# visitor a license text.
printf 'Checking the web GUI footer links resolve...\n'
grep -oE 'href="[A-Z_]+\.txt"' web/app/index.html | cut -d'"' -f2 | sort -u >"$tmp/links"
if [ ! -s "$tmp/links" ]; then
    err "web/app/index.html links to no license text; the footer notices are gone"
fi
while IFS= read -r link; do
    if [ -f "web/app/$link" ]; then
        note "ok  footer -> web/app/$link"
    else
        err "web/app/index.html links missing file: $link"
    fi
done <"$tmp/links"

# ---------------------------------------------------------------------------------

if [ "$fail" -ne 0 ]; then
    printf '\nLicense check FAILED. See docs/84 for what each distribution has to carry.\n' >&2
    exit 1
fi
printf '\nLicense check passed.\n'
