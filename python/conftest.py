# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
#
# Make the test suite import the INSTALLED rsymbolic2, never the source tree beside
# this file.
#
# `python/rsymbolic2/` contains only `__init__.py`; the compiled `_core` extension is
# built by CMake and lands in site-packages at install time. So whenever this directory
# is on sys.path ahead of site-packages, `import rsymbolic2` finds the source package,
# fails at `from ._core import ...`, and reports `No module named 'rsymbolic2._core'` —
# which reads like a broken build rather than a shadowed import, and sends you looking
# in the wrong place.
#
# pyproject.toml sets `--import-mode=importlib`, which stops pytest from prepending the
# rootdir. That covers a plain `pytest` invocation. It does NOT cover `python -m pytest`:
# the `-m` flag makes CPython itself put the working directory at sys.path[0] before
# pytest ever runs. This hook removes that entry, so both invocations behave the same
# from any working directory (docs/74).

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

for _entry in list(sys.path):
    # "" and "." both denote the working directory depending on how Python was started.
    _resolved = os.path.abspath(_entry) if _entry else os.getcwd()
    if _resolved == _HERE:
        sys.path.remove(_entry)

# Fail loudly and specifically if the extension still is not importable, rather than
# letting each test module raise the same opaque ImportError during collection.
try:
    import rsymbolic2 as _rsymbolic2
except ImportError as exc:  # pragma: no cover - only hit on a broken environment
    raise ImportError(
        "rsymbolic2 could not be imported. The tests run against the INSTALLED "
        "package, so build and install it first:\n"
        "    pip install ./python\n"
        f"(original error: {exc})"
    ) from exc
