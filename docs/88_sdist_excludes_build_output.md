# 88 — The sdist stops shipping local build output, and its own tests finally run

Two residual items closed together, because verifying the first is what makes the second
possible: `docs/86`'s observation that an sdist swallowed a leftover tarball from
`python/dist/`, and `docs/85` §6's note that the `tests/` directory inside the sdist had
never been run.

## 1. The swallowing was not environment-specific

`docs/86` recorded it as a curiosity: an sdist built **in WSL against the Windows
checkout** contained an extra file — the tarball a previous Windows run had left in
`python/dist/` — while the same build on Windows excluded it. It concluded the cause was
"environment-specific rather than a property of the packaging config", and left it.

That conclusion was wrong, and the packaging config was exactly where the cause was.
Reading `scikit_build_core/build/_file_processor.py` settles it without any experiment:

```python
for gi in [Path(".git/info/exclude"), Path(".gitignore")]:
    ...   # global excludes
nested_excludes = {... for dirpath, _, filenames in os.walk(".") ... if filename == ".gitignore" and dirpath != "."}
```

The sdist is collected by walking from **`python/`**, and the only `.gitignore` files
consulted are the one *in* that directory and any *below* it. There is no
`python/.gitignore`. The repository root `.gitignore` — which is where
`python/build/`, `python/dist/` and `python/_skbuild/` are written — sits **above** the
walk root and is never read. So nothing in the configuration excluded those directories,
on any platform.

Measured on Windows, with decoys planted in all three directories plus `.pytest_cache/`:

```
total members: 62
  SWALLOWED: rsymbolic2-0.1.0/_skbuild/DECOY_skbuild.txt
  SWALLOWED: rsymbolic2-0.1.0/build/DECOY_build.txt
  SWALLOWED: rsymbolic2-0.1.0/dist/DECOY_dist.txt
  SWALLOWED: rsymbolic2-0.1.0/dist/rsymbolic2-0.0.1.tar.gz
```

Windows swallows them too. What differed between the two runs in `docs/86` was not the
platform but whether `python/dist/` had anything in it: the documented Windows procedure
(`docs/85`) says to delete it before building, and that is what made the Windows tarball
look clean.

One detail is worth stating so it is not mistaken for a safeguard. The tarball **being
written** escapes its own output — `tarfile.TarFile.add()` skips the archive's own file.
Nothing protects earlier ones. So the failure only ever shows up as a *stale* artefact
inside a fresh sdist, which is the harder version to notice.

## 2. The fix

`python/pyproject.toml`:

```toml
sdist.exclude = ["/build/**", "/dist/**", "/_skbuild/**", "/.pytest_cache/**"]
```

This restates the root `.gitignore`'s `python/` rules where the sdist build can actually
see them. The paths are anchored with a leading `/` so they bind to the sdist root and
cannot match a similarly-named directory nested somewhere inside the package.

It is a restatement, so the two can drift; the comment in `pyproject.toml` says to keep
them in step. Deriving them from the root file instead was considered and rejected —
`pyproject.toml` is static TOML with nowhere to put a derivation, and inventing one would
mean either a generated file or a build-time hook, both of which cost more than the four
lines they would replace.

## 3. Verification, and A4: the sdist's own tests

`docs/85` §6 recorded that both platforms had only ever run the *repository's*
`python/tests` against the sdist-installed package. The tarball's own copy — the thing an
actual user of the sdist would be handed — had never executed. Closing that is also the
strongest available check that the new exclusion did not remove something needed, so the
two were verified as one procedure:

1. build the sdist,
2. copy the tarball **outside the repository**,
3. install it into a clean venv with build isolation **on** (so pip fetches
   `scikit-build-core` and `pybind11` itself, as a real user would),
4. extract the same tarball,
5. run `pytest` against the **extracted** `tests/`.

| | Windows 11 (Python 3.13, Rtools45/MinGW) | Ubuntu 24.04 WSL (Python 3.12, GCC) |
|---|---|---|
| tarball members | 58 | 58 |
| anything under `dist/`, `build/`, `_skbuild/`, `.pytest_cache/` | none | none |
| install from tarball, isolation on | ok | ok, wheel built `cp312-cp312-linux_x86_64` |
| `import rsymbolic2` resolves to | the venv's `site-packages` | the venv's `site-packages` |
| **the tarball's own `tests/`** | **86 passed, 24 skipped** | **86 passed, 24 skipped** |

58 members is the count `docs/86` recorded for a correct sdist, so the exclusion removed
only the decoys. The 24 skips are `pandas` and `matplotlib` absent from a bare venv, the
documented behaviour of those tests. The import path line matters: it confirms the tests
exercised the installed extension and not the extracted source tree, which contains only
`__init__.py` and would have failed at `from ._core import ...` — `conftest.py` is what
strips the shadowing entry, and it is inside the tarball too.

## 4. Documents corrected

- `docs/86`'s closing section called this "an unrelated observation, not acted on" and
  attributed it to the environment. The cause is the walk root, and it is now fixed;
  that section points here.
- `docs/85` §6's residual — the sdist's `tests/` never running — is closed by the table
  above. It has run once on each platform. Making it run *every* time is CI work
  (Phase 2), not something this change claims.
