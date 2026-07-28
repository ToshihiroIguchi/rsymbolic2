<!--
SPDX-License-Identifier: Apache-2.0
Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
-->

# 63 · Web GUI: what the browser remembers, and what it deliberately does not

Status: implemented. Extends `docs/51` (web GUI) and `docs/59` (large data in the browser).
Presentation layer only — no change to the C++ core, the R package or the Python package.

Until now the web GUI kept exactly one thing in `localStorage`: the theme. Everything else
lived in the DOM and died on reload — the eighteen fields of the settings dialog, the operator
selection, the hand-typed macro bodies, the two high-accuracy opt-ins. An accidental refresh, or
a Pages redeploy, threw all of it away.

The GUI now remembers **the inputs to the search, and nothing else.** That boundary is the
substance of this note; the persistence mechanism itself is unremarkable.

## 1. What is stored

One key, `rsymbolic2.search-settings.v1`, holding one JSON object (well under 2 KB):

```jsonc
{
  "v": 1,
  "fields":  { "generations": "2800", ..., "batching": false },  // readSettingsFields() verbatim
  "binary_ops": ["add", "sub", "mul"],
  "unary_ops":  ["neg", "exp", "log", "sin", "cos"],
  "macros":  [ { "name": "gauss", "body": "exp(-square(x))" } ],
  "opt_ins": { "linear_scaling": false, "eval_cache": false }
}
```

`fields` is exactly what `readSettingsFields()` returns — strings for the `DEFAULTS` keys and a
boolean for `batching` — so `writeSettingsFields()` restores it with its existing follow-ups
(`syncBatchingDependants()`, `updateSettingsSummary()`). No second copy of the settings table
exists; every reader and writer here is one that already served the dialog.

The theme keeps its own older `theme` key. Folding it into this one would silently reset the
theme of every current visitor, which is a worse outcome than two keys.

## 2. What is not stored, and why

**The data.** Three independent reasons, any one of which decides it:

- The site's proposition is that the data never leaves the browser. Leaving user CSVs in
  `localStorage` on a shared machine undercuts the claim for a convenience nobody asked for.
- It does not fit. `localStorage` is ~5 MB per origin; `docs/59` shows the tables people actually
  load are that size and larger.
- Restoring one would be worse than losing it. The WASM heap is fixed at 128 MB
  (`ALLOW_MEMORY_GROWTH` off, `docs/51`), so auto-loading a large saved table on the next visit
  turns a reload into a page that is broken before the user does anything. Re-picking a file is
  two clicks.

**The results.** They are derived from data that is deliberately gone, so a restored Pareto front
could be neither re-run nor checked against anything. The paths that already preserve a result
survive a reload properly and say what they are: the Pareto CSV download, the expression copy, and
the R/Python snippets in `export.js` that reproduce the *run* rather than its output.

**The data-derived selections** — target column, feature ticks, `sample-rows`/`sample-size`. They
are statements about one table and are meaningless against the next one.

**The results-view controls** — `model_selection`, `logloss`, `fit-view`. One click each, and
`model_selection` is not a preference like the others: it decides which equation the page
*recommends*. A fresh visit must recommend the GUI's documented default (`score`, the sanctioned
divergence in CLAUDE.md), not a choice made weeks ago on different data.

The resulting rule is short enough to hold in the head: **what you set up before pressing Run is
remembered; what the run produced, and how it is displayed, is not.**

## 3. Restoring is announced, never silent

The GUI's defaults promise that pressing Run on arrival runs the PySR-parity search. Restoring a
non-default configuration silently would break that promise invisibly, so the Search card shows a
one-line notice on arrival — *"Restored the settings you last used here."* — with a **Use
defaults** button beside it, whenever the restored state differs from the shipped one. A browser
that remembered the defaults has nothing to disclose and shows nothing.

The notice is an arrival disclosure: it clears on the first Run, when the user has seen the panel
and committed to it. The ongoing signal is the one that already existed — the Settings summary's
`· modified` marker and the per-field highlights, which are computed from the restored values on
first paint because the restore runs before `updateSettingsSummary()` in `init()`.

**Use defaults** is a wider reset than the dialog's own **Reset to PySR defaults**, which stays
field-only on purpose (a button inside the dialog must not rewrite operators and opt-ins the user
cannot see from where they clicked). This one lives in the rail, where all of that state is on
screen: it clears the stored key and returns fields, operators, macros and opt-ins to the shipped
values.

## 4. Storage is untrusted input

A stale schema from an older deploy, a truncated write, or a hand-edited value can all appear at
that key. Restoration is therefore whitelist-based and total: anything unrecognised is dropped
rather than half-trusted, and the cost of a bad blob is the shipped defaults, never a broken
panel.

- The raw string is rejected past 64 KB, and the object past `v !== 1`.
- `fields` keeps only keys present in `DEFAULTS` whose value is a non-blank numeric string, plus
  `CHECKBOX_DEFAULTS` keys whose value is a boolean.
- Operator lists are produced by filtering the canonical `BINARY`/`UNARY` arrays against the
  stored list, so an operator this build no longer has disappears and the rest come back in the
  order `checkedOps()` uses.
- Macros are capped at 20 entries of 200 characters, and entries without string `name`/`body` are
  dropped.
- A section that is absent (an older schema) restores nothing and leaves the shipped defaults in
  place — "invalid" and "absent" must not collapse together, or a missing key would come back as
  an empty operator set.

Content only the engine can judge — a macro body naming a removed operator, an operator set the
user emptied — is restored as-is and still meets the existing Run-time validation (`macroError`,
and the engine's own parser). An empty operator set was already reachable before persistence;
what persistence adds is that it survives a reload, which is what the notice and **Use defaults**
exist to undo.

Every `localStorage` call is wrapped in `try/catch`, as `toggleTheme()` already was: private mode
and a full quota both throw, and the app must not notice. Persistence is a convenience, never a
requirement.

## 5. Saving

One debounced (200 ms) delegated listener on `document` for `input` and `change` covers everything
persisted, so a control added later is covered by construction; events from things we do not store
cost only a redundant identical write. It is guarded by `if ($("settings-dialog").open) return;` —
the dialog is transactional (open snapshots, all four dismissal paths restore), so what is typed
inside it is not committed state. Three paths fire no such event on a persisted control and call
the saver directly: `closeSettings()` (the dialog's commit point, for Apply *and* every restore
path), the example handler after `applyExampleOps()` (checkboxes ticked programmatically), and a
macro row's × button (a click).

## 6. Parity

Nothing here reaches the engine. With the key absent the app behaves exactly as it did before, so
the default-parity comparison against PySR is untouched, and the divergence CLAUDE.md already
sanctions for the GUI (`model_selection = score`) is explicitly *not* persisted — see §2.

## 7. Verification

There is no GUI test harness (`web/wasm/test/parity_test.cjs` is a Node-only engine gate and never
loads `main.js`), so this was verified by driving a real browser against `python web/serve.py`:

1. clean slate — nothing stored, no notice, shipped defaults, summary without `· modified`;
2. round trip — generations 500 via the dialog, `sub` off, `sqrt` on, a `gauss` macro, linear
   scaling on; after reload every one is back, the notice is shown, `· modified` and the per-field
   marks are correct;
3. **Use defaults** — fields, operators, macros and opt-ins back to shipped values, notice gone,
   key removed;
4. Cancel/Esc in the dialog is never persisted (a committed value from earlier survives it);
5. hostile blobs — unparseable, `v: 99`, wrong types per field, a non-array operator list, 40
   macros with a 500-character body, and a 70 KB blob: each falls back exactly as §4 describes,
   with zero console errors;
6. storage unavailable — `setItem` stubbed to throw: the UI stays fully usable;
7. engine untouched — with storage cleared, the Quadratic example still recovers
   `((square(x0) * 2.5) - 1.3)`, loss 2.2e-10, R² 1.
