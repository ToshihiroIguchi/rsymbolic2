# 76 — Web GUI defect pass

A second pass over the browser GUI (`web/app/`), after the audit in docs/75. That audit read the
code and drove the paths the reading made suspect; this one drove the app at viewport sizes and
through input paths the first pass did not exercise, and found four defects — one layout, two of
one kind, and one that is the file-picker twin of a bug docs/75 already fixed for the example
picker.

Every finding below was reproduced in a browser before being fixed and re-checked there after.
Nothing here touches the search: the changes are `web/app/css/style.css` and the event wiring at
the bottom of `web/app/js/main.js`. The vendored WASM, the bridge and the shared C++ core are
untouched, and `parity_test.cjs` reports the same recovered expression and the same Pareto losses.

## Fixed

### 1. The page scrolled sideways on every phone, and the theme toggle was off-screen

At a 375 px viewport `document.scrollWidth` was 431: the whole page had a horizontal scrollbar,
and because the header is `position: sticky` that clipped header was on screen at every scroll
position. The theme toggle sat entirely past the right edge and could only be reached by scrolling
the document sideways.

Two independent rows overflowed, both for the same reason — a `display: flex` row with the default
`flex-wrap: nowrap` whose contents cannot wrap and whose labels are `white-space: nowrap`:

| row | natural width | space available |
| --- | --- | --- |
| `.header-actions` (status chip + Run + PDF + theme) | 415 px | 375 px viewport |
| `.results-head` in `#pareto-card` (`h2` + recommend + loss axis) | 392 px | 317 px card |

The header case had a second cause on top of the missing wrap: `.header-actions { flex: none }`
means the group will not shrink to its line either, so `flex-wrap` alone on the group would have
had nothing to wrap *within* — the container stays as wide as its content. It needs
`flex: 0 1 auto; min-width: 0` as well.

The Pareto row's overflow cascaded outward through `#pareto-card` → `.charts-row` →
`#results-area` → `.layout`, which is how a card-level miss became a document-level scrollbar.

Fixed:

- **`.header-actions`**, inside the existing `@media (max-width: 1100px)` block — where the header
  already wraps, so the change cannot reach the desktop layout at all: `flex: 0 1 auto;
  min-width: 0; flex-wrap: wrap; justify-content: flex-end`. Run keeps its place beside the chip;
  PDF and the theme toggle drop to a second right-aligned row.
- **`.status`**, same block: `max-width: 100%; overflow: hidden; text-overflow: ellipsis`. The chip
  keeps `white-space: nowrap` — the run timer rewrites it five times a second and a wrapping chip
  would reflow the header on every tick — so below ~320 px the ellipsis is the last resort. The
  full text is announced through `#status-live` either way.
- **`.results-head` / `.results-head-controls`**: `flex-wrap: wrap`, unconditionally. Wrapping is a
  no-op at any width where the row already fits, so this needs no breakpoint of its own.

Verified by measuring `scrollWidth` against `clientWidth` for every laid-out element at 320, 375,
768, 1100 and 1440 px, with a result on screen. Before: five overflowing elements at 375 px and a
431 px document. After: none, and `scrollWidth === clientWidth` at every width. The header is one
row at 1100 px and above, exactly as before. (The one remaining entry at 320 px is `#macro-preset`
clipping its own longest `<option>` label, which is native `<select>` behaviour and does not
overflow the page.)

### 2. Ctrl+Enter started a run from behind the open settings dialog — on values the dialog had not committed

The settings panel is transactional by design (docs/75): `openSettings()` snapshots every field it
owns and all four dismissal paths restore them, so nothing typed inside it takes effect until
Apply. But `run()` reads the settings straight from the DOM, and the Ctrl+Enter handler is
registered on `document` — and a modal `<dialog>` makes the page behind it inert for **pointer**
input only. Keyboard events still reach `document`.

Reproduced: open Settings, type `Generations = 7`, press Ctrl+Enter, press Cancel. The search runs
to completion with 7 generations; Cancel then restores 2800, so the rail reads
"2800 generations" beside a status chip that reads "0.17s | generations: 7". The run also starts
*behind* the modal, where the header progress bar and the Stop button cannot be reached.

### 3. Ctrl+V replaced the loaded table from behind the open data preview

Same root cause, different shortcut. The paste handler is also on `document` (it deliberately
ignores events from form controls so typing into fields keeps working, but it knew nothing about
modality). Pasting while the Data preview is open loads the pasted table — and leaves that dialog
showing, and its note counting, the rows of the table that is no longer loaded.

Reproduced: load the 40-row quadratic example, click the summary line to open the preview, paste a
3-row table. The summary behind the dialog becomes "3 rows × 2 columns" while the dialog still
shows 40 rows under the note "40 rows.".

Both fixed by one guard, `modalOpen()` (`document.querySelector("dialog[open]")`), applied to the
two document-level shortcuts that **mutate** state: while a dialog is open it owns the keyboard,
as it already owns the pointer.

Ctrl+P is deliberately **not** guarded. It prints rather than mutates, the print stylesheet already
hides dialogs (`body.has-print-report dialog { display: none }`), and blocking our handler there
would only hand the print to the browser's own path — which cannot await the chart images
(`printReport()`), so it would make the first print of a session worse, not safer.

### 4. Re-picking the same file did nothing

`#file-input`'s `change` listener never cleared the control. A file input fires `change` only when
the chosen file list *differs* from the one it holds, so choosing the same path a second time fires
nothing at all — and the page silently kept the old table. That is the shape of "I fixed the CSV,
load it again", which is the main reason to open the picker twice.

This is the file-picker twin of docs/75 finding 5, which fixed exactly this for `#example-select`
(a `<select>` fires no `change` when the already-selected option is picked again) and did not look
at the other loader. Same remedy: reset the control after reading it. `e.target.value = ""` is safe
before the asynchronous read, because `file` is a `Blob` reference that outlives the input's value
— verified by loading a file and confirming both that the table arrives and that `input.value` is
empty afterwards.

Note on evidence: unlike findings 1–3 this one was **not** reproduced end to end in the headless
browser, because the behaviour lives in the native file picker, which cannot be driven from a test.
It is Chromium's documented behaviour (`FileInputType::FilesChosen` fires `change` only when the
list changed) and Firefox behaves the same way. The fix is unconditional and its normal path is
verified above.

## Checked and correct — do not re-audit

Beyond docs/75's own list:

- **The WASM bridge reads every option `readConfig()` sends.** All 25 keys the GUI emits appear in
  `rsymbolic2_wasm.cpp`; a silently dropped setting was the failure mode being looked for.
- **Both dialogs at a 375 px viewport.** Neither overflows; they are `max-width`-bounded already.
- **Settings persistence across a reload.** Fields, operator selection and the opt-ins round-trip;
  the arrival notice fires; `reset-ops` re-enables itself from the restored selection.
- **The macro path.** Preset → row → run, including the engine-side rejection of a body using `x`
  twice (`error: macro operator 'sigmoid': body must use 'x' exactly once (found 2)`).
- **Run lifecycle under the new guards.** Stop mid-run, error mid-run, and Ctrl+Enter / paste with
  no dialog open all behave exactly as before.
- **A degenerate constant target** (`y` the same in every row). `sst = 0`, so R² correctly reads
  "—" rather than a divide-by-zero; nothing throws. Worth recording that the `score` rule then
  recommends an 8-node expression over the 1-node constant that fits equally well, because both
  losses are at floating-point noise (0 and 3e-30) and the score of that step is unbounded. That is
  the shared C++ `select_best()` and PySR's own score definition, not a GUI defect — changing it
  would break the R/Python parity the core owes them, so it is left alone.

## Verification

- Browser (Chromium, headless): every finding above reproduced before the fix and re-checked after;
  overflow measured at 320/375/768/1100/1440 px; full run on the quadratic and gravity examples;
  `model_selection` through all three modes; fit and residual views; theme toggle; print report
  built through `beforeprint` with both figures rendering. **No console messages of any level.**
- `node web/wasm/test/parity_test.cjs` — PASSED. The changes are UI-only, so this is the
  regression check that the engine path is untouched.
- The shared C++ core, the R package and the Python package are not touched by this work.
