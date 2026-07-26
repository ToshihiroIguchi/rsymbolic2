# Macro Operators (user-defined operators without a runtime language)

**Date:** 2026-07-23
**Status:** implemented, **off by default** (`macro_ops` empty)
**Interfaces:** R, Python, WebAssembly (web GUI)

---

## 1. The problem

PySR lets a user define an operator by handing it Julia source
(`unary_operators=["myop(x) = ..."]`), because a Julia runtime is there to compile it.
rsymbolic2 must not gain a runtime language (CLAUDE.md: the shipped library and its runtime
must not depend on Julia; the Dependency Policy's default answer is "no"). The two literal
ports of PySR's feature are both unacceptable here:

1. **Embed an expression interpreter + its own AD.** A new subsystem to maintain, which
   every evaluation path (scalar `Dual`, `MultiDual`, the SoA kernels) would have to route
   through — against Simplicity and Maintainability, for a feature nobody has measured.
2. **Call back into R/Python per evaluation.** Catastrophic at millions of evaluations, and
   unsound: the islands run under OpenMP and R is single-threaded.

## 2. The design: a macro is an expansion template, not a node kind

A macro operator is a **single-argument expression template built from the existing
primitive operators**, stored as a small postfix tree with a placeholder where its argument
goes, and **expanded into the expression the moment a growth mutation creates such a node**:

```
gauss(x) = exp(neg(square(x)))          # body: [x, square, neg, exp]
prepend gauss onto (x0 + 1.5)     ->    exp(neg(square((x0 + 1.5))))
```

The engine's node set therefore stays **closed**. Nothing downstream of mutation needs to
know macros exist: `evaluate`, the AD paths, the SoA kernels, `to_string`, `to_latex`,
`simplify`, `display_simplify`, the e-graph, dimensional analysis, the frozen expression
string (`docs/48` D2) and every `predict()` implementation are untouched.

The rejected alternative was a first-class `NodeKind::Macro` with complexity 1. It buys a
stronger parsimony bias, at the price of a per-macro case in every switch listed above plus
a name→definition table in every result consumer, and expression strings that cannot be
evaluated without shipping the macro definitions alongside them. Not worth it.

### Consequences, all deliberate

| Consequence | Why it is acceptable |
|---|---|
| **Complexity counts expanded nodes** (a 4-node macro costs 4) | A macro biases *which structures get proposed*, not what parsimony charges. The bias is real: one mutation now reaches a motif that previously needed four lucky ones. |
| **Macros are invisible in results** (the expanded primitive form is printed) | This is exactly what keeps the reported expression evaluatable by R, Python and JS `predict()` with no macro knowledge. |
| **Numeric literals become tunable constants** seeded at their value | The engine has no frozen-constant concept. `exp(-0.5 * x)` seeds a decay rate rather than fixing one — usually what a user wants, and documented so it is never a surprise. |
| **Only single-argument macros** | Expansion stays a splice. Two placeholders would duplicate the argument subtree (growth, plus the copies' constants fitted independently — never what the user drew). |

## 3. Where it plugs into the search

Everything lives in `mutation.cpp`. `random_tree.cpp` needs no changes: `gen_random_tree`
(initial population) and `gen_random_tree_fixed_size` (`randomize` mutation) both build
through `append_random_op`, so they inherit macros for free.

The **unary alphabet** a growth step draws from is: the enabled primitive operators (one
node each) followed by the macros that still fit in the free space
(`macro_extra_nodes(m) <= room`). One uniform draw picks an entry
(`unary_alphabet_size` / `unary_alphabet_at` / `draw_unary`), and `wrap_unary` turns it into
either one operator node or the expanded template. Three call sites use it:
`append_random_op`, `prepend_random_op`, `insert_random_op`.

**Not hooked, on purpose:**

- `mutate_operator` — an in-place relabel stays a relabel; swapping in a macro there would
  silently grow the tree past the size cap.
- The forced-unary step of `gen_random_tree_fixed_size`
  (`append_random_op(..., make_new_bin_op = false)`) draws from the **primitives only**: its
  size arithmetic assumes a unary operator adds exactly one node, and a macro would
  overshoot the requested tree size.

`ops_within_search_space` (the gate that stops opt-in `strong_simplify` from adopting a
rewrite that uses a disabled operator) now also accepts operators that appear in a macro
body: declaring `gauss(x) = exp(neg(square(x)))` is an explicit statement that `exp` may
appear, so it would be wrong to force the user to enable `exp` for the whole search.

## 4. Parity: inert when unused

With `macro_ops` empty the alphabet **is** `space.unary_ops`, and the single index draw is
the identical RNG call the pre-macro code made. Two things enforce this:

- `test_evolutionary_search.cpp` asserts the exact expression a fixed seed produces on a
  transcendental-free problem — an RNG-stream guard that any reordered or extra draw breaks.
  The strings were captured *before* this work and are unchanged after it.
- `test_macro_op.cpp::test_unfittable_macro_is_inert` runs the growth mutations side by side
  with and without a macro that is too large to ever fit, and requires identical trees.

The R and Python suites additionally assert that `macro_ops = NULL` / `None` reproduces the
run exactly (`expression` and `loss` identical), not merely a comparable one.

## 5. Validation (one parser, every interface)

`expression/parse_expression.hpp` is a recursive-descent parser for the grammar
`to_string` emits — the C++ member of the family that already includes R's `eval(parse())`
path, Python's restricted-namespace `eval`, and the browser parser in `web/app/js/predict.js`.
`make_macro_op` (`evolution/macro_op.hpp`) wraps it with the checks the search relies on, so
every binding rejects the same bodies with the same message:

- unknown function name, or a binary operator written in call form (`add(x, 1)`);
- an identifier other than the argument `x` — a macro cannot reference data columns;
- zero or ≥2 occurrences of `x`;
- empty body, syntax error, a name that shadows a built-in operator;
- a body too large to fit under `max_nodes` (rejected at configuration time rather than
  silently never used).

## 6. Usage

```r
symbolic_regression(X, y,
                    unary_ops  = character(0),
                    binary_ops = c("add", "mul"),
                    macro_ops  = c(gauss = "exp(neg(square(x)))"))
```

```python
symbolic_regression(X, y, unary_ops=[], binary_ops=["add", "mul"],
                    macro_ops={"gauss": "exp(neg(square(x)))"})
```

In the browser the WASM bridge takes the mapping as **two parallel arrays** — embind cannot
enumerate the keys of an arbitrary JS object, and a macro's name is user-chosen, so the
bridge takes the same `(names, bodies)` pair the R and Python bridges are given:

```js
Module.run({ X, y, nrow, ncol, binary_ops: ["add", "mul"], unary_ops: [],
             macro_names: ["gauss"], macro_bodies: ["exp(-square(x))"] });
```

The web GUI (`web/app/`) exposes this as a **Custom operators (macros)** disclosure under the
operator checkboxes — a name/body row per macro, plus a preset list, since a syntax nobody can
guess is a feature nobody uses. Two things are worth stating about where errors come from:

- **The body is validated by the engine, at Run.** `make_macro_op` is the only validator in
  every interface (§5), so the browser rejects exactly what R and Python reject, with the same
  message, and the GUI never carries a second copy of the grammar. A bad body surfaces through
  the bridge's existing `{error}` path into the status bar.
- **Name-level problems are caught in the page** (blank, duplicate, non-identifier, shadowing a
  built-in), because those need no parser and it would be rude to spend a launched run on them.

Operators used *inside* a body need not be checked in the operator panel: declaring the body is
the statement that those operators may appear (the same reasoning as `ops_within_search_space`
in §3).

### The preset list, and what earns a place on it

The presets exist because the syntax is unguessable, so they are also the only place this
feature explains itself. What qualifies a body is **distance**: a macro's entire benefit is
reaching a motif in one mutation that the primitive set reaches only in several (§2), so the
node count it adds over its argument (`macro_extra_nodes`) is the figure of merit. Each
preset's tooltip prints it. The sixteen shipped entries are grouped by the *shape* a user is
looking for rather than by operator family — Peaks (`gauss` 3, `lorentz` 5, `semicircle` 4,
`lognormal` 4), S-curves (`sigmoid` 6, `softplus` 4, `log1p` 3, `saturate` 6), Growth/decay
(`decay` 3, `arrhenius` 3, `planck` 5, `stretchexp` 4, `coth` 3), Powers/roots (`powlaw` 2,
`rsqrt` 3, `relgamma` 6) — and lean towards motifs from the physical
sciences, since ground-truth recovery on Feynman is the primary benchmark. `relgamma`
(`1 / sqrt(1 - square(x))`, the relativistic Lorentz factor) is the clearest case for the
feature: six nodes including a `sqrt` under a subtraction, which a random walk essentially
never assembles.

The five later additions were chosen the same way, and two of them say something the earlier
list did not. `saturate` (`1 / (1 + 1/x)`, six nodes: Langmuir, Michaelis–Menten, saturation
magnetisation) is the strongest argument for shipping presets at all: the motif everyone writes
as `x / (1 + x)` uses `x` twice and is therefore *rejected*, and the single-occurrence rewrite is
not something a user guesses under a form field. `coth` (`1 / tanh(x)`) is the opposite lesson —
it is only the leading term of the Langevin and Brillouin functions, because the full
`coth(x) - 1/x` cannot be written at all; the preset is offered as the reachable part, under the
name of what it actually is. `semicircle` (`sqrt(1 - square(x))`) and `lognormal`
(`exp(-square(log(x)))`) are ordinary distance entries, and `stretchexp` (`exp(-x^c)` seeded at 2,
Kohlrausch relaxation) is named for the family its fitted exponent spans, per the `cube` lesson
below.

`invsq = 1 / square(x)` was **considered and left out as genuinely redundant**, which is the
distinction the `cube` paragraph does not cover: the inverse-square law is `powlaw` with its
fitted exponent landing on -2, so the preset would add a second button for a template already on
the list. Redundancy with an existing *preset* disqualifies an entry; distance from the
*primitive set* is what earns one.

An earlier `cube = x^3` preset was **removed, as a misnamed entry rather than a redundant
one**: because a numeric literal becomes a tunable constant (§2), the body is `x^c` seeded at
3, and its name promised a fixed exponent the engine does not have. The same template is now
offered honestly as `powlaw`. This is the general lesson for writing a preset — name the
*family* the fitted constants span, not one member of it. `lorentz`'s two `1`s become amplitude
and width; `planck`'s `-1` can fit `+1` and reach Fermi–Dirac.

Two rules are stated in the disclosure prose rather than left to the engine's error message,
because they are what a user breaks first: `x` must appear **exactly once** (so `sin(x)/x`, the
obvious diffraction motif, is simply not expressible as a macro — nor are `coth(x) - 1/x`,
`sinh`/`cosh`, `atanh` or Yukawa's `exp(-x)/x`), and literals are fitted, not fixed.

`web/wasm/test/parity_test.cjs` §2f asserts every shipped preset is accepted by the engine —
otherwise a preset is a button that only produces an error message, and nothing in the build
would notice, since the GUI holds no copy of the grammar (§5). The test reads the bodies out of
`main.js` rather than restating them; a second copy of the list is the drift it exists to catch.

## 7. What is NOT claimed

No accuracy claim. Macros are a **capability** (a user can express a domain motif the
primitive set reaches only in several steps), verified for correctness and inertness, not
measured for recovery gain. Anyone proposing to enable a macro by default must first run the
`docs/44` / `docs/47` screen protocol — and the default would still have to stay empty to
keep PySR parity.
