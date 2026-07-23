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

## 7. What is NOT claimed

No accuracy claim. Macros are a **capability** (a user can express a domain motif the
primitive set reaches only in several steps), verified for correctness and inertness, not
measured for recovery gain. Anyone proposing to enable a macro by default must first run the
`docs/44` / `docs/47` screen protocol — and the default would still have to stay empty to
keep PySR parity.
