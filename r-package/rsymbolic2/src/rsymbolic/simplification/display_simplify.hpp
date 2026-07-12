// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Display-only algebraic simplification, richer than simplify() (docs/29 §11 parity
// simplifier). This is called ONLY at result finalization, on a COPY of the reported
// tree — never from the search loop, never on `best.tree`/`m.tree` in place, and never
// on anything that feeds evaluation/optimisation during evolution. It exists purely to
// make the reported `expression_simplified` / `latex_simplified` strings shorter and
// more readable; the frozen `expression` field (docs/48 D2) is untouched and remains
// the predict() round-trip source.
//
// Rule set (bottom-up, post-order; see docs/52 for the full rationale):
//   1. Constant-subtree folding (as simplify()'s fold stage), isfinite-guarded.
//   2. Add/Sub/Mul constant reassociation and commutative canonicalisation (identical
//      logic to simplify()'s combine stage, duplicated here so this file has no
//      dependency on simplify.cpp / the search-loop simplifier).
//   3. NEW: Div/Mul constant-chain folding, isfinite-guarded (not part of SR.jl's
//      combine_operators, hence not in simplify(); safe because this path is never
//      evaluated during search):
//        (t * c1) / c2 -> t * (c1/c2)      (t / c1) / c2 -> t / (c1*c2)
//        (t / c1) * c2 -> t * (c2/c1)      c1 / (t * c2) -> (c1/c2) / t
//        c1 / (t / c2) -> (c1*c2) / t
//   4. Identity elimination: t*1, 1*t, t+0, 0+t, t-0, t/1 all -> t. Deliberately
//      excludes t*0, 0*t, t^0 (would discard a NaN/Inf that t itself would have
//      produced) and t^1 (the core's pow(x,1) is NOT exactly x for x>0 — it evaluates
//      as exp(1*log(x)), which loses a rounding step relative to x — so t^1 -> t would
//      silently change the displayed value's precision; excluded, see docs/52).
//   5. Double negation: neg(neg(t)) -> t.
//   6. t * (-1) -> neg(t) (IEEE-exact; safe even when neg is outside the user's
//      requested unary_ops, since this is display-only).
//
// Applied in a fixed-point loop (bounded passes) until the tree stops changing. Like
// simplify()'s reassociation, floating-point reassociation here can shift the
// evaluated value by a rounding step (same caveat as PySR's own sympy_format
// simplification) — this function must never turn a finite value into NaN/Inf or vice
// versa; every rewrite is isfinite-guarded to preserve that.
//
// `passes_used`, when non-null, receives the number of fixed-point passes actually
// run (for tests asserting the pass cap is never exhausted). Default nullptr keeps the
// call site (result finalization) a plain single-argument call.
Tree display_simplify(const Tree& tree, int* passes_used = nullptr);

}  // namespace rsymbolic
