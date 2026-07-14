// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/simplification/egraph.hpp"

namespace rsymbolic {

// Display-only algebraic simplification, richer than simplify() (docs/29 §11 parity
// simplifier). This is called ONLY at result finalization, on a COPY of the reported
// tree — never from the search loop, never on `best.tree`/`m.tree` in place, and never
// on anything that feeds evaluation/optimisation during evolution. It exists purely to
// make the reported `expression_simplified` / `latex_simplified` strings shorter and
// more readable; the frozen `expression` field (docs/48 D2) is untouched and remains
// the predict() round-trip source.
//
// Two layers (docs/54; supersedes the docs/52 rule list):
//
//   Layer 1 — deterministic normalisation (Cohen 2003 "automatic simplification"
//   style), always runs: isfinite-guarded constant folding; flattening of +/-/neg
//   chains into an n-ary sum and of *//neg chains into a numerator/denominator
//   product; collection of constants, like terms (c1*t + c2*t -> (c1+c2)*t) and like
//   factors (t*t -> square(t), which is bit-exact because square evaluates as t*t);
//   canonical operand ordering; negation normal form (neg(neg t) -> t,
//   neg(a-b) -> b-a, sign absorption into constants); identity elimination (t+0, t*1,
//   t/1, t*-1 -> neg t); and a small set of exact unary rewrites (abs/square of
//   sign-invariant arguments, odd/even functions of neg, sqrt(square t) -> abs t).
//
//   Layer 2 — bounded equality saturation (egraph.hpp): the Layer-1 form seeds an
//   e-graph, an audited rule set saturates within hard limits (iterations / e-nodes /
//   wall-clock; EGraphLimits), and the minimum-node-count equivalent is extracted.
//   The result is adopted only when it is STRICTLY smaller than the Layer-1 form;
//   otherwise — including when any limit was hit — the Layer-1 result stands (the
//   fallback contract). Layer 2 is what finds cross-structure reductions Layer 1
//   cannot, e.g. factoring x*y + x*z -> x*(y+z).
//
// Floating-point policy (docs/54): every rewrite is an exact identity over the reals
// on the expression's domain; rewrites that reassociate/redistribute may shift the
// evaluated value by a rounding step (the same caveat simplify()'s combine pass and
// PySR's sympy display path carry), but no rewrite may change whether the expression
// evaluates to NaN/Inf — so x*0 -> 0, exp(log x) -> x, log(exp x) -> x, pow rewrites,
// and term cancellation to 0 are all excluded, and constant folding is isfinite-
// guarded. The displayed value therefore never diverges from predict() beyond
// rounding drift. Node count never increases.
//
// `stats` (optional) reports what Layer 2 did — for tests and diagnostics only.
// `limits` bounds Layer 2; `limits.max_iterations = 0` disables Layer 2 entirely
// (pure Layer-1 normalisation). The defaults keep the per-expression cost at a few
// milliseconds. Existing one-argument call sites are unchanged.
struct DisplaySimplifyStats {
    int egraph_iterations = 0;
    int egraph_enodes = 0;
    bool egraph_saturated = false;
    bool layer2_adopted = false;  // extraction was strictly smaller than Layer 1
};

Tree display_simplify(const Tree& tree, DisplaySimplifyStats* stats = nullptr,
                      const EGraphLimits& limits = EGraphLimits());

}  // namespace rsymbolic
