// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Algebraic simplification of an expression tree, matching SymbolicRegression.jl /
// DynamicExpressions for PySR default parity (CLAUDE.md "PySR Default Parity"; docs/29
// §11). Two passes, applied in this order — the order matters, folding first lets the
// reassociation pass see fully-reduced constant operands:
//   1. fold (SR.jl `simplify_tree!`): a node whose children are all constants collapses
//      to one constant, when the result is finite (SR.jl's `is_valid` guard).
//   2. combine_operators (SR.jl `combine_operators`): reassociate/merge a constant into a
//      same-operator child, and canonicalise commutative operators with the constant on
//      the right:
//        commutative (+,*): (c1 ∘ x) ∘ c2 -> (op(c1,c2) ∘ x);  c + x -> x + c
//        subtraction (-):   four rewrites that fold a nested constant, e.g.
//                           (c - (var - c2)) -> ((c+c2) - var)
//
// Deliberately does NOT eliminate identities (x+0, x*1, x*0, x/1, x^0, x^1), cancel
// double negation, or rewrite x^2 -> square: SR.jl performs none of these, so neither do
// we (parity). Constant nodes in the result are re-indexed to a contiguous 0..k-1 range.
// The result evaluates to the same value as the input under exact arithmetic; in floating
// point the reassociation in pass 2 (like SR.jl's) may shift the value by a rounding step.
//
// This single entry point backs both simplification sites, mirroring SR.jl: the
// `simplify` mutation and the per-iteration population pass both call the same pair.
Tree simplify(const Tree& tree);

}  // namespace rsymbolic
