// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

namespace rsymbolic {

// Relative weights for the weighted mutation menu, mirroring PySR /
// SymbolicRegression.jl's `MutationWeights`. When mutate() draws a mutation it
// considers only the kinds that are *feasible* for the current tree (e.g. rotate
// needs a binary node with a binary child) and samples among them in proportion to
// these weights. Weights are relative; they need not sum to one.
//
// Defaults are PySR's *installed* `__init__` weight_* defaults (pysr/sr.py 1.5.10),
// which override SymbolicRegression.jl's own MutationWeights struct defaults — so a
// default-constructed SearchOptions reproduces what `PySRRegressor()` actually runs
// (docs/28 §A). Weights are relative (SR.jl normalises them to sum 1 before sampling);
// only the ratios matter. Two deliberate deviations from the PySR set, both documented
// in docs/27:
//   - `optimize` is omitted. PySR ships it at weight 0.0 (off), and rsymbolic2
//     already controls constant optimisation through SearchOptions.optimize_probability
//     (a measured mechanism, docs/13). Adding an optimise *mutation* would duplicate
//     that control point.
//   - `form_connection` / `break_connection` are omitted: they apply only to PySR's
//     expression-graph mode, not the default tree representation used here.
struct MutationWeights {
    double mutate_constant = 0.0346;   // perturb one constant's value
    double mutate_operator = 0.293;    // swap one operator for another of equal arity
    double swap_operands   = 0.198;    // swap a binary node's two operand subtrees
    double rotate_tree     = 4.26;     // associativity-style rotation of a binary node
    double add_node        = 2.47;     // grow a leaf into an operator (append_random_op)
    double insert_node     = 0.0112;   // wrap an arbitrary subtree in a new operator
    double delete_node     = 0.870;    // remove an operator, splicing in one child
    double do_nothing      = 0.273;    // leave the inherited tree unchanged
    double simplify        = 0.00209;  // apply the algebraic simplifier
    double randomize       = 0.000502; // replace the whole tree with a fresh random one
};

}  // namespace rsymbolic
