// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <random>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Subtree crossover: replace a random subtree in parent_a with a random subtree from
// parent_b.  The crossover point (root of the replaced subtree) and the donor point
// (root of the inserted subtree) are both drawn uniformly over all nodes in their
// respective trees.
//
// If the resulting tree would exceed max_nodes, parent_a is returned unchanged (the
// size guard acts as the standard "bloat control" in GP crossover).
//
// Constant indices are reassigned to 0..k-1 after the structural edit.
Tree subtree_crossover(const Tree& parent_a, const Tree& parent_b,
                       std::mt19937_64& rng, int max_nodes);

// Result of a two-child crossover: both children plus whether the breed succeeded.
struct CrossoverChildren {
    Tree child1;
    Tree child2;
    bool accepted = false;  // false => constraints could not be met; children are unspecified
};

// Two-child subtree crossover, faithful to SymbolicRegression.jl crossover_trees +
// crossover_generation: from a SINGLE pair of cut points (one subtree in each parent) it
// produces both children by swapping those subtrees symmetrically —
//   child1 = parent_a with its cut subtree replaced by parent_b's donor subtree,
//   child2 = parent_b with its donor subtree replaced by parent_a's cut subtree.
// The cut points are re-drawn for up to `max_tries` (10, as in SR.jl) until BOTH children
// satisfy the size cap; if none do, `accepted` is false and the caller skips the crossover
// (SR.jl skip_mutation_failures) — no member is replaced. Constant indices are reassigned to
// 0..k-1 in each child. (The RNG mechanism — uniform node index vs SR.jl's NodeSampler — is an
// allowed implementation difference per CLAUDE.md; the two-child semantics are what matter.)
CrossoverChildren subtree_crossover_pair(const Tree& parent_a, const Tree& parent_b,
                                         std::mt19937_64& rng, int max_nodes,
                                         int max_tries = 10);

}  // namespace rsymbolic
