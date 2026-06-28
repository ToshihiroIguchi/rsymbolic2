// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <optional>
#include <random>

#include "rsymbolic/evolution/mutation_weights.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Perturb a randomly chosen constant's value. Returns false if there are no constants.
// Structure and parameter indices are preserved. Uses PySR's multiplicative kernel
// (SymbolicRegression.jl mutate_factor): the constant is multiplied by a random factor
// derived from perturbation_factor, with a sign flip governed by
// probability_negate_constant (defaulted to PySR's 0.00743 so existing callers are
// unaffected). See mutation.cpp for the exact, intentionally faithful formula.
bool mutate_constant(Tree& tree, std::mt19937_64& rng, double perturbation_factor,
                     double probability_negate_constant = 0.00743);

// Replace a randomly chosen operator with another of the same arity from the search
// space. Returns false if there are no operator nodes. Structure is preserved.
bool mutate_operator(Tree& tree, const SearchSpace& space, std::mt19937_64& rng);

// Grow a randomly chosen leaf into an operator. Faithful to SymbolicRegression.jl
// append_random_op: the chosen leaf is REPLACED by a new operator over FRESH random leaves
// (the original leaf is discarded, not reused as an operand). The operator is binary with
// probability nbin/(nuna+nbin) (SR.jl's make_new_bin_op), else unary. `make_new_bin_op`
// overrides that choice when set (SR.jl's keyword of the same name): faithful to SR.jl's
// `@something`, the binary/unary random draw is then NOT taken, so the RNG stream is
// unperturbed. Returns false if the size cap leaves no room (binary needs 2 free slots,
// unary 1) or the forced/feasible operator kind is unavailable. Constants are re-indexed.
bool append_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
                      std::optional<bool> make_new_bin_op = std::nullopt);

// Grow the tree at the ROOT: wrap the entire tree as the (left) operand of a new operator,
// with a fresh random leaf as the right operand for a binary op. Faithful to
// SymbolicRegression.jl prepend_random_op; pairs with append_random_op to form SR.jl's
// add_node mutation (chosen 50/50). Binary with probability nbin/(nuna+nbin). Returns false
// if the size cap leaves no room. Constants are re-indexed.
bool prepend_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng);

// Splice out a randomly chosen node, faithful to SymbolicRegression.jl delete_random_op!:
// pick a random operator as the parent, then one of its operand subtrees as the target. If
// the target is a leaf it is replaced by a fresh random leaf (not a deletion); if it is a
// unary/binary operator one of its children is spliced into its place. Returns false if
// there are no operator nodes. Constants are re-indexed.
bool delete_random_op(const SearchSpace& space, Tree& tree, std::mt19937_64& rng);

// Wrap a randomly chosen subtree (rooted at any node, not only a leaf) in a new operator.
// Faithful to SymbolicRegression.jl insert_random_op: the wrapped subtree is ALWAYS the
// left operand of the new binary op (with a fresh random leaf on the right), or the operand
// of a new unary op. Binary with probability nbin/(nuna+nbin). The wrapped node may be an
// interior operator, so structure can grow in the middle of the tree. Returns false if the
// size cap leaves no room or no operators are available. Constants are re-indexed.
bool insert_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng);

// Rotate a randomly chosen valid rotation node, faithful to SymbolicRegression.jl
// randomly_rotate_tree!. A node is a valid rotation node when (degree>0 && left.degree>0)
// or (degree==2 && right.degree>0), which INCLUDES unary nodes (via the leftmost/rightmost
// fallback), so rotations involving unary operators are performed too. The node count is
// preserved. This is a purely structural move and may change the expression's value (the
// candidate is re-evaluated afterwards). Returns false if there is no valid rotation node.
// Constants are re-indexed.
bool rotate_tree(Tree& tree, std::mt19937_64& rng);

// Swap the two operand subtrees of a randomly chosen binary node. Returns false if
// there are no binary nodes. Constants are re-indexed.
bool swap_operands(Tree& tree, std::mt19937_64& rng);

// Replace the entire tree with a freshly generated random tree. Always succeeds and is
// a source of structural novelty.
void randomize_tree(Tree& tree, const SearchSpace& space, std::mt19937_64& rng);

// Apply a single mutation, sampled from the kinds that are feasible for `tree` in
// proportion to `weights` (see MutationWeights). do_nothing / simplify / randomize are
// always feasible; the structural kinds depend on the tree's content and the size cap.
// Returns true if a mutation was applied (the tree may be unchanged for do_nothing),
// false if the sampled kind turned out to be infeasible at apply time and produced no
// change — the caller treats this as SR.jl's `skip_mutation_failures` no-op cycle. The
// trailing arguments are defaulted so existing 4-argument callers keep PySR-default
// behaviour; probability_negate_constant is forwarded to mutate_constant.
bool mutate(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
            double const_perturb_scale = 0.5,
            const MutationWeights& weights = MutationWeights{},
            double probability_negate_constant = 0.00743);

}  // namespace rsymbolic
