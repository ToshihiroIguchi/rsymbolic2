// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <random>

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Make a single random leaf, faithful to SymbolicRegression.jl make_random_leaf: a
// constant or a variable, chosen 50/50 (here governed by space.const_prob, defaulted to
// 0.5 for SR.jl parity). Constant nodes are emitted with index 0; callers that assemble a
// multi-constant tree must re-index afterwards (reindex_constants). Shared by the random
// generators below and by the structural mutation operators (append/prepend/insert/delete).
Node make_random_leaf(const SearchSpace& space, std::mt19937_64& rng);

// Generate the small initial-population tree, faithful to SymbolicRegression.jl
// gen_random_tree(nlength): start from a placeholder leaf and apply append_random_op exactly
// `nlength` times, yielding a tree with `nlength` operators (a few nodes). SR.jl hardcodes
// nlength=3 for the initial population. Constants are contiguously indexed.
Tree gen_random_tree(int nlength, const SearchSpace& space, std::mt19937_64& rng);

// Generate a random tree of approximately `node_count` nodes, faithful to
// SymbolicRegression.jl gen_random_tree_fixed_size: start from a random leaf and grow by
// append_random_op until the node count is reached, forcing a unary operator on the last
// step so the size lands exactly on `node_count` (when a unary op exists). Used by the
// randomize mutation with node_count drawn from 1..curmaxsize. Constants are contiguously
// indexed.
Tree gen_random_tree_fixed_size(int node_count, const SearchSpace& space,
                                std::mt19937_64& rng);

// General-purpose, depth-driven random tree generator producing trees of DIVERSE sizes and
// depths, with contiguously indexed constants. This is NOT the production population/
// randomize generator (those use gen_random_tree / gen_random_tree_fixed_size, faithful to
// SymbolicRegression.jl). Retained for tests and benchmarks that need varied trees.
Tree generate_random_tree(const SearchSpace& space, std::mt19937_64& rng);

}  // namespace rsymbolic
