// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/evolution/random_tree.hpp"

#include <random>

#include "rsymbolic/evolution/mutation.hpp"

namespace rsymbolic {

namespace {

double uniform01(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

template <typename T>
const T& pick(const std::vector<T>& options, std::mt19937_64& rng) {
    std::uniform_int_distribution<std::size_t> dist(0, options.size() - 1);
    return options[dist(rng)];
}

// Recursively append a subtree in postfix order. Leaves come from make_random_leaf; the
// whole tree is re-indexed once by the caller (generate_random_tree).
void grow(Tree& out, int depth, const SearchSpace& space, std::mt19937_64& rng) {
    const bool has_ops = !space.unary_ops.empty() || !space.binary_ops.empty();
    // Honour the node cap during generation. mutation (append/crossover) already
    // respects max_nodes; generation did not, so an initial tree could reach the
    // depth-bounded maximum (~2^max_depth nodes) — far larger than evolution ever
    // permits. Once the cap is reached, force leaves. Like append/crossover this is a
    // soft cap: operator nodes pending along the current spine may push the final size a
    // few nodes past max_nodes, which is acceptable.
    const bool budget_hit =
        space.max_nodes > 0 && static_cast<int>(out.size()) >= space.max_nodes;
    const bool make_leaf =
        depth <= 0 || !has_ops || budget_hit || uniform01(rng) < space.terminal_prob;

    if (make_leaf) {
        out.push_back(make_random_leaf(space, rng));
        return;
    }

    const bool can_unary = !space.unary_ops.empty();
    const bool can_binary = !space.binary_ops.empty();
    const bool choose_unary =
        can_unary && (!can_binary || uniform01(rng) < 0.5);

    if (choose_unary) {
        grow(out, depth - 1, space, rng);
        out.push_back(unary_node(pick(space.unary_ops, rng)));
    } else {
        grow(out, depth - 1, space, rng);
        grow(out, depth - 1, space, rng);
        out.push_back(binary_node(pick(space.binary_ops, rng)));
    }
}

}  // namespace

Node make_random_leaf(const SearchSpace& space, std::mt19937_64& rng) {
    // SR.jl make_random_leaf chooses a constant or a feature 50/50 (`rand(rng, Bool)`); here
    // the split is governed by const_prob (defaulted to 0.5 for that parity). The num_features
    // guard forces a constant when the problem has no input variables. The constant carries
    // index 0; reindex_constants assigns contiguous indices once the tree is assembled.
    const bool make_const =
        space.num_features <= 0 || uniform01(rng) < space.const_prob;
    if (make_const) {
        const double value = std::uniform_real_distribution<double>(
            space.const_min, space.const_max)(rng);
        return constant_node(0, value);
    }
    std::uniform_int_distribution<int> feat(0, space.num_features - 1);
    return variable_node(feat(rng));
}

Tree gen_random_tree(int nlength, const SearchSpace& space, std::mt19937_64& rng) {
    // SR.jl gen_random_tree: a single placeholder leaf, replaced and grown by `nlength`
    // append_random_op calls. The placeholder's value is irrelevant — the first append
    // discards it — and it draws no RNG, matching SR.jl's `val=init_value(T)` placeholder.
    Tree tree{constant_node(0, 0.0)};
    for (int i = 0; i < nlength; ++i) {
        append_random_op(tree, space, rng);
    }
    return tree;
}

Tree gen_random_tree_fixed_size(int node_count, const SearchSpace& space,
                                std::mt19937_64& rng) {
    // SR.jl gen_random_tree_fixed_size: start from a random leaf and append operators until
    // the target node count is reached. On the penultimate step (cur_size == node_count - 1)
    // only a unary op can land exactly on the target; if no unary op exists we break and
    // accept being one node short (SR.jl's `nuna == 0 && break`).
    Tree tree{make_random_leaf(space, rng)};
    int cur_size = static_cast<int>(tree.size());
    while (cur_size < node_count) {
        if (cur_size == node_count - 1) {
            if (space.unary_ops.empty()) break;
            append_random_op(tree, space, rng, /*make_new_bin_op=*/false);
        } else {
            append_random_op(tree, space, rng);
        }
        cur_size = static_cast<int>(tree.size());
    }
    return tree;
}

Tree generate_random_tree(const SearchSpace& space, std::mt19937_64& rng) {
    // General-purpose, depth-driven random tree generator producing trees of DIVERSE sizes
    // and depths. NOTE: this is NOT the production population/randomize generator — the
    // search seeds its initial population with gen_random_tree(3) and randomizes with
    // gen_random_tree_fixed_size, both faithful to SymbolicRegression.jl. This generator is
    // retained for tests and benchmarks that need varied trees for stress coverage.
    Tree tree;
    grow(tree, space.max_depth, space, rng);
    reindex_constants(tree);
    return tree;
}

}  // namespace rsymbolic
