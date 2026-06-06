#include "rsymbolic/evolution/mutation.hpp"

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "rsymbolic/evolution/random_tree.hpp"

namespace rsymbolic {

namespace {

double uniform01(std::mt19937_64& rng) {
    return std::uniform_real_distribution<double>(0.0, 1.0)(rng);
}

std::vector<std::size_t> node_indices_of(const Tree& tree, NodeKind kind) {
    std::vector<std::size_t> positions;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (tree[i].kind == kind) positions.push_back(i);
    }
    return positions;
}

bool has_operator(const Tree& tree) {
    for (const Node& n : tree) {
        if (n.kind == NodeKind::Unary || n.kind == NodeKind::Binary) return true;
    }
    return false;
}

}  // namespace

bool mutate_constant(Tree& tree, std::mt19937_64& rng, double scale) {
    const std::vector<std::size_t> positions =
        node_indices_of(tree, NodeKind::Constant);
    if (positions.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, positions.size() - 1);
    Node& node = tree[positions[pick(rng)]];

    std::normal_distribution<double> gaussian(0.0, 1.0);
    node.value += gaussian(rng) * scale * (std::fabs(node.value) + 1.0);
    return true;
}

bool mutate_operator(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    std::vector<std::size_t> positions;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (tree[i].kind == NodeKind::Unary || tree[i].kind == NodeKind::Binary) {
            positions.push_back(i);
        }
    }
    if (positions.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, positions.size() - 1);
    Node& node = tree[positions[pick(rng)]];

    if (node.kind == NodeKind::Unary) {
        if (space.unary_ops.empty()) return false;
        std::uniform_int_distribution<std::size_t> op(0, space.unary_ops.size() - 1);
        node.uop = space.unary_ops[op(rng)];
    } else {
        if (space.binary_ops.empty()) return false;
        std::uniform_int_distribution<std::size_t> op(0, space.binary_ops.size() - 1);
        node.bop = space.binary_ops[op(rng)];
    }
    return true;
}

namespace {

Node random_leaf(const SearchSpace& space, std::mt19937_64& rng) {
    const bool make_const =
        space.num_features <= 0 || uniform01(rng) < space.const_prob;
    if (make_const) {
        const double value = std::uniform_real_distribution<double>(
            space.const_min, space.const_max)(rng);
        return constant_node(0, value);  // index fixed by reindex_constants
    }
    std::uniform_int_distribution<int> feat(0, space.num_features - 1);
    return variable_node(feat(rng));
}

}  // namespace

bool append_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    if (static_cast<int>(tree.size()) >= space.max_nodes) return false;

    const std::vector<std::size_t> leaves = [&] {
        std::vector<std::size_t> out;
        for (std::size_t i = 0; i < tree.size(); ++i) {
            if (arity(tree[i]) == 0) out.push_back(i);
        }
        return out;
    }();
    if (leaves.empty()) return false;

    const bool can_unary = !space.unary_ops.empty();
    const bool can_binary = !space.binary_ops.empty();
    if (!can_unary && !can_binary) return false;

    std::uniform_int_distribution<std::size_t> pick_leaf(0, leaves.size() - 1);
    const std::size_t p = leaves[pick_leaf(rng)];
    const Node leaf = tree[p];

    Tree replacement;
    const bool use_unary = can_unary && (!can_binary || uniform01(rng) < 0.5);
    if (use_unary) {
        std::uniform_int_distribution<std::size_t> op(0, space.unary_ops.size() - 1);
        replacement = {leaf, unary_node(space.unary_ops[op(rng)])};
    } else {
        std::uniform_int_distribution<std::size_t> op(0, space.binary_ops.size() - 1);
        const Node binop = binary_node(space.binary_ops[op(rng)]);
        const Node other = random_leaf(space, rng);
        if (uniform01(rng) < 0.5) {
            replacement = {leaf, other, binop};
        } else {
            replacement = {other, leaf, binop};
        }
    }

    Tree result;
    result.reserve(tree.size() + replacement.size());
    result.insert(result.end(), tree.begin(), tree.begin() + p);
    result.insert(result.end(), replacement.begin(), replacement.end());
    result.insert(result.end(), tree.begin() + p + 1, tree.end());
    tree = std::move(result);
    reindex_constants(tree);
    return true;
}

bool delete_random_op(Tree& tree, std::mt19937_64& rng) {
    std::vector<std::size_t> ops;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (arity(tree[i]) >= 1) ops.push_back(i);
    }
    if (ops.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, ops.size() - 1);
    const std::size_t i = ops[pick(rng)];
    const std::size_t start = subtree_start(tree, i);

    std::size_t child_start = 0;
    std::size_t child_end = 0;
    if (tree[i].kind == NodeKind::Unary) {
        child_end = i - 1;
        child_start = subtree_start(tree, child_end);
    } else {  // Binary: choose left or right child
        const std::size_t right_end = i - 1;
        const std::size_t right_start = subtree_start(tree, right_end);
        const std::size_t left_end = right_start - 1;
        const std::size_t left_start = subtree_start(tree, left_end);
        if (uniform01(rng) < 0.5) {
            child_start = left_start;
            child_end = left_end;
        } else {
            child_start = right_start;
            child_end = right_end;
        }
    }

    Tree result;
    result.reserve(tree.size());
    result.insert(result.end(), tree.begin(), tree.begin() + start);
    result.insert(result.end(), tree.begin() + child_start,
                  tree.begin() + child_end + 1);
    result.insert(result.end(), tree.begin() + i + 1, tree.end());
    tree = std::move(result);
    reindex_constants(tree);
    return true;
}

void randomize_tree(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    tree = generate_random_tree(space, rng);
}

void mutate(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
            double const_perturb_scale) {
    const bool has_const = !node_indices_of(tree, NodeKind::Constant).empty();
    const bool has_op = has_operator(tree);
    const bool can_append = static_cast<int>(tree.size()) < space.max_nodes;

    // Weighted menu over the feasible mutation kinds. Structural mutations (append /
    // delete) let topology evolve locally; randomize is the smaller-weight escape.
    enum Kind { kConstant, kOperator, kAppend, kDelete, kRandomize };
    std::vector<std::pair<double, Kind>> menu;
    if (has_const) menu.emplace_back(0.20, kConstant);
    if (has_op) menu.emplace_back(0.20, kOperator);
    if (can_append) menu.emplace_back(0.25, kAppend);
    if (has_op) menu.emplace_back(0.20, kDelete);
    menu.emplace_back(0.15, kRandomize);

    double total = 0.0;
    for (const auto& entry : menu) total += entry.first;
    double r = uniform01(rng) * total;
    Kind choice = kRandomize;
    for (const auto& entry : menu) {
        if (r < entry.first) {
            choice = entry.second;
            break;
        }
        r -= entry.first;
    }

    bool ok = false;
    switch (choice) {
        case kConstant:
            ok = mutate_constant(tree, rng, const_perturb_scale);
            break;
        case kOperator:
            ok = mutate_operator(tree, space, rng);
            break;
        case kAppend:
            ok = append_random_op(tree, space, rng);
            break;
        case kDelete:
            ok = delete_random_op(tree, rng);
            break;
        case kRandomize:
            randomize_tree(tree, space, rng);
            ok = true;
            break;
    }
    if (!ok) randomize_tree(tree, space, rng);
}

}  // namespace rsymbolic
