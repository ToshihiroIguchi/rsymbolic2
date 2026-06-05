#include "rsymbolic/evolution/mutation.hpp"

#include <cmath>
#include <cstddef>
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

void randomize_tree(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    tree = generate_random_tree(space, rng);
}

void mutate(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
            double const_perturb_scale) {
    const bool has_const = !node_indices_of(tree, NodeKind::Constant).empty();
    const bool has_op = has_operator(tree);

    const double r = uniform01(rng);

    // 20% structural novelty; otherwise refine an operator or a constant.
    if (r < 0.2 || (!has_const && !has_op)) {
        randomize_tree(tree, space, rng);
        return;
    }
    if (r < 0.6 && has_op) {
        if (mutate_operator(tree, space, rng)) return;
    }
    if (has_const) {
        if (mutate_constant(tree, rng, const_perturb_scale)) return;
    }
    if (has_op && mutate_operator(tree, space, rng)) return;
    randomize_tree(tree, space, rng);
}

}  // namespace rsymbolic
