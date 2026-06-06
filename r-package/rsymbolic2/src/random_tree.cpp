#include "rsymbolic/evolution/random_tree.hpp"

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

// Recursively append a subtree in postfix order. `const_index` is shared across the
// whole tree so constants receive contiguous indices.
void grow(Tree& out, int depth, const SearchSpace& space, std::mt19937_64& rng,
          int& const_index) {
    const bool has_ops = !space.unary_ops.empty() || !space.binary_ops.empty();
    const bool make_leaf =
        depth <= 0 || !has_ops || uniform01(rng) < space.terminal_prob;

    if (make_leaf) {
        const bool make_const =
            space.num_features <= 0 || uniform01(rng) < space.const_prob;
        if (make_const) {
            const double value =
                std::uniform_real_distribution<double>(space.const_min,
                                                       space.const_max)(rng);
            out.push_back(constant_node(const_index++, value));
        } else {
            std::uniform_int_distribution<int> feat(0, space.num_features - 1);
            out.push_back(variable_node(feat(rng)));
        }
        return;
    }

    const bool can_unary = !space.unary_ops.empty();
    const bool can_binary = !space.binary_ops.empty();
    const bool choose_unary =
        can_unary && (!can_binary || uniform01(rng) < 0.5);

    if (choose_unary) {
        grow(out, depth - 1, space, rng, const_index);
        out.push_back(unary_node(pick(space.unary_ops, rng)));
    } else {
        grow(out, depth - 1, space, rng, const_index);
        grow(out, depth - 1, space, rng, const_index);
        out.push_back(binary_node(pick(space.binary_ops, rng)));
    }
}

}  // namespace

Tree generate_random_tree(const SearchSpace& space, std::mt19937_64& rng) {
    Tree tree;
    int const_index = 0;
    grow(tree, space.max_depth, space, rng, const_index);
    return tree;
}

}  // namespace rsymbolic
