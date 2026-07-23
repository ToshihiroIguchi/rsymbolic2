// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/evolution/mutation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/simplification/simplify.hpp"

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

bool has_binary(const Tree& tree) {
    for (const Node& n : tree) {
        if (n.kind == NodeKind::Binary) return true;
    }
    return false;
}

// Spans of the two operand subtrees of the binary node at position `i` (postfix:
// [left ...][right ...][op@i]). Requires tree[i] to be a Binary node.
struct BinaryChildren {
    std::size_t left_start, left_end, right_start, right_end;
};
BinaryChildren binary_children(const Tree& tree, std::size_t i) {
    const std::size_t right_end   = i - 1;
    const std::size_t right_start = subtree_start(tree, right_end);
    const std::size_t left_end    = right_start - 1;
    const std::size_t left_start  = subtree_start(tree, left_end);
    return {left_start, left_end, right_start, right_end};
}

// --- unary alphabet (primitive operators + opt-in macro operators) -------------------
//
// A growth mutation that creates a unary node draws from the enabled primitive operators
// (one node each) followed by the macro operators that still fit in the free space
// (macro_extra_nodes each; macro_op.hpp). With space.macro_ops empty — the default — the
// alphabet IS space.unary_ops and the single index draw below is the identical RNG call
// the pre-macro code made. That is what keeps the default search bit-identical.
//
// Size/at rather than a materialised vector: this is the hot path, and the no-macro case
// must stay allocation-free.

struct UnaryChoice {
    bool is_macro = false;
    UnaryOp op = UnaryOp::Neg;
    std::size_t macro_index = 0;
};

// `primitives_only` restricts the alphabet to cost-one entries. Used by the forced-unary
// step of gen_random_tree_fixed_size, whose size arithmetic assumes a unary operator adds
// exactly one node — a macro would overshoot the requested tree size.
std::size_t unary_alphabet_size(const SearchSpace& space, int room, bool primitives_only) {
    std::size_t n = room >= 1 ? space.unary_ops.size() : 0;
    if (primitives_only) return n;
    for (const MacroOp& m : space.macro_ops) {
        if (macro_extra_nodes(m) <= room) ++n;
    }
    return n;
}

UnaryChoice unary_alphabet_at(const SearchSpace& space, int room, bool primitives_only,
                              std::size_t index) {
    const std::size_t n_primitive = room >= 1 ? space.unary_ops.size() : 0;
    if (index < n_primitive) return {false, space.unary_ops[index], 0};
    std::size_t k = index - n_primitive;
    for (std::size_t i = 0; i < space.macro_ops.size() && !primitives_only; ++i) {
        if (macro_extra_nodes(space.macro_ops[i]) > room) continue;
        if (k == 0) return {true, UnaryOp::Neg, i};
        --k;
    }
    return {false, space.unary_ops.empty() ? UnaryOp::Neg : space.unary_ops[0], 0};
}

UnaryChoice draw_unary(const SearchSpace& space, int room, bool primitives_only,
                       std::mt19937_64& rng) {
    const std::size_t n = unary_alphabet_size(space, room, primitives_only);
    std::uniform_int_distribution<std::size_t> op(0, n - 1);
    return unary_alphabet_at(space, room, primitives_only, op(rng));
}

// Wrap `operand` (a complete postfix subtree) in the chosen unary form: one operator node
// for a primitive, the expanded template for a macro.
Tree wrap_unary(const UnaryChoice& choice, const SearchSpace& space, Tree operand) {
    if (choice.is_macro) return expand_macro(space.macro_ops[choice.macro_index], operand);
    operand.push_back(unary_node(choice.op));
    return operand;
}

// Probability that a newly grown operator is binary, faithful to SR.jl
// MutationFunctions.jl: `rand() < nbin / (nuna + nbin)`. This weights the unary/binary
// split by the operator-set sizes rather than a fixed 50/50, so larger unary sets bias
// growth toward unary operators exactly as the reference does. `nuna` is the unary
// alphabet size, so opt-in macros widen the unary side exactly as extra primitives would;
// with no macros it is space.unary_ops.size(), the SR.jl-faithful value.
double binary_op_fraction(const SearchSpace& space, std::size_t n_unary) {
    const double nbin = static_cast<double>(space.binary_ops.size());
    const double nuna = static_cast<double>(n_unary);
    const double total = nbin + nuna;
    return total > 0.0 ? nbin / total : 0.0;
}

}  // namespace

bool mutate_constant(Tree& tree, std::mt19937_64& rng, double perturbation_factor,
                     double probability_negate_constant) {
    const std::vector<std::size_t> positions =
        node_indices_of(tree, NodeKind::Constant);
    if (positions.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, positions.size() - 1);
    Node& node = tree[positions[pick(rng)]];

    // PySR-parity multiplicative perturbation, faithful to SymbolicRegression.jl 1.11.0
    // MutationFunctions.jl::mutate_factor (L134-146): the constant is multiplied by a
    // random factor, not shifted additively.
    //   maxChange = perturbation_factor * temperature + 1 + 0.1
    //   factor    = maxChange ^ U[0,1)         (50% chance replaced by its reciprocal)
    //   negate    when U[0,1) > probability_negate_constant
    // temperature is fixed at 1.0: SR.jl's SingleIteration.jl (L31-36) holds it at 1.0
    // when annealing is off, which is PySR's default and the only mode rsymbolic2 has.
    constexpr double temperature = 1.0;
    const double max_change = perturbation_factor * temperature + 1.0 + 0.1;
    double factor = std::pow(max_change, uniform01(rng));
    if (uniform01(rng) < 0.5) factor = 1.0 / factor;
    // Faithful to SR.jl: the sign is flipped when the draw EXCEEDS the (small) negate
    // probability, i.e. on ~(1 - probability_negate_constant) of calls. This looks
    // inverted versus sr.py's "probability of negating" docstring but matches the
    // installed engine exactly; it is intentional, not a bug. The post-mutation LM
    // constant optimiser refines magnitude and sign, so the search does not break.
    if (uniform01(rng) > probability_negate_constant) factor = -factor;
    node.value *= factor;
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

bool append_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
                      std::optional<bool> make_new_bin_op) {
    // Net node growth: leaf -> op(leaf) adds 1; leaf -> op(leaf, leaf) adds 2. A macro
    // operator adds macro_extra_nodes(m) and is offered only while that fits in `room`.
    const int room = space.max_nodes - static_cast<int>(tree.size());
    // A forced-unary caller (gen_random_tree_fixed_size) needs exactly one added node, so
    // it draws from the primitives only — see unary_alphabet_size.
    const bool forced_unary = make_new_bin_op.has_value() && !*make_new_bin_op;
    const std::size_t n_unary = unary_alphabet_size(space, room, forced_unary);
    const bool can_unary  = n_unary > 0;
    const bool can_binary = !space.binary_ops.empty() && room >= 2;
    if (!can_unary && !can_binary) return false;

    std::vector<std::size_t> leaves;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (arity(tree[i]) == 0) leaves.push_back(i);
    }
    if (leaves.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick_leaf(0, leaves.size() - 1);
    const std::size_t p = leaves[pick_leaf(rng)];

    // SR.jl append_random_op: the chosen leaf is REPLACED by a new operator whose operands
    // are fresh random leaves (`set_node!(node, op(make_random_leaf, make_random_leaf))`).
    // The original leaf is discarded, not reused as one of the operands.
    // `make_new_bin_op` (SR.jl keyword): when set it overrides the binary/unary choice and,
    // faithful to SR.jl's `@something`, the random draw is skipped (RNG stream unperturbed).
    // A forced kind that is infeasible (no op of that arity, or no room) cannot be honoured.
    bool make_binary;
    if (make_new_bin_op.has_value()) {
        make_binary = *make_new_bin_op;
        if (make_binary ? !can_binary : !can_unary) return false;
    } else {
        make_binary =
            can_binary && (!can_unary || uniform01(rng) < binary_op_fraction(space, n_unary));
    }
    Tree replacement;
    if (make_binary) {
        std::uniform_int_distribution<std::size_t> op(0, space.binary_ops.size() - 1);
        replacement = {make_random_leaf(space, rng), make_random_leaf(space, rng),
                       binary_node(space.binary_ops[op(rng)])};
    } else {
        // RNG order is load-bearing: the leaf is drawn BEFORE the operator index, exactly as
        // the braced initialiser did before macros existed.
        Tree operand{make_random_leaf(space, rng)};
        replacement = wrap_unary(draw_unary(space, room, forced_unary, rng), space,
                                 std::move(operand));
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

bool prepend_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    // Net node growth: tree -> op(tree) adds 1; tree -> op(tree, leaf) adds 2 (a macro adds
    // macro_extra_nodes(m)).
    const int room = space.max_nodes - static_cast<int>(tree.size());
    const std::size_t n_unary = unary_alphabet_size(space, room, /*primitives_only=*/false);
    const bool can_unary  = n_unary > 0;
    const bool can_binary = !space.binary_ops.empty() && room >= 2;
    if (!can_unary && !can_binary) return false;
    if (tree.empty()) return false;

    // SR.jl prepend_random_op: wrap the ENTIRE tree at the root. The whole tree becomes the
    // (left) operand of a new operator; for a binary op a fresh random leaf is the right
    // operand (`l=copy(tree), r=make_random_leaf`). Postfix: append the right leaf (if any)
    // then the new operator after the whole existing tree.
    const bool make_binary =
        can_binary && (!can_unary || uniform01(rng) < binary_op_fraction(space, n_unary));
    if (make_binary) {
        std::uniform_int_distribution<std::size_t> op(0, space.binary_ops.size() - 1);
        tree.push_back(make_random_leaf(space, rng));
        tree.push_back(binary_node(space.binary_ops[op(rng)]));
    } else {
        // The whole tree is the operand, so a macro wraps it by splicing it into the
        // template's placeholder position.
        tree = wrap_unary(draw_unary(space, room, /*primitives_only=*/false, rng), space,
                          std::move(tree));
    }
    reindex_constants(tree);
    return true;
}

bool delete_random_op(const SearchSpace& space, Tree& tree, std::mt19937_64& rng) {
    // SR.jl delete_random_op! via random_node_and_parent: pick a random OPERATOR as the
    // parent, then one of its operand subtrees as the target node. The action then depends
    // on the target's arity:
    //   - leaf  (arity 0): REPLACE it with a fresh random leaf (this is not a deletion).
    //   - unary (arity 1): splice in its single child (drop the unary operator).
    //   - binary(arity 2): splice in one of its two children, chosen 50/50.
    // (SR.jl's isroot branch only triggers for a single-leaf tree, where delete is not
    // offered, so it is intentionally omitted here.)
    std::vector<std::size_t> parents;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (arity(tree[i]) >= 1) parents.push_back(i);
    }
    if (parents.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, parents.size() - 1);
    const std::size_t parent = parents[pick(rng)];

    // The operand subtree of `parent` that becomes the target node.
    std::size_t child_start = 0;
    std::size_t child_end = 0;
    if (tree[parent].kind == NodeKind::Unary) {
        child_end = parent - 1;
        child_start = subtree_start(tree, child_end);
    } else {  // Binary: left child for parent.l, right child for parent.r (50/50).
        const std::size_t right_end = parent - 1;
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

    // Build the replacement for the target span [child_start, child_end].
    Tree repl;
    if (arity(tree[child_end]) == 0) {
        repl = {make_random_leaf(space, rng)};  // leaf -> fresh random leaf
    } else if (tree[child_end].kind == NodeKind::Unary) {
        const std::size_t gc_end = child_end - 1;
        const std::size_t gc_start = subtree_start(tree, gc_end);
        repl.insert(repl.end(), tree.begin() + gc_start, tree.begin() + gc_end + 1);
    } else {  // binary target: splice in one of its children (50/50)
        const std::size_t r_end = child_end - 1;
        const std::size_t r_start = subtree_start(tree, r_end);
        const std::size_t l_end = r_start - 1;
        const std::size_t l_start = subtree_start(tree, l_end);
        if (uniform01(rng) < 0.5) {
            repl.insert(repl.end(), tree.begin() + l_start, tree.begin() + l_end + 1);
        } else {
            repl.insert(repl.end(), tree.begin() + r_start, tree.begin() + r_end + 1);
        }
    }

    Tree result;
    result.reserve(tree.size());
    result.insert(result.end(), tree.begin(), tree.begin() + child_start);
    result.insert(result.end(), repl.begin(), repl.end());
    result.insert(result.end(), tree.begin() + child_end + 1, tree.end());
    tree = std::move(result);
    reindex_constants(tree);
    return true;
}

bool insert_random_op(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    const int room = space.max_nodes - static_cast<int>(tree.size());
    const std::size_t n_unary =  // a primitive adds 1 node, a macro macro_extra_nodes(m)
        unary_alphabet_size(space, room, /*primitives_only=*/false);
    const bool can_unary  = n_unary > 0;
    const bool can_binary = !space.binary_ops.empty() && room >= 2;  // adds 2 nodes
    if (!can_unary && !can_binary) return false;
    if (tree.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick_node(0, tree.size() - 1);
    const std::size_t p = pick_node(rng);            // root of the subtree to wrap
    const std::size_t s = subtree_start(tree, p);    // subtree occupies [s, p]

    const bool make_binary =
        can_binary && (!can_unary || uniform01(rng) < binary_op_fraction(space, n_unary));

    Tree wrapped;  // the replacement for the span [s, p]
    wrapped.reserve((p - s + 1) + 2);
    if (make_binary) {
        // SR.jl insert_random_op: the wrapped subtree is ALWAYS the left operand of the new
        // binary op (`l=copy(node), r=make_random_leaf`); the random leaf is on the right.
        std::uniform_int_distribution<std::size_t> op(0, space.binary_ops.size() - 1);
        wrapped.insert(wrapped.end(), tree.begin() + s, tree.begin() + p + 1);
        wrapped.push_back(make_random_leaf(space, rng));
        wrapped.push_back(binary_node(space.binary_ops[op(rng)]));
    } else {
        const UnaryChoice choice = draw_unary(space, room, /*primitives_only=*/false, rng);
        wrapped.insert(wrapped.end(), tree.begin() + s, tree.begin() + p + 1);
        wrapped = wrap_unary(choice, space, std::move(wrapped));
    }

    Tree result;
    result.reserve(tree.size() + wrapped.size() - (p - s + 1));
    result.insert(result.end(), tree.begin(), tree.begin() + s);
    result.insert(result.end(), wrapped.begin(), wrapped.end());
    result.insert(result.end(), tree.begin() + p + 1, tree.end());
    tree = std::move(result);
    reindex_constants(tree);
    return true;
}

bool swap_operands(Tree& tree, std::mt19937_64& rng) {
    std::vector<std::size_t> bins;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (tree[i].kind == NodeKind::Binary) bins.push_back(i);
    }
    if (bins.empty()) return false;

    std::uniform_int_distribution<std::size_t> pick(0, bins.size() - 1);
    const std::size_t i = bins[pick(rng)];
    const BinaryChildren c = binary_children(tree, i);

    // Span [left_start, i] becomes [right][left][op]; node count is unchanged.
    Tree sub;
    sub.reserve(i - c.left_start + 1);
    sub.insert(sub.end(), tree.begin() + c.right_start, tree.begin() + c.right_end + 1);
    sub.insert(sub.end(), tree.begin() + c.left_start,  tree.begin() + c.left_end + 1);
    sub.push_back(tree[i]);

    Tree result;
    result.reserve(tree.size());
    result.insert(result.end(), tree.begin(), tree.begin() + c.left_start);
    result.insert(result.end(), sub.begin(), sub.end());
    result.insert(result.end(), tree.begin() + i + 1, tree.end());
    tree = std::move(result);
    reindex_constants(tree);
    return true;
}

namespace {

// A minimal owning pointer tree used only by rotate_tree. SR.jl's randomly_rotate_tree!
// manipulates child pointers (including the unary leftmost/rightmost fallback), which maps
// far more legibly onto a pointer tree than onto postfix-span splicing. Trees are tiny
// (<= maxsize), so the per-call build/serialize cost is negligible (rotate is in the
// non-fit path, ~0.4% of compute per docs/30).
struct PNode {
    Node value;
    std::unique_ptr<PNode> l;
    std::unique_ptr<PNode> r;
};

int pdegree(const PNode* n) { return arity(n->value); }

std::unique_ptr<PNode> build_pnode(const Tree& tree) {
    std::vector<std::unique_ptr<PNode>> stack;
    for (const Node& n : tree) {
        auto node = std::make_unique<PNode>();
        node->value = n;
        const int a = arity(n);
        if (a == 2) {
            node->r = std::move(stack.back()); stack.pop_back();
            node->l = std::move(stack.back()); stack.pop_back();
        } else if (a == 1) {
            node->l = std::move(stack.back()); stack.pop_back();
        }
        stack.push_back(std::move(node));
    }
    return std::move(stack.back());
}

void serialize_pnode(const PNode* n, Tree& out) {
    if (n == nullptr) return;
    serialize_pnode(n->l.get(), out);
    serialize_pnode(n->r.get(), out);
    out.push_back(n->value);
}

void collect_pnodes(PNode* n, std::vector<PNode*>& out) {
    if (n == nullptr) return;
    out.push_back(n);
    collect_pnodes(n->l.get(), out);
    collect_pnodes(n->r.get(), out);
}

// SR.jl is_valid_rotation_node: (degree>0 && l.degree>0) || (degree==2 && r.degree>0).
// The first clause includes unary nodes (l is the single operand), so rotations involving
// unary operators are valid candidates.
bool is_valid_rotation(const PNode* n) {
    return (pdegree(n) > 0 && pdegree(n->l.get()) > 0) ||
           (pdegree(n) == 2 && pdegree(n->r.get()) > 0);
}

// Detach and return the rightmost child, with the unary fallback (rightmost == l).
std::unique_ptr<PNode> take_rightmost(PNode& n) {
    return pdegree(&n) == 1 ? std::move(n.l) : std::move(n.r);
}
void set_rightmost(PNode& n, std::unique_ptr<PNode> child) {
    if (pdegree(&n) == 1) n.l = std::move(child); else n.r = std::move(child);
}

// Rotate the subtree owned by `slot` in place, mirroring SR.jl's node_3/4/5 pointer moves.
void rotate_subtree(std::unique_ptr<PNode>& slot, std::mt19937_64& rng) {
    PNode* sr = slot.get();
    const bool right_valid = pdegree(sr->l.get()) > 0;
    const bool left_valid  = pdegree(sr) == 2 && pdegree(sr->r.get()) > 0;
    const bool rotate_right = right_valid && (!left_valid || uniform01(rng) < 0.5);

    if (rotate_right) {
        std::unique_ptr<PNode> node_5 = std::move(slot);
        std::unique_ptr<PNode> node_3 = std::move(node_5->l);       // leftmost(node_5)
        std::unique_ptr<PNode> node_4 = take_rightmost(*node_3);    // rightmost(node_3)
        node_5->l = std::move(node_4);                              // set_leftmost(node_5, node_4)
        set_rightmost(*node_3, std::move(node_5));                  // set_rightmost(node_3, node_5)
        slot = std::move(node_3);
    } else {
        std::unique_ptr<PNode> node_3 = std::move(slot);
        std::unique_ptr<PNode> node_5 = take_rightmost(*node_3);    // rightmost(node_3)
        std::unique_ptr<PNode> node_4 = std::move(node_5->l);       // leftmost(node_5)
        set_rightmost(*node_3, std::move(node_4));                  // set_rightmost(node_3, node_4)
        node_5->l = std::move(node_3);                              // set_leftmost(node_5, node_3)
        slot = std::move(node_5);
    }
}

// True iff some node is a valid rotation node (array-side check matching is_valid_rotation,
// used only to decide whether to offer the rotate mutation).
bool has_valid_rotation(const Tree& tree) {
    for (std::size_t i = 0; i < tree.size(); ++i) {
        const int a = arity(tree[i]);
        if (a == 0) continue;
        if (a == 1) {
            if (arity(tree[i - 1]) > 0) return true;
        } else {
            const std::size_t right_end = i - 1;
            const std::size_t right_start = subtree_start(tree, right_end);
            const std::size_t left_end = right_start - 1;
            if (arity(tree[left_end]) > 0 || arity(tree[right_end]) > 0) return true;
        }
    }
    return false;
}

}  // namespace

bool rotate_tree(Tree& tree, std::mt19937_64& rng) {
    if (tree.empty()) return false;
    std::unique_ptr<PNode> root = build_pnode(tree);

    std::vector<PNode*> nodes;
    collect_pnodes(root.get(), nodes);

    int num_valid = 0;
    for (PNode* n : nodes) {
        if (is_valid_rotation(n)) ++num_valid;
    }
    if (num_valid == 0) return false;

    // SR.jl: rotate at the root with probability 1/num_valid (always when it is the only
    // valid node, since rand() < 1.0); otherwise sample a parent whose chosen child is a
    // valid rotation node and rotate that child.
    const bool root_valid = is_valid_rotation(root.get());
    const bool rotate_at_root =
        root_valid && (uniform01(rng) < 1.0 / static_cast<double>(num_valid));

    if (rotate_at_root) {
        rotate_subtree(root, rng);
    } else {
        std::vector<PNode*> parents;
        for (PNode* t : nodes) {
            const bool l_ok = pdegree(t) > 0 && is_valid_rotation(t->l.get());
            const bool r_ok = pdegree(t) == 2 && is_valid_rotation(t->r.get());
            if (l_ok || r_ok) parents.push_back(t);
        }
        std::uniform_int_distribution<std::size_t> pick(0, parents.size() - 1);
        PNode* parent = parents[pick(rng)];
        const bool l_ok = pdegree(parent) > 0 && is_valid_rotation(parent->l.get());
        const bool r_ok = pdegree(parent) == 2 && is_valid_rotation(parent->r.get());
        bool use_left;
        if (pdegree(parent) == 1) {
            use_left = true;
        } else {
            use_left = l_ok && (!r_ok || uniform01(rng) < 0.5);
        }
        rotate_subtree(use_left ? parent->l : parent->r, rng);
    }

    Tree out;
    out.reserve(tree.size());
    serialize_pnode(root.get(), out);
    reindex_constants(out);
    tree = std::move(out);
    return true;
}

void randomize_tree(Tree& tree, const SearchSpace& space, std::mt19937_64& rng) {
    // SR.jl randomize_tree: draw a target size from 1..curmaxsize, then build a fixed-size
    // random tree. With PySR's default warmup_maxsize_by = 0.0 the maxsize warmup is off, so
    // curmaxsize is constant and equal to maxsize (= max_nodes); we use it directly.
    const int curmaxsize = std::max(1, space.max_nodes);
    const int node_count = std::uniform_int_distribution<int>(1, curmaxsize)(rng);
    tree = gen_random_tree_fixed_size(node_count, space, rng);
}

bool mutate(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
            double const_perturb_scale, const MutationWeights& weights,
            double probability_negate_constant) {
    const std::size_t n_const = node_indices_of(tree, NodeKind::Constant).size();
    const bool has_const = n_const > 0;
    const bool has_op    = has_operator(tree);
    const int  room      = space.max_nodes - static_cast<int>(tree.size());
    const bool can_binary = !space.binary_ops.empty();
    // add_node (append_random_op / prepend_random_op) and insert_random_op all grow the
    // tree: a unary op adds 1 node, a binary op adds 2, a macro operator adds
    // macro_extra_nodes(m). So growth needs a unary alphabet entry that fits in `room`,
    // or a binary op with two free slots.
    const bool can_grow   =
        unary_alphabet_size(space, room, /*primitives_only=*/false) > 0 ||
        (can_binary && room >= 2);
    const bool can_add    = can_grow;
    const bool can_insert = can_grow;

    // Weighted menu over the kinds feasible for this tree (PySR-style MutationWeights).
    // do_nothing / simplify / randomize are always feasible; the rest depend on the
    // tree's content and the size cap. Infeasible kinds are simply not offered, so
    // their weight is redistributed over the feasible ones.
    enum Kind { kConstant, kOperator, kSwap, kRotate, kAdd, kInsert, kDelete,
                kDoNothing, kSimplify, kRandomize };
    std::vector<std::pair<double, Kind>> menu;
    auto offer = [&](bool feasible, double weight, Kind kind) {
        if (feasible && weight > 0.0) menu.emplace_back(weight, kind);
    };
    // SR.jl condition_mutate_constant!: scale the mutate_constant weight down on
    // constant-sparse trees, c *= min(8, n_constants)/8 (MutateModule). With one constant
    // the weight is 1/8 of nominal, reaching full weight only at >= 8 constants.
    const double mutate_constant_w =
        weights.mutate_constant *
        (std::min<std::size_t>(8, n_const) / 8.0);
    offer(has_const,         mutate_constant_w,       kConstant);
    offer(has_op,            weights.mutate_operator, kOperator);
    offer(has_binary(tree),  weights.swap_operands,   kSwap);
    offer(has_valid_rotation(tree), weights.rotate_tree, kRotate);
    offer(can_add,           weights.add_node,        kAdd);
    offer(can_insert,        weights.insert_node,     kInsert);
    offer(has_op,            weights.delete_node,     kDelete);
    offer(true,              weights.do_nothing,      kDoNothing);
    offer(true,              weights.simplify,        kSimplify);
    offer(true,              weights.randomize,       kRandomize);

    double total = 0.0;
    for (const auto& entry : menu) total += entry.first;
    // Degenerate guard: if every offered weight is zero, fall back to randomize. This is a
    // pathological-configuration guard (the PySR-default menu always offers do_nothing), not
    // a mutation failure, so it reports success.
    if (total <= 0.0) { randomize_tree(tree, space, rng); return true; }

    double r = uniform01(rng) * total;
    Kind choice = kDoNothing;
    for (const auto& entry : menu) {
        if (r < entry.first) { choice = entry.second; break; }
        r -= entry.first;
    }

    bool ok = true;
    switch (choice) {
        case kConstant:  ok = mutate_constant(tree, rng, const_perturb_scale,
                                              probability_negate_constant);     break;
        case kOperator:  ok = mutate_operator(tree, space, rng);               break;
        case kSwap:      ok = swap_operands(tree, rng);                        break;
        case kRotate:    ok = rotate_tree(tree, rng);                          break;
        // SR.jl add_node dispatch: 50% append (grow a leaf), 50% prepend (wrap the root).
        case kAdd:       ok = uniform01(rng) < 0.5 ? append_random_op(tree, space, rng)
                                                   : prepend_random_op(tree, space, rng);
                         break;
        case kInsert:    ok = insert_random_op(tree, space, rng);             break;
        case kDelete:    ok = delete_random_op(space, tree, rng);             break;
        case kDoNothing: ok = true;                                          break;
        case kSimplify:  tree = simplify(tree); ok = true;                   break;
        case kRandomize: randomize_tree(tree, space, rng); ok = true;        break;
    }
    // SR.jl skip_mutation_failures: an infeasible sampled kind leaves the tree unchanged
    // and reports failure, so the caller skips the replacement (a no-op cycle) rather than
    // injecting a disruptive whole-tree randomization. With the feasibility-filtered menu
    // above this is rare, but the contract matches PySR's failure handling exactly.
    return ok;
}

}  // namespace rsymbolic
