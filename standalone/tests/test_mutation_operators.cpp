// Tests for the PySR-parity mutation operators added in Stage A:
// insert_random_op, rotate_tree, swap_operands, and the weighted mutate() menu.
// The invariants under test are structural (well-formed postfix + contiguous constant
// indices) plus, for the deterministic cases, the exact value the rearranged tree
// evaluates to.

#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/mutation_weights.hpp"
#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s (%s:%d)\n", expr, file, line);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

using namespace rsymbolic;

bool structurally_valid(const Tree& tree) {
    if (tree.empty()) return false;
    if (!is_valid_postfix(tree)) return false;
    std::set<int> indices;
    for (const Node& n : tree) {
        if (n.kind == NodeKind::Constant) indices.insert(n.index);
    }
    const int k = count_constants(tree);
    if (static_cast<int>(indices.size()) != k) return false;
    int expected = 0;
    for (int idx : indices) {
        if (idx != expected++) return false;
    }
    return true;
}

double eval_at(const Tree& tree, const std::vector<double>& row) {
    const std::vector<double> c = initial_constants(tree);
    return evaluate<double>(tree, row.data(), c.data());
}

bool nodes_equal(const Tree& a, const Tree& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].kind != b[i].kind) return false;
        if (a[i].kind == NodeKind::Constant &&
            (a[i].index != b[i].index || a[i].value != b[i].value))
            return false;
        if (a[i].kind == NodeKind::Variable && a[i].index != b[i].index) return false;
        if (a[i].kind == NodeKind::Unary && a[i].uop != b[i].uop) return false;
        if (a[i].kind == NodeKind::Binary && a[i].bop != b[i].bop) return false;
    }
    return true;
}

// ---- swap_operands ------------------------------------------------------------

void test_swap_operands_semantics() {
    // (x0 - c0) with x0=3, c0=1 -> 2; after swap -> (c0 - x0) = -2.
    Tree t = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Sub)};
    CHECK(std::fabs(eval_at(t, {3.0}) - 2.0) < 1e-12);
    std::mt19937_64 rng(1);
    CHECK(swap_operands(t, rng));
    CHECK(structurally_valid(t));
    CHECK(std::fabs(eval_at(t, {3.0}) - (-2.0)) < 1e-12);
}

void test_swap_operands_infeasible() {
    Tree t = {variable_node(0)};  // no binary node
    std::mt19937_64 rng(1);
    const Tree before = t;
    CHECK(!swap_operands(t, rng));
    CHECK(nodes_equal(t, before));  // unchanged on failure
}

// ---- rotate_tree --------------------------------------------------------------

void test_rotate_right_value_preserving() {
    // ((x0 + c0) + c1): left child is binary, right child (c1) is not -> right rotation
    // is the only candidate, so the move is deterministic. Add is associative, so the
    // value is preserved while the structure changes.
    Tree t = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Add),
              constant_node(1, 2.0), binary_node(BinaryOp::Add)};
    const Tree before = t;
    CHECK(std::fabs(eval_at(t, {5.0}) - 8.0) < 1e-12);
    std::mt19937_64 rng(7);
    CHECK(rotate_tree(t, rng));
    CHECK(structurally_valid(t));
    CHECK(!nodes_equal(t, before));                       // structure changed
    CHECK(std::fabs(eval_at(t, {5.0}) - 8.0) < 1e-12);    // value preserved (assoc)
}

void test_rotate_left_value_preserving() {
    // (x0 + (c0 + c1)): right child binary, left child (x0) not -> left rotation only.
    Tree t = {variable_node(0), constant_node(0, 1.0), constant_node(1, 2.0),
              binary_node(BinaryOp::Add), binary_node(BinaryOp::Add)};
    const Tree before = t;
    CHECK(std::fabs(eval_at(t, {5.0}) - 8.0) < 1e-12);
    std::mt19937_64 rng(9);
    CHECK(rotate_tree(t, rng));
    CHECK(structurally_valid(t));
    CHECK(!nodes_equal(t, before));
    CHECK(std::fabs(eval_at(t, {5.0}) - 8.0) < 1e-12);
}

void test_rotate_infeasible() {
    Tree leaf = {variable_node(0)};
    Tree flat = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Add)};
    std::mt19937_64 rng(3);
    CHECK(!rotate_tree(leaf, rng));   // no binary node
    CHECK(!rotate_tree(flat, rng));   // binary node but no binary child
}

// ---- insert_random_op ---------------------------------------------------------

void test_insert_grows_and_valid() {
    SearchSpace space;
    space.unary_ops = {UnaryOp::Exp};
    space.binary_ops = {BinaryOp::Add, BinaryOp::Mul};
    space.max_nodes = 40;
    std::mt19937_64 rng(13);
    Tree t = {variable_node(0)};
    const std::size_t before = t.size();
    CHECK(insert_random_op(t, space, rng));
    CHECK(t.size() > before);
    CHECK(structurally_valid(t));
}

void test_insert_respects_size_cap() {
    SearchSpace space;
    space.unary_ops = {};                       // only binary growth (adds 2)
    space.binary_ops = {BinaryOp::Add};
    std::mt19937_64 rng(5);
    Tree t = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Add)};
    space.max_nodes = static_cast<int>(t.size());  // no room at all
    CHECK(!insert_random_op(t, space, rng));
    space.max_nodes = static_cast<int>(t.size()) + 1;  // room for 1, binary needs 2
    CHECK(!insert_random_op(t, space, rng));
}

// ---- weighted mutate() --------------------------------------------------------

void test_mutate_fuzz_valid() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    space.unary_ops = {UnaryOp::Neg, UnaryOp::Exp, UnaryOp::Sin};
    space.max_depth = 4;
    space.max_nodes = 40;
    std::mt19937_64 rng(2025);
    for (int trial = 0; trial < 1000; ++trial) {
        Tree tree = generate_random_tree(space, rng);
        for (int step = 0; step < 40; ++step) {
            mutate(tree, space, rng, 0.5);  // default PySR MutationWeights
            if (!structurally_valid(tree)) {
                std::printf("invalid after trial %d step %d: %s\n", trial, step,
                            to_string(tree).c_str());
                CHECK(false);
                break;
            }
        }
    }
}

void test_mutate_deterministic() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Mul};
    space.unary_ops = {UnaryOp::Sin};
    std::mt19937_64 rng_a(42), rng_b(42);
    Tree a = generate_random_tree(space, rng_a);
    Tree b = generate_random_tree(space, rng_b);
    for (int i = 0; i < 50; ++i) {
        mutate(a, space, rng_a, 0.5);
        mutate(b, space, rng_b, 0.5);
    }
    CHECK(nodes_equal(a, b));  // same seed -> identical mutation sequence
}

void test_mutate_weight_selects_kind() {
    // With all weight on add_node and only a unary operator available, a single-leaf tree
    // grows to exactly two nodes whether add_node picks append or prepend.
    SearchSpace space;
    space.unary_ops = {UnaryOp::Exp};
    space.binary_ops = {};
    MutationWeights w{};
    w.mutate_constant = 0; w.mutate_operator = 0; w.swap_operands = 0;
    w.rotate_tree = 0; w.insert_node = 0; w.delete_node = 0;
    w.do_nothing = 0; w.simplify = 0; w.randomize = 0;
    w.add_node = 1.0;
    std::mt19937_64 rng(99);
    Tree t = {variable_node(0)};
    mutate(t, space, rng, 0.5, w);
    CHECK(structurally_valid(t));
    CHECK(t.size() == 2);  // leaf wrapped in one unary operator
}

// ---- faithfulness to SR.jl MutationFunctions.jl 1.11.0 ------------------------

void test_prepend_wraps_whole_tree_on_left() {
    // SR.jl prepend_random_op: the ENTIRE tree becomes the LEFT operand of the new root.
    SearchSpace space;
    space.unary_ops = {};
    space.binary_ops = {BinaryOp::Add};
    space.max_nodes = 40;
    const Tree orig = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Mul)};
    Tree t = orig;
    std::mt19937_64 rng(17);
    CHECK(prepend_random_op(t, space, rng));
    CHECK(structurally_valid(t));
    CHECK(t.back().kind == NodeKind::Binary);  // a new binary root was added
    const std::size_t root = t.size() - 1;
    const std::size_t right_start = subtree_start(t, root - 1);
    const std::size_t left_end = right_start - 1;
    const std::size_t left_start = subtree_start(t, left_end);
    const Tree left(t.begin() + left_start, t.begin() + left_end + 1);
    CHECK(nodes_equal(left, orig));  // original tree preserved verbatim as the left operand
}

void test_insert_wraps_subtree_on_left() {
    // SR.jl insert_random_op: the wrapped subtree is ALWAYS the left operand of the new op.
    SearchSpace space;
    space.unary_ops = {};
    space.binary_ops = {BinaryOp::Add};
    space.max_nodes = 40;
    Tree t = {variable_node(0)};
    std::mt19937_64 rng(23);
    CHECK(insert_random_op(t, space, rng));
    CHECK(structurally_valid(t));
    CHECK(t.back().kind == NodeKind::Binary);
    const std::size_t root = t.size() - 1;
    const std::size_t right_start = subtree_start(t, root - 1);
    const std::size_t left_end = right_start - 1;
    const std::size_t left_start = subtree_start(t, left_end);
    CHECK(left_start == 0 && left_end == 0);                  // left operand is the original
    CHECK(t[0].kind == NodeKind::Variable && t[0].index == 0);
}

void test_append_binary_fraction_matches_op_counts() {
    // SR.jl make_new_bin_op: P(binary) = nbin/(nuna+nbin), not a fixed 50/50.
    SearchSpace space;
    space.unary_ops = {UnaryOp::Exp, UnaryOp::Sin, UnaryOp::Cos};  // nuna = 3
    space.binary_ops = {BinaryOp::Add};                            // nbin = 1
    space.max_nodes = 40;
    std::mt19937_64 rng(2026);
    const int N = 40000;
    int binary_count = 0;
    for (int i = 0; i < N; ++i) {
        Tree t = {variable_node(0)};
        CHECK(append_random_op(t, space, rng));
        if (t.size() == 3) ++binary_count;  // binary -> op + 2 leaves; unary -> leaf + op
    }
    const double frac = static_cast<double>(binary_count) / N;
    CHECK(std::fabs(frac - 0.25) < 0.02);  // nbin/(nuna+nbin) = 1/4
}

void test_rotate_fires_on_unary_node() {
    // exp(x0 + c0): the only valid rotation node is the unary exp (degree 1 with an operator
    // child). The generalized rotation must fire here; the old binary-only rotate could not.
    Tree t = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Add),
              unary_node(UnaryOp::Exp)};
    const Tree before = t;
    std::mt19937_64 rng(31);
    CHECK(rotate_tree(t, rng));
    CHECK(structurally_valid(t));
    CHECK(!nodes_equal(t, before));  // structure changed
}

void test_delete_randomizes_leaf_same_size() {
    // (x0 + c0): the only operator is the parent and both children are leaves, so SR.jl's
    // delete replaces a leaf target with a fresh random leaf -> size unchanged, op retained.
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add};
    std::mt19937_64 rng(37);
    for (int i = 0; i < 50; ++i) {
        Tree t = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Add)};
        const std::size_t before = t.size();
        CHECK(delete_random_op(space, t, rng));
        CHECK(t.size() == before);                 // leaf -> random leaf, never shrinks here
        CHECK(structurally_valid(t));
        CHECK(t.back().kind == NodeKind::Binary);   // the operator itself remains
    }
}

// ---- mutate_constant multiplicative kernel (PySR parity) ----------------------

// The constant kernel mirrors SymbolicRegression.jl 1.11.0 mutate_factor: the constant
// is MULTIPLIED by +/- maxChange^U[0,1) (or its reciprocal), never shifted additively.
// With perturbation_factor=0.129 and the fixed temperature 1.0, maxChange = 1.229, so
// the magnitude ratio |new/old| stays within [1/maxChange, maxChange]. The sign flips
// on ~(1 - probability_negate_constant) of calls — intentionally faithful, not a bug.
void test_mutate_constant_multiplicative_kernel() {
    const double perturbation_factor = 0.129;
    const double prob_negate = 0.00743;
    const double max_change = perturbation_factor * 1.0 + 1.0 + 0.1;  // 1.229
    const double base = 2.0;  // positive, so any negative result is a sign flip
    std::mt19937_64 rng(20260622);
    const int N = 200000;
    int sign_flips = 0;
    for (int i = 0; i < N; ++i) {
        Tree t = {constant_node(0, base)};
        CHECK(mutate_constant(t, rng, perturbation_factor, prob_negate));
        const double v = t[0].value;
        const double ratio = std::fabs(v) / base;
        CHECK(ratio >= 1.0 / max_change - 1e-9 && ratio <= max_change + 1e-9);
        if (v < 0.0) ++sign_flips;
    }
    const double flip_frac = static_cast<double>(sign_flips) / N;
    CHECK(std::fabs(flip_frac - (1.0 - prob_negate)) < 0.01);  // ~0.99257
}

void test_mutate_constant_infeasible() {
    SearchSpace space;
    std::mt19937_64 rng(1);
    Tree t = {variable_node(0)};  // no constant present
    CHECK(!mutate_constant(t, rng, 0.129, 0.00743));
}

}  // namespace

int main() {
    test_swap_operands_semantics();
    test_swap_operands_infeasible();
    test_rotate_right_value_preserving();
    test_rotate_left_value_preserving();
    test_rotate_infeasible();
    test_insert_grows_and_valid();
    test_insert_respects_size_cap();
    test_mutate_fuzz_valid();
    test_mutate_deterministic();
    test_mutate_weight_selects_kind();
    test_prepend_wraps_whole_tree_on_left();
    test_insert_wraps_subtree_on_left();
    test_append_binary_fraction_matches_op_counts();
    test_rotate_fires_on_unary_node();
    test_delete_randomizes_leaf_same_size();
    test_mutate_constant_multiplicative_kernel();
    test_mutate_constant_infeasible();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
