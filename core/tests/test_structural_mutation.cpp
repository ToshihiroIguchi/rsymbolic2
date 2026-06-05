// Fuzz test: structural mutations must always produce well-formed trees with a
// contiguous constant index range. This is the key safety net for the postfix
// structural edits (the highest-risk part of the encoding).

#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "rsymbolic/evolution/mutation.hpp"
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

// A tree is structurally valid if it is well-formed postfix and its constant indices
// are exactly {0, ..., k-1}.
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

void test_subtree_start_basics() {
    // Postfix for a*x + b: [a x * b +]
    const Tree tree = {constant_node(0, 1.0), variable_node(0),
                       binary_node(BinaryOp::Mul), constant_node(1, 1.0),
                       binary_node(BinaryOp::Add)};
    CHECK(subtree_start(tree, 4) == 0);  // whole tree
    CHECK(subtree_start(tree, 2) == 0);  // (a*x)
    CHECK(subtree_start(tree, 3) == 3);  // leaf b
    CHECK(is_valid_postfix(tree));
}

void test_append_and_delete_keep_validity() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    space.unary_ops = {UnaryOp::Neg, UnaryOp::Exp};
    space.max_depth = 3;
    space.max_nodes = 40;

    std::mt19937_64 rng(2024);
    for (int trial = 0; trial < 500; ++trial) {
        Tree tree = generate_random_tree(space, rng);
        CHECK(structurally_valid(tree));
        // Apply a sequence of mutations, checking validity after each step.
        for (int step = 0; step < 30; ++step) {
            mutate(tree, space, rng, 0.5);
            if (!structurally_valid(tree)) {
                std::printf("invalid after step %d: %s\n", step,
                            to_string(tree).c_str());
                CHECK(false);
                break;
            }
        }
    }
}

void test_delete_shrinks() {
    SearchSpace space;
    std::mt19937_64 rng(11);
    // Build a tree guaranteed to have an operator.
    Tree tree = {constant_node(0, 2.0), variable_node(0), binary_node(BinaryOp::Mul)};
    const std::size_t before = tree.size();
    const bool ok = delete_random_op(tree, rng);
    CHECK(ok);
    CHECK(tree.size() < before);
    CHECK(structurally_valid(tree));
}

void test_append_grows() {
    SearchSpace space;
    space.unary_ops = {UnaryOp::Exp};
    std::mt19937_64 rng(13);
    Tree tree = {variable_node(0)};  // single leaf
    const std::size_t before = tree.size();
    const bool ok = append_random_op(tree, space, rng);
    CHECK(ok);
    CHECK(tree.size() > before);
    CHECK(structurally_valid(tree));
}

}  // namespace

int main() {
    test_subtree_start_basics();
    test_append_and_delete_keep_validity();
    test_delete_shrinks();
    test_append_grows();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
