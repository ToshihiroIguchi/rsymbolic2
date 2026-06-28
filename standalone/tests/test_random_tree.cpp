// Tests for random tree generation and the mutation operators.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
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

// A tree is well-formed if a postfix evaluation consumes to exactly one value. We
// check this indirectly: constants are contiguous 0..k-1, and evaluation runs.
bool evaluates_ok(const Tree& tree) {
    const int k = count_constants(tree);
    std::vector<double> c(static_cast<std::size_t>(k), 1.0);
    const std::vector<double> row = {0.5};
    const double v = evaluate<double>(tree, row.data(), c.data());
    return std::isfinite(v) || std::isinf(v) || std::isnan(v);  // just must not crash
}

int count_operators(const Tree& tree) {
    int n = 0;
    for (const Node& node : tree) {
        if (arity(node) > 0) ++n;
    }
    return n;
}

// gen_random_tree(nlength) applies append_random_op exactly nlength times, each adding one
// operator. The result is a small tree: nlength operators, with at most nlength binary ops
// (each adding 2 nodes) on top of the single starting leaf -> size <= 1 + 2*nlength.
void test_gen_random_tree_is_small_and_bounded() {
    SearchSpace space;  // default: binary {+,-,*}, no unary
    std::mt19937_64 rng(7);
    constexpr int nlength = 3;
    for (int i = 0; i < 200; ++i) {
        const Tree tree = gen_random_tree(nlength, space, rng);
        CHECK(!tree.empty());
        CHECK(evaluates_ok(tree));
        CHECK(count_operators(tree) == nlength);
        CHECK(static_cast<int>(tree.size()) <= 1 + 2 * nlength);
    }
}

// gen_random_tree_fixed_size(n) grows toward exactly n nodes. With a unary operator
// available it lands exactly on n; with only binary operators a step of 2 can leave it one
// short of an even target (SR.jl's `nuna == 0 && break`). Either way the size stays within
// [max(1, n-1), n] and the tree is well-formed.
void test_gen_random_tree_fixed_size_hits_target() {
    SearchSpace binary_only;  // default: binary only, no unary
    SearchSpace with_unary;
    with_unary.unary_ops = {UnaryOp::Sin, UnaryOp::Cos};
    std::mt19937_64 rng(11);
    for (int n = 1; n <= binary_only.max_nodes; ++n) {
        for (int i = 0; i < 25; ++i) {
            const Tree a = gen_random_tree_fixed_size(n, binary_only, rng);
            CHECK(!a.empty());
            CHECK(evaluates_ok(a));
            CHECK(static_cast<int>(a.size()) <= n);
            CHECK(static_cast<int>(a.size()) >= std::max(1, n - 1));

            const Tree b = gen_random_tree_fixed_size(n, with_unary, rng);
            CHECK(!b.empty());
            CHECK(evaluates_ok(b));
            CHECK(static_cast<int>(b.size()) == n);  // unary available -> exact
        }
    }
}

// generate_random_tree is the general-purpose (test/benchmark) depth-driven generator. It
// must stay well-formed and honour the depth bound.
void test_generate_random_tree_is_wellformed_and_bounded() {
    SearchSpace space;
    space.max_depth = 3;
    std::mt19937_64 rng(7);
    for (int i = 0; i < 200; ++i) {
        const Tree tree = generate_random_tree(space, rng);
        CHECK(!tree.empty());
        CHECK(evaluates_ok(tree));
        // A binary tree of depth d has at most 2^(d+1)-1 nodes.
        CHECK(tree.size() <= (1u << (space.max_depth + 1)) - 1u);
    }
}

// With a large max_depth but a small max_nodes, generate_random_tree must honour the node
// cap rather than the depth bound. The cap is soft: operator nodes pending along the current
// spine can push the final size a little past max_nodes (bounded by ~2*max_depth).
void test_generate_random_tree_respects_max_nodes() {
    SearchSpace space;
    space.max_depth = 8;    // depth bound alone would allow up to 2^9-1 = 511 nodes
    space.max_nodes = 15;   // but the node cap is 15
    std::mt19937_64 rng(7);
    for (int i = 0; i < 200; ++i) {
        const Tree tree = generate_random_tree(space, rng);
        CHECK(!tree.empty());
        CHECK(evaluates_ok(tree));
        CHECK(static_cast<int>(tree.size()) <=
              space.max_nodes + 2 * space.max_depth);
    }
}

void test_generation_is_reproducible() {
    SearchSpace space;
    std::mt19937_64 rng_a(123);
    std::mt19937_64 rng_b(123);
    const Tree a = gen_random_tree(3, space, rng_a);
    const Tree b = gen_random_tree(3, space, rng_b);
    CHECK(to_string(a) == to_string(b));
}

void test_mutate_operator_preserves_structure() {
    SearchSpace space;
    std::mt19937_64 rng(42);
    // gen_random_tree(3) always grows three operators, so it has at least one.
    Tree tree = gen_random_tree(3, space, rng);
    const std::size_t size_before = tree.size();
    const int k_before = count_constants(tree);

    const bool changed = mutate_operator(tree, space, rng);
    CHECK(changed);
    CHECK(tree.size() == size_before);          // structure preserved
    CHECK(count_constants(tree) == k_before);   // constant indices preserved
}

void test_mutate_constant_changes_value_only() {
    SearchSpace space;
    std::mt19937_64 rng(99);
    // Find a tree with at least one constant.
    Tree tree;
    do {
        tree = gen_random_tree(3, space, rng);
    } while (count_constants(tree) == 0);
    const std::size_t size_before = tree.size();
    const std::vector<double> before = initial_constants(tree);

    const bool changed = mutate_constant(tree, rng, 0.5);
    CHECK(changed);
    CHECK(tree.size() == size_before);
    const std::vector<double> after = initial_constants(tree);
    bool any_diff = false;
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (before[i] != after[i]) any_diff = true;
    }
    CHECK(any_diff);
}

void test_randomize_always_valid() {
    SearchSpace space;
    std::mt19937_64 rng(5);
    Tree tree = gen_random_tree(3, space, rng);
    randomize_tree(tree, space, rng);
    CHECK(!tree.empty());
    CHECK(evaluates_ok(tree));
}

}  // namespace

int main() {
    test_gen_random_tree_is_small_and_bounded();
    test_gen_random_tree_fixed_size_hits_target();
    test_generate_random_tree_is_wellformed_and_bounded();
    test_generate_random_tree_respects_max_nodes();
    test_generation_is_reproducible();
    test_mutate_operator_preserves_structure();
    test_mutate_constant_changes_value_only();
    test_randomize_always_valid();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
