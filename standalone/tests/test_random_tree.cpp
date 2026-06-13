// Tests for random tree generation and the mutation operators.

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

void test_generation_is_wellformed_and_bounded() {
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

// With a large max_depth but a small max_nodes, generation must honour the node cap
// rather than the depth bound. mutation/crossover already respect max_nodes; this is
// the matching guard for generation (docs/21 option B). The cap is soft: operator
// nodes pending along the current spine can push the final size a little past
// max_nodes (bounded by ~2*max_depth), but nowhere near the 2^(d+1)-1 depth maximum.
void test_generation_respects_max_nodes() {
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
    const Tree a = generate_random_tree(space, rng_a);
    const Tree b = generate_random_tree(space, rng_b);
    CHECK(to_string(a) == to_string(b));
}

void test_mutate_operator_preserves_structure() {
    SearchSpace space;
    std::mt19937_64 rng(42);
    // Find a tree that has at least one operator.
    Tree tree;
    do {
        tree = generate_random_tree(space, rng);
    } while (count_constants(tree) == tree.size());  // all-leaf (no ops); regenerate
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
        tree = generate_random_tree(space, rng);
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
    Tree tree = generate_random_tree(space, rng);
    randomize_tree(tree, space, rng);
    CHECK(!tree.empty());
    CHECK(evaluates_ok(tree));
}

}  // namespace

int main() {
    test_generation_is_wellformed_and_bounded();
    test_generation_respects_max_nodes();
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
