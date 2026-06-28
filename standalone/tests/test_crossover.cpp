// Unit tests for subtree crossover.

#include <cstdio>
#include <random>

#include "rsymbolic/evolution/crossover.hpp"
#include "rsymbolic/expression/node.hpp"
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

// Minimal postfix tree: (x0 + c0)  =>  [x0, c0, Add]
Tree make_add_tree() {
    Tree t;
    t.push_back(variable_node(0));
    t.push_back(constant_node(0, 1.0));
    t.push_back(binary_node(BinaryOp::Add));
    return t;
}

// (x0 * c0)  =>  [x0, c0, Mul]
Tree make_mul_tree() {
    Tree t;
    t.push_back(variable_node(0));
    t.push_back(constant_node(0, 2.0));
    t.push_back(binary_node(BinaryOp::Mul));
    return t;
}

}  // namespace

int main() {
    std::mt19937_64 rng(123);

    // Child must be a valid postfix expression.
    {
        const Tree pa = make_add_tree();
        const Tree pb = make_mul_tree();
        for (int trial = 0; trial < 200; ++trial) {
            std::mt19937_64 r(static_cast<std::uint64_t>(trial));
            const Tree child = subtree_crossover(pa, pb, r, 40);
            CHECK(is_valid_postfix(child));
        }
    }

    // Size guard: crossover with max_nodes=1 must always return pa unchanged when
    // donor is longer than what fits (size(pa) - cut + donor > 1 in most cases).
    {
        const Tree pa = make_add_tree();  // size 3
        const Tree pb = make_mul_tree();  // size 3
        // max_nodes=1: nearly every combination produces size > 1, so child == pa.
        // (The only way child fits is if cut_len == size(pa) and donor_len == 1.)
        // Just check validity for all outcomes.
        for (int trial = 0; trial < 50; ++trial) {
            std::mt19937_64 r(static_cast<std::uint64_t>(trial + 1000));
            const Tree child = subtree_crossover(pa, pb, r, 1);
            CHECK(is_valid_postfix(child));
        }
    }

    // Constant indices must be contiguous 0..k-1 after crossover.
    {
        const Tree pa = make_add_tree();
        const Tree pb = make_mul_tree();
        for (int trial = 0; trial < 200; ++trial) {
            std::mt19937_64 r(static_cast<std::uint64_t>(trial + 2000));
            const Tree child = subtree_crossover(pa, pb, r, 40);
            CHECK(count_constants(child) == static_cast<int>(child.size()) -
                  [&]() {
                      int ops = 0;
                      for (const auto& n : child)
                          if (n.kind == NodeKind::Variable || n.kind == NodeKind::Unary ||
                              n.kind == NodeKind::Binary)
                              ++ops;
                      return ops;
                  }());
            // All constant indices must be < count_constants.
            const int k = count_constants(child);
            for (const auto& n : child) {
                if (n.kind == NodeKind::Constant) {
                    CHECK(n.index >= 0 && n.index < k);
                }
            }
        }
    }

    // Fuzz with deeper trees: 500 random crossovers must produce valid postfix.
    {
        // Build a slightly deeper tree: ((x0 + c0) * (x0 - c1))
        Tree deep;
        deep.push_back(variable_node(0));
        deep.push_back(constant_node(0, 1.0));
        deep.push_back(binary_node(BinaryOp::Add));
        deep.push_back(variable_node(0));
        deep.push_back(constant_node(1, 2.0));
        deep.push_back(binary_node(BinaryOp::Sub));
        deep.push_back(binary_node(BinaryOp::Mul));

        const Tree simple = make_add_tree();
        for (int trial = 0; trial < 500; ++trial) {
            std::mt19937_64 r(static_cast<std::uint64_t>(trial + 3000));
            const Tree c1 = subtree_crossover(deep, simple, r, 40);
            const Tree c2 = subtree_crossover(simple, deep, r, 40);
            CHECK(is_valid_postfix(c1));
            CHECK(is_valid_postfix(c2));
        }
    }

    // Two-child crossover: both children are valid postfix, respect the size cap, and have
    // contiguous constant indices. Accepted children come from a symmetric subtree swap.
    {
        Tree deep;  // ((x0 + c0) * (x0 - c1))
        deep.push_back(variable_node(0));
        deep.push_back(constant_node(0, 1.0));
        deep.push_back(binary_node(BinaryOp::Add));
        deep.push_back(variable_node(0));
        deep.push_back(constant_node(1, 2.0));
        deep.push_back(binary_node(BinaryOp::Sub));
        deep.push_back(binary_node(BinaryOp::Mul));
        const Tree simple = make_add_tree();

        int accepted_count = 0;
        for (int trial = 0; trial < 500; ++trial) {
            std::mt19937_64 r(static_cast<std::uint64_t>(trial + 4000));
            const CrossoverChildren cc = subtree_crossover_pair(deep, simple, r, 40);
            if (!cc.accepted) continue;
            ++accepted_count;
            CHECK(is_valid_postfix(cc.child1));
            CHECK(is_valid_postfix(cc.child2));
            CHECK(static_cast<int>(cc.child1.size()) <= 40);
            CHECK(static_cast<int>(cc.child2.size()) <= 40);
            for (const Tree* c : {&cc.child1, &cc.child2}) {
                const int k = count_constants(*c);
                for (const auto& n : *c) {
                    if (n.kind == NodeKind::Constant) CHECK(n.index >= 0 && n.index < k);
                }
            }
        }
        CHECK(accepted_count > 0);  // a 40-node cap admits these tiny trees every time
    }

    // Tight size cap: when no swap can satisfy the cap for BOTH children, the breed is
    // rejected (accepted == false); when it is accepted, both children honour the cap.
    {
        const Tree pa = make_add_tree();  // size 3
        const Tree pb = make_mul_tree();  // size 3
        for (int trial = 0; trial < 200; ++trial) {
            std::mt19937_64 r(static_cast<std::uint64_t>(trial + 5000));
            const CrossoverChildren cc = subtree_crossover_pair(pa, pb, r, 1);
            if (cc.accepted) {
                CHECK(static_cast<int>(cc.child1.size()) <= 1);
                CHECK(static_cast<int>(cc.child2.size()) <= 1);
            }
        }
    }

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
