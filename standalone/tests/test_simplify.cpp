// Tests for the algebraic simplifier: folding/identity rules, semantics preservation
// (the key correctness property), and bloat reduction.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/simplification/simplify.hpp"

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

bool close(double a, double b, double tol) { return std::fabs(a - b) < tol; }

using namespace rsymbolic;

double eval_at(const Tree& tree, double x) {
    const std::vector<double> c = initial_constants(tree);
    const std::vector<double> row = {x};
    return evaluate<double>(tree, row.data(), c.data());
}

// sub(const 5, const 3) -> const 2
void test_constant_folding_binary() {
    const Tree tree = {constant_node(0, 5.0), constant_node(1, 3.0),
                       binary_node(BinaryOp::Sub)};
    const Tree s = simplify(tree);
    CHECK(s.size() == 1);
    CHECK(count_constants(s) == 1);
    CHECK(close(eval_at(s, 0.0), 2.0, 1e-12));
}

// nested constants: mul(const, mul(const, const)) -> single constant
void test_constant_folding_nested() {
    const Tree tree = {constant_node(0, -0.25), constant_node(1, 0.4),
                       constant_node(2, 0.25), binary_node(BinaryOp::Mul),
                       binary_node(BinaryOp::Mul)};
    const Tree s = simplify(tree);
    CHECK(s.size() == 1);
    CHECK(close(eval_at(s, 0.0), -0.25 * (0.4 * 0.25), 1e-12));
}

// x * 1 -> x
void test_identity_mul_one() {
    const Tree tree = {variable_node(0), constant_node(0, 1.0),
                       binary_node(BinaryOp::Mul)};
    const Tree s = simplify(tree);
    CHECK(s.size() == 1);
    CHECK(s[0].kind == NodeKind::Variable);
}

// x + 0 -> x
void test_identity_add_zero() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0),
                       binary_node(BinaryOp::Add)};
    const Tree s = simplify(tree);
    CHECK(s.size() == 1);
    CHECK(s[0].kind == NodeKind::Variable);
}

// x * 0 -> 0
void test_identity_mul_zero() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0),
                       binary_node(BinaryOp::Mul)};
    const Tree s = simplify(tree);
    CHECK(s.size() == 1);
    CHECK(s[0].kind == NodeKind::Constant);
    CHECK(close(eval_at(s, 3.0), 0.0, 1e-12));
}

// neg(neg(x)) -> x
void test_double_negation() {
    const Tree tree = {variable_node(0), unary_node(UnaryOp::Neg),
                       unary_node(UnaryOp::Neg)};
    const Tree s = simplify(tree);
    CHECK(s.size() == 1);
    CHECK(s[0].kind == NodeKind::Variable);
}

// The bloated form from the search reduces in size and keeps its value.
// ((2.5*x0) - (-3.5 - -1.8)) - (-0.25 * (0.4 * 0.25))
void test_reduces_bloat() {
    const Tree tree = {
        constant_node(0, 2.5), variable_node(0), binary_node(BinaryOp::Mul),
        constant_node(1, -3.5), constant_node(2, -1.8), binary_node(BinaryOp::Sub),
        binary_node(BinaryOp::Sub),
        constant_node(3, -0.25), constant_node(4, 0.4), constant_node(5, 0.25),
        binary_node(BinaryOp::Mul), binary_node(BinaryOp::Mul),
        binary_node(BinaryOp::Sub)};

    const Tree s = simplify(tree);
    CHECK(s.size() < tree.size());  // bloat reduced
    // value preserved at several points
    for (double x : {-2.0, 0.0, 1.0, 5.0}) {
        CHECK(close(eval_at(s, x), eval_at(tree, x), 1e-9));
    }
}

// Semantics preservation on random trees (the key correctness property).
void test_semantics_preserved_random() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};  // no Div: avoid inf
    space.unary_ops = {UnaryOp::Neg};
    space.max_depth = 4;
    std::mt19937_64 rng(2024);

    for (int t = 0; t < 300; ++t) {
        const Tree tree = generate_random_tree(space, rng);
        const Tree s = simplify(tree);
        CHECK(s.size() <= tree.size());  // never grows
        for (double x : {-1.5, -0.3, 0.0, 0.7, 2.0}) {
            const double a = eval_at(tree, x);
            const double b = eval_at(s, x);
            if (std::isfinite(a) || std::isfinite(b)) {
                CHECK(close(a, b, 1e-9));
            }
        }
    }
}

}  // namespace

int main() {
    test_constant_folding_binary();
    test_constant_folding_nested();
    test_identity_mul_one();
    test_identity_add_zero();
    test_identity_mul_zero();
    test_double_negation();
    test_reduces_bloat();
    test_semantics_preserved_random();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
