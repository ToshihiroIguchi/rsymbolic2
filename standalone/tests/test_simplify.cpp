// Tests for the algebraic simplifier. The rule set mirrors SymbolicRegression.jl /
// DynamicExpressions for PySR default parity (docs/29 §11): constant folding plus
// combine_operators (constant reassociation + commutative canonicalisation). It does NOT
// eliminate identities or double negation, so those are asserted to be left unchanged.
// Structural cases are taken from DynamicExpressions' own test_simplification.jl.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
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

// Assert the simplified tree renders to `expected` (infix, via to_string).
void check_str(const Tree& tree, const std::string& expected, const char* file, int line) {
    const std::string got = to_string(simplify(tree));
    ++g_checks;
    if (got != expected) {
        ++g_failures;
        std::printf("FAIL: simplify -> \"%s\", expected \"%s\" (%s:%d)\n", got.c_str(),
                    expected.c_str(), file, line);
    }
}

#define CHECK_STR(tree, expected) check_str((tree), (expected), __FILE__, __LINE__)

// --- Stage 1: constant folding (unchanged from before) ----------------------------

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

// --- Parity: identities and double negation are NOT simplified --------------------

// x * 1 stays x * 1 (SR.jl does not eliminate identities).
void test_identity_mul_one_unchanged() {
    const Tree tree = {variable_node(0), constant_node(0, 1.0),
                       binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "(x0 * 1)");
}

// x + 0 stays x + 0.
void test_identity_add_zero_unchanged() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0),
                       binary_node(BinaryOp::Add)};
    CHECK_STR(tree, "(x0 + 0)");
}

// x * 0 stays x * 0 (not folded to 0).
void test_identity_mul_zero_unchanged() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0),
                       binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "(x0 * 0)");
}

// neg(neg(x)) stays neg(neg(x)).
void test_double_negation_unchanged() {
    const Tree tree = {variable_node(0), unary_node(UnaryOp::Neg),
                       unary_node(UnaryOp::Neg)};
    CHECK_STR(tree, "neg(neg(x0))");
}

// --- Stage 2: combine_operators (cases from test_simplification.jl) ---------------

// Commutative reassociation through recursion: (0.5 + (0.2 + x0)) -> (x0 + 0.7).
void test_combine_commutative_nested() {
    const Tree tree = {constant_node(0, 0.5), constant_node(1, 0.2), variable_node(0),
                       binary_node(BinaryOp::Add), binary_node(BinaryOp::Add)};
    CHECK_STR(tree, "(x0 + 0.7)");
}

// Pure commutative canonicalisation (constant to the right): (2 + x0) -> (x0 + 2).
void test_combine_canonicalise() {
    const Tree tree = {constant_node(0, 2.0), variable_node(0), binary_node(BinaryOp::Add)};
    CHECK_STR(tree, "(x0 + 2)");
}

// (const - (const - var)) => (var - const'): (0.5 - (0.2 - x0)) -> (x0 - -0.3).
void test_combine_sub_c_minus_c_minus_var() {
    const Tree tree = {constant_node(0, 0.5), constant_node(1, 0.2), variable_node(0),
                       binary_node(BinaryOp::Sub), binary_node(BinaryOp::Sub)};
    CHECK_STR(tree, "(x0 - -0.3)");
}

// ((const - var) - const) => (const' - var): ((0.5 - x0) - 0.2) -> (0.3 - x0).
void test_combine_sub_c_minus_var_minus_c() {
    const Tree tree = {constant_node(0, 0.5), variable_node(0), binary_node(BinaryOp::Sub),
                       constant_node(1, 0.2), binary_node(BinaryOp::Sub)};
    CHECK_STR(tree, "(0.3 - x0)");
}

// (const - (var - const)) => (const' - var): (0.5 - (x0 - 0.2)) -> (0.7 - x0).
void test_combine_sub_c_minus_var_minus_c2() {
    const Tree tree = {constant_node(0, 0.5), variable_node(0), constant_node(1, 0.2),
                       binary_node(BinaryOp::Sub), binary_node(BinaryOp::Sub)};
    CHECK_STR(tree, "(0.7 - x0)");
}

// ((var - const) - const) => (var - const'): ((x0 - 0.2) - 0.6) -> (x0 - 0.8).
void test_combine_sub_var_minus_c_minus_c() {
    const Tree tree = {variable_node(0), constant_node(0, 0.2), binary_node(BinaryOp::Sub),
                       constant_node(1, 0.6), binary_node(BinaryOp::Sub)};
    CHECK_STR(tree, "(x0 - 0.8)");
}

// --- Whole-pass behaviour ---------------------------------------------------------

// A bloated form reduces in size (fold then combine) and keeps its value.
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

// Semantics preservation on random trees (the key correctness property). combine_operators
// reassociates constants across a variable, so the simplified value can differ from the
// original by a floating-point rounding step (SR.jl weakens its own tolerance for the same
// reason, test_simplification.jl); use a looser absolute tolerance.
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
                CHECK(close(a, b, 1e-6));
            }
        }
    }
}

}  // namespace

int main() {
    test_constant_folding_binary();
    test_constant_folding_nested();
    test_identity_mul_one_unchanged();
    test_identity_add_zero_unchanged();
    test_identity_mul_zero_unchanged();
    test_double_negation_unchanged();
    test_combine_commutative_nested();
    test_combine_canonicalise();
    test_combine_sub_c_minus_c_minus_var();
    test_combine_sub_c_minus_var_minus_c();
    test_combine_sub_c_minus_var_minus_c2();
    test_combine_sub_var_minus_c_minus_c();
    test_reduces_bloat();
    test_semantics_preserved_random();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
