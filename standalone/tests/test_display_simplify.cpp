// Tests for the display-only simplifier (docs/52). Unlike simplify.cpp's simplify()
// (parity-preserving, matches SR.jl's combine_operators exactly and therefore never
// eliminates identities or double negation), display_simplify() is richer: it also
// folds Div/Mul constant chains, eliminates safe identities, cancels double negation,
// and rewrites `t * -1` to `neg(t)`. It is never called from the search loop — this
// file only exercises the function in isolation.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/simplification/display_simplify.hpp"

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

double eval_at(const Tree& tree, double x) {
    const std::vector<double> c = initial_constants(tree);
    const std::vector<double> row = {x};
    return evaluate<double>(tree, row.data(), c.data());
}

// Hybrid absolute/relative tolerance: safe for compounded double-precision rounding
// (a handful of extra float ops per fixed-point pass, at most kMaxPasses=8 passes).
bool close(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(a) + std::fabs(b));
}

void check_str(const Tree& tree, const std::string& expected, const char* file, int line) {
    const std::string got = to_string(display_simplify(tree));
    ++g_checks;
    if (got != expected) {
        ++g_failures;
        std::printf("FAIL: display_simplify -> \"%s\", expected \"%s\" (%s:%d)\n",
                    got.c_str(), expected.c_str(), file, line);
    }
}

#define CHECK_STR(tree, expected) check_str((tree), (expected), __FILE__, __LINE__)

const int kMaxPasses = 8;  // mirrors display_simplify.cpp's kMaxPasses

// --- pow(x, 1) exactness finding ---------------------------------------------------
// The core's safe pow (dual.hpp) evaluates x>0 as exp(y * log(x)), which is NOT bit-
// exact to x for y=1 (a round trip through log/exp loses a rounding step for almost
// every x). This is why display_simplify does NOT include a t^1 -> t rewrite (see
// display_simplify.hpp). Verified here as living documentation of the finding.
void test_pow_one_not_exact_for_positive_base() {
    bool any_inexact = false;
    for (double x : {3.0, 0.1, 1e-10, 1e10, 7.0, 100.0, 123456.789}) {
        const double viaSafePow = detail::apply_binary<double>(BinaryOp::Pow, x, 1.0);
        if (viaSafePow != x) any_inexact = true;
    }
    CHECK(any_inexact);  // confirms the exclusion of t^1 -> t is justified
}

// --- GUI regression fixture ---------------------------------------------------------
// ((((sin((x0 * -3)) / exp((x0 * 0.5))) * 0.600906) / 0.863117) / -0.696205)
// The trailing three-constant multiplicative/divisive chain collapses to ONE constant.
void test_gui_regression_constant_chain_collapse() {
    const Tree tree = {
        variable_node(0), constant_node(0, -3.0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Sin),
        variable_node(0), constant_node(1, 0.5), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Exp),
        binary_node(BinaryOp::Div),
        constant_node(2, 0.600906), binary_node(BinaryOp::Mul),
        constant_node(3, 0.863117), binary_node(BinaryOp::Div),
        constant_node(4, -0.696205), binary_node(BinaryOp::Div),
    };

    int passes = 0;
    const Tree s = display_simplify(tree, &passes);
    CHECK(passes < kMaxPasses);  // converged strictly before the cap
    CHECK(s.size() < tree.size());
    CHECK(count_constants(s) == 3);  // -3, 0.5, and the ONE merged trailing constant

    const double v1 = 0.600906 / 0.863117;
    const double v2 = v1 / -0.696205;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", v2);
    const std::string expected =
        "((sin((x0 * -3)) / exp((x0 * 0.5))) * " + std::string(buf) + ")";
    CHECK(to_string(s) == expected);

    // Value preserved (up to FP rounding from the reassociation).
    for (double x : {-2.0, 0.0, 1.0, 3.0}) {
        CHECK(close(eval_at(s, x), eval_at(tree, x), 1e-6));
    }
}

// --- Div/Mul constant-chain folding: each of the eight shapes, standalone ----------

// (t * c1) / c2 -> t * (c1/c2)
void test_div_chain_mul_then_div() {
    const Tree tree = {variable_node(0), constant_node(0, 6.0), binary_node(BinaryOp::Mul),
                       constant_node(1, 3.0), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(x0 * 2)");
}

// (t / c1) / c2 -> t / (c1*c2)
void test_div_chain_div_then_div() {
    const Tree tree = {variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Div),
                       constant_node(1, 3.0), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(x0 / 6)");
}

// (t / c1) * c2 -> t * (c2/c1)
void test_div_chain_div_then_mul() {
    const Tree tree = {variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Div),
                       constant_node(1, 6.0), binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "(x0 * 3)");
}

// c1 / (t * c2) -> (c1/c2) / t
void test_div_chain_c_over_mul() {
    const Tree tree = {constant_node(0, 12.0), variable_node(0), constant_node(1, 4.0),
                       binary_node(BinaryOp::Mul), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(3 / x0)");
}

// c1 / (t / c2) -> (c1*c2) / t
void test_div_chain_c_over_div() {
    const Tree tree = {constant_node(0, 2.0), variable_node(0), constant_node(1, 3.0),
                       binary_node(BinaryOp::Div), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(6 / x0)");
}

// (c1 / t) / c2 -> (c1/c2) / t
void test_div_chain_c_over_t_then_div() {
    const Tree tree = {constant_node(0, 6.0), variable_node(0), binary_node(BinaryOp::Div),
                       constant_node(1, 3.0), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(2 / x0)");
}

// c1 / (c2 / t) -> t * (c1/c2)
void test_div_chain_c_over_c_over_t() {
    const Tree tree = {constant_node(0, 6.0), constant_node(1, 3.0), variable_node(0),
                       binary_node(BinaryOp::Div), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(x0 * 2)");
}

// (c1 / t) * c2 -> (c1*c2) / t
void test_div_chain_c_over_t_then_mul() {
    const Tree tree = {constant_node(0, 2.0), variable_node(0), binary_node(BinaryOp::Div),
                       constant_node(1, 3.0), binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "(6 / x0)");
}

// Nested one level: ((t/c1)/c2)/c3 -> t/(c1*c2*c3), fully collapsed within the pass cap.
void test_div_chain_nested_one_level() {
    const Tree tree = {variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Div),
                       constant_node(1, 3.0), binary_node(BinaryOp::Div),
                       constant_node(2, 5.0), binary_node(BinaryOp::Div)};
    int passes = 0;
    const Tree s = display_simplify(tree, &passes);
    CHECK(passes < kMaxPasses);
    CHECK(s.size() < tree.size());
    CHECK(to_string(s) == "(x0 / 30)");
}

// A div-chain merge that would divide by zero is skipped (isfinite guard).
void test_div_chain_guard_skips_nonfinite() {
    const Tree tree = {variable_node(0), constant_node(0, 5.0), binary_node(BinaryOp::Mul),
                       constant_node(1, 0.0), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "((x0 * 5) / 0)");  // unchanged: 5/0 is not finite
}

// --- Identity elimination ------------------------------------------------------------

void test_identity_mul_one() {
    const Tree tree = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "x0");
}

void test_identity_one_mul() {
    const Tree tree = {constant_node(0, 1.0), variable_node(0), binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "x0");
}

void test_identity_add_zero() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0), binary_node(BinaryOp::Add)};
    CHECK_STR(tree, "x0");
}

void test_identity_zero_add() {
    const Tree tree = {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Add)};
    CHECK_STR(tree, "x0");
}

void test_identity_sub_zero() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0), binary_node(BinaryOp::Sub)};
    CHECK_STR(tree, "x0");
}

void test_identity_div_one() {
    const Tree tree = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "x0");
}

// t*0 and 0*t are NOT eliminated (would discard a NaN/Inf that t itself could
// produce); both canonicalise to the same "(x0 * 0)" form, mirroring simplify()'s own
// (parity) canonicalisation of commutative constant operands to the right.
void test_identity_mul_zero_excluded() {
    const Tree t1 = {variable_node(0), constant_node(0, 0.0), binary_node(BinaryOp::Mul)};
    const Tree t2 = {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul)};
    CHECK_STR(t1, "(x0 * 0)");
    CHECK_STR(t2, "(x0 * 0)");
}

// t^0 is NOT eliminated (would discard t's NaN/Inf).
void test_identity_pow_zero_excluded() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0), binary_node(BinaryOp::Pow)};
    CHECK_STR(tree, "(x0 ^ 0)");
}

// t^1 is NOT eliminated (pow(x,1) is not bit-exact to x for x>0; see
// test_pow_one_not_exact_for_positive_base and display_simplify.hpp).
void test_identity_pow_one_excluded() {
    const Tree tree = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Pow)};
    CHECK_STR(tree, "(x0 ^ 1)");
}

// --- Negation rules -------------------------------------------------------------------

void test_double_negation_cancels() {
    const Tree tree = {variable_node(0), unary_node(UnaryOp::Neg), unary_node(UnaryOp::Neg)};
    CHECK_STR(tree, "x0");
}

void test_mul_neg_one_becomes_neg() {
    const Tree tree = {variable_node(0), constant_node(0, -1.0), binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "neg(x0)");
}

// A case requiring two fixed-point passes: neg(x0) * -1 -> neg(neg(x0)) [pass 1] ->
// x0 [pass 2]. Exercises the multi-pass loop, still well under the cap.
void test_mul_neg_one_then_double_negation_needs_two_passes() {
    const Tree tree = {variable_node(0), unary_node(UnaryOp::Neg), constant_node(0, -1.0),
                       binary_node(BinaryOp::Mul)};
    int passes = 0;
    const Tree s = display_simplify(tree, &passes);
    CHECK(passes >= 2);
    CHECK(passes < kMaxPasses);
    CHECK(to_string(s) == "x0");
}

// --- NaN/Inf guards on constant folding -----------------------------------------------

// log(-1) is NaN; the fold is skipped, leaving the literal subtree unchanged.
void test_fold_guard_skips_nan() {
    const Tree tree = {constant_node(0, -1.0), unary_node(UnaryOp::Log)};
    CHECK_STR(tree, "log(-1)");
    CHECK(!std::isfinite(eval_at(display_simplify(tree), 0.0)));
}

// 1/0 is +inf; the fold is skipped.
void test_fold_guard_skips_inf() {
    const Tree tree = {constant_node(0, 1.0), constant_node(1, 0.0), binary_node(BinaryOp::Div)};
    CHECK_STR(tree, "(1 / 0)");
    CHECK(!std::isfinite(eval_at(display_simplify(tree), 0.0)));
}

// --- Whole-pass semantics: never grows, matches original within FP tolerance,
//     matches non-finite-ness exactly, and never exhausts the pass cap. -------------
void test_semantics_preserved_random() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    space.unary_ops  = {UnaryOp::Neg, UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp,
                        UnaryOp::Sqrt, UnaryOp::Square, UnaryOp::Abs};
    space.max_depth = 4;
    std::mt19937_64 rng(2024);

    for (int t = 0; t < 300; ++t) {
        const Tree tree = generate_random_tree(space, rng);
        int passes = 0;
        const Tree s = display_simplify(tree, &passes);
        CHECK(passes < kMaxPasses);       // fixed point reached before the cap
        CHECK(s.size() <= tree.size());   // never grows

        for (double x : {-1.5, -0.3, 0.0, 0.7, 2.0}) {
            const double a = eval_at(tree, x);
            const double b = eval_at(s, x);
            const bool a_fin = std::isfinite(a);
            const bool b_fin = std::isfinite(b);
            CHECK(a_fin == b_fin);  // NaN/Inf-ness matches exactly
            if (a_fin && b_fin) {
                CHECK(close(a, b));
            }
        }
    }
}

}  // namespace

int main() {
    test_pow_one_not_exact_for_positive_base();
    test_gui_regression_constant_chain_collapse();
    test_div_chain_mul_then_div();
    test_div_chain_div_then_div();
    test_div_chain_div_then_mul();
    test_div_chain_c_over_mul();
    test_div_chain_c_over_div();
    test_div_chain_c_over_t_then_div();
    test_div_chain_c_over_c_over_t();
    test_div_chain_c_over_t_then_mul();
    test_div_chain_nested_one_level();
    test_div_chain_guard_skips_nonfinite();
    test_identity_mul_one();
    test_identity_one_mul();
    test_identity_add_zero();
    test_identity_zero_add();
    test_identity_sub_zero();
    test_identity_div_one();
    test_identity_mul_zero_excluded();
    test_identity_pow_zero_excluded();
    test_identity_pow_one_excluded();
    test_double_negation_cancels();
    test_mul_neg_one_becomes_neg();
    test_mul_neg_one_then_double_negation_needs_two_passes();
    test_fold_guard_skips_nan();
    test_fold_guard_skips_inf();
    test_semantics_preserved_random();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
