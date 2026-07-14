// Tests for the display-only simplifier (docs/54). Unlike simplify.cpp's simplify()
// (parity-preserving, matches SR.jl's combine_operators exactly and therefore never
// eliminates identities or double negation), display_simplify() is a two-layer
// simplifier: deterministic normalisation (Layer 1) plus bounded equality saturation
// over an e-graph (Layer 2, egraph.cpp), adopted only when strictly smaller. It is
// never called from the search loop — this file only exercises it in isolation.

#include <cmath>
#include <cstdio>
#include <limits>
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
// from reassociation/redistribution (docs/54 FP class B rewrites).
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

// Layer-1-only limits: max_iterations = 0 disables the e-graph entirely.
EGraphLimits layer1_only() {
    EGraphLimits lim;
    lim.max_iterations = 0;
    return lim;
}

// --- pow(x, 1) exactness finding ---------------------------------------------------
// The core's safe pow (dual.hpp) evaluates x>0 as exp(y * log(x)), which is NOT bit-
// exact to x for y=1 (a round trip through log/exp loses a rounding step for almost
// every x). This — together with safe_pow's 0-fallback for NaN/undefined bases — is
// why display_simplify has NO Pow rewrite at all (docs/54). Verified here as living
// documentation of the finding.
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

    const Tree s = display_simplify(tree);
    CHECK(s.size() < tree.size());
    CHECK(count_constants(s) == 3);  // -3, 0.5, and the ONE merged trailing constant

    // Layer 1 merges the numerator constant with the product of the two denominator
    // constants: q = cn / (cd1 * cd2).
    const double q = 0.600906 / (0.863117 * -0.696205);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6g", q);
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

// (t / c1) / c2 -> t / (c1*c2)  (kept as a division: docs/52's t/c precision rule)
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

// Nested one level: ((t/c1)/c2)/c3 -> t/(c1*c2*c3), fully collapsed.
void test_div_chain_nested_one_level() {
    const Tree tree = {variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Div),
                       constant_node(1, 3.0), binary_node(BinaryOp::Div),
                       constant_node(2, 5.0), binary_node(BinaryOp::Div)};
    const Tree s = display_simplify(tree);
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
// produce); both canonicalise to the same "(x0 * 0)" form (constant on the right).
void test_identity_mul_zero_excluded() {
    const Tree t1 = {variable_node(0), constant_node(0, 0.0), binary_node(BinaryOp::Mul)};
    const Tree t2 = {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Mul)};
    CHECK_STR(t1, "(x0 * 0)");
    CHECK_STR(t2, "(x0 * 0)");
}

// t^0 is NOT eliminated (would discard t's NaN/Inf; and safe_pow(0,0) is 0, not 1).
void test_identity_pow_zero_excluded() {
    const Tree tree = {variable_node(0), constant_node(0, 0.0), binary_node(BinaryOp::Pow)};
    CHECK_STR(tree, "(x0 ^ 0)");
}

// t^1 is NOT eliminated (pow(x,1) is not bit-exact to x for x>0; see
// test_pow_one_not_exact_for_positive_base and docs/54).
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

// neg(x0) * -1: the sign is pulled out of the factor and cancels with the -1.
void test_mul_neg_one_with_neg_factor_cancels() {
    const Tree tree = {variable_node(0), unary_node(UnaryOp::Neg), constant_node(0, -1.0),
                       binary_node(BinaryOp::Mul)};
    CHECK_STR(tree, "x0");
}

// 0 - t -> neg(t); t - (-c) -> t + c (both exact).
void test_sub_normal_forms() {
    const Tree t1 = {constant_node(0, 0.0), variable_node(0), binary_node(BinaryOp::Sub)};
    CHECK_STR(t1, "neg(x0)");
    const Tree t2 = {variable_node(0), constant_node(0, -2.0), binary_node(BinaryOp::Sub)};
    CHECK_STR(t2, "(x0 + 2)");
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

// --- Layer 1: like terms, like factors, constant chains through +/- -----------------

void test_like_terms_collect() {
    // x + x -> x * 2
    const Tree t1 = {variable_node(0), variable_node(0), binary_node(BinaryOp::Add)};
    CHECK_STR(t1, "(x0 * 2)");
    // 2x + 3x -> x * 5
    const Tree t2 = {variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Mul),
                     variable_node(0), constant_node(1, 3.0), binary_node(BinaryOp::Mul),
                     binary_node(BinaryOp::Add)};
    CHECK_STR(t2, "(x0 * 5)");
}

// x - x keeps a `x * 0` term, never a bare 0: a NaN/Inf produced by the term itself
// must still propagate (docs/54 FP policy). Checked at x = Inf: both forms are NaN.
void test_cancellation_keeps_zero_times_term() {
    const Tree tree = {variable_node(0), variable_node(0), binary_node(BinaryOp::Sub)};
    CHECK_STR(tree, "(x0 * 0)");
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(std::isnan(eval_at(tree, inf)));
    CHECK(std::isnan(eval_at(display_simplify(tree), inf)));
}

void test_like_factors_become_square() {
    // x * x -> square(x): bit-exact (square evaluates as x*x).
    const Tree t1 = {variable_node(0), variable_node(0), binary_node(BinaryOp::Mul)};
    CHECK_STR(t1, "square(x0)");
    // x*x*x*x -> square(square(x)).
    const Tree t2 = {variable_node(0), variable_node(0), binary_node(BinaryOp::Mul),
                     variable_node(0), binary_node(BinaryOp::Mul), variable_node(0),
                     binary_node(BinaryOp::Mul)};
    CHECK_STR(t2, "square(square(x0))");
}

void test_add_constant_chain() {
    // ((x + 1) + 2) + 3 -> x + 6
    const Tree tree = {variable_node(0), constant_node(0, 1.0), binary_node(BinaryOp::Add),
                       constant_node(1, 2.0), binary_node(BinaryOp::Add),
                       constant_node(2, 3.0), binary_node(BinaryOp::Add)};
    CHECK_STR(tree, "(x0 + 6)");
}

// --- Layer 1: exact unary rewrites ---------------------------------------------------

void test_unary_rewrites() {
    // sqrt(square(t)) -> abs(t) (the one drift+overflow-caveat unary rule; docs/54).
    const Tree t1 = {variable_node(0), unary_node(UnaryOp::Square), unary_node(UnaryOp::Sqrt)};
    CHECK_STR(t1, "abs(x0)");
    // abs(neg(t)) -> abs(t); abs(abs(t)) -> abs(t).
    const Tree t2 = {variable_node(0), unary_node(UnaryOp::Neg), unary_node(UnaryOp::Abs)};
    CHECK_STR(t2, "abs(x0)");
    const Tree t3 = {variable_node(0), unary_node(UnaryOp::Abs), unary_node(UnaryOp::Abs)};
    CHECK_STR(t3, "abs(x0)");
    // square(neg(t)) -> square(t).
    const Tree t4 = {variable_node(0), unary_node(UnaryOp::Neg), unary_node(UnaryOp::Square)};
    CHECK_STR(t4, "square(x0)");
    // Odd/even: sin(neg t) -> neg(sin t); cos(neg t) -> cos(t).
    const Tree t5 = {variable_node(0), unary_node(UnaryOp::Neg), unary_node(UnaryOp::Sin)};
    CHECK_STR(t5, "neg(sin(x0))");
    const Tree t6 = {variable_node(0), unary_node(UnaryOp::Neg), unary_node(UnaryOp::Cos)};
    CHECK_STR(t6, "cos(x0)");
    // Excluded inverse compositions stay untouched (docs/54): exp(log t), log(exp t).
    const Tree t7 = {variable_node(0), unary_node(UnaryOp::Log), unary_node(UnaryOp::Exp)};
    CHECK_STR(t7, "exp(log(x0))");
    const Tree t8 = {variable_node(0), unary_node(UnaryOp::Exp), unary_node(UnaryOp::Log)};
    CHECK_STR(t8, "log(exp(x0))");
}

// --- Layer 2: cross-structure factoring the deterministic layer cannot do -----------

void test_layer2_factors_common_term() {
    // x0*x1 + x0*x2 -> x0 * (x1 + x2): 7 nodes -> 5, found only by the e-graph.
    const Tree tree = {variable_node(0), variable_node(1), binary_node(BinaryOp::Mul),
                       variable_node(0), variable_node(2), binary_node(BinaryOp::Mul),
                       binary_node(BinaryOp::Add)};
    DisplaySimplifyStats st;
    const Tree s = display_simplify(tree, &st);
    CHECK(st.layer2_adopted);
    CHECK(st.egraph_saturated);
    CHECK(s.size() == 5);
    CHECK(to_string(s) == "(x0 * (x1 + x2))");

    // Layer 1 alone leaves the 7-node form — the e-graph is what finds the factoring.
    const Tree l1 = display_simplify(tree, nullptr, layer1_only());
    CHECK(l1.size() == 7);

    // Semantics preserved on a few points (2 features used; row has 3 columns).
    for (double x : {-1.0, 0.5, 2.0}) {
        const std::vector<double> row = {x, x + 1.0, x - 2.0};
        const std::vector<double> c0 = initial_constants(tree);
        const std::vector<double> c1 = initial_constants(s);
        CHECK(close(evaluate<double>(tree, row.data(), c0.data()),
                    evaluate<double>(s, row.data(), c1.data())));
    }
}

// --- Layer 2: limits and fallback -----------------------------------------------------

void test_layer2_fallback_on_tiny_limits() {
    const Tree tree = {variable_node(0), variable_node(1), binary_node(BinaryOp::Mul),
                       variable_node(0), variable_node(2), binary_node(BinaryOp::Mul),
                       binary_node(BinaryOp::Add)};
    // An e-node cap too small for any rewriting: the Layer-1 result must come back.
    EGraphLimits lim;
    lim.max_enodes = 2;
    DisplaySimplifyStats st;
    const Tree s = display_simplify(tree, &st, lim);
    CHECK(!st.layer2_adopted);
    const Tree l1 = display_simplify(tree, nullptr, layer1_only());
    CHECK(to_string(s) == to_string(l1));
}

// --- Layer 1 is idempotent ------------------------------------------------------------

void test_layer1_idempotent_random() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    space.unary_ops = {UnaryOp::Neg, UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp,
                       UnaryOp::Sqrt, UnaryOp::Square, UnaryOp::Abs};
    space.max_depth = 4;
    std::mt19937_64 rng(7);
    for (int t = 0; t < 100; ++t) {
        const Tree tree = generate_random_tree(space, rng);
        const Tree once = display_simplify(tree, nullptr, layer1_only());
        const Tree twice = display_simplify(once, nullptr, layer1_only());
        CHECK(to_string(once) == to_string(twice));
    }
}

// --- Whole-pipeline semantics: never grows, matches original within FP tolerance,
//     and never turns a non-finite evaluation into a finite one or vice versa. -------
void test_semantics_preserved_random() {
    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div,
                        BinaryOp::Pow};
    space.unary_ops = {UnaryOp::Neg, UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp,
                       UnaryOp::Log, UnaryOp::Sqrt, UnaryOp::Square, UnaryOp::Abs,
                       UnaryOp::Tanh};
    space.max_depth = 4;
    std::mt19937_64 rng(2024);

    for (int t = 0; t < 300; ++t) {
        const Tree tree = generate_random_tree(space, rng);
        const Tree s = display_simplify(tree);
        CHECK(s.size() <= tree.size());  // never grows

        for (double x : {-1.5, -0.3, 0.0, 0.7, 2.0}) {
            const double a = eval_at(tree, x);
            const double b = eval_at(s, x);
            const bool a_fin = std::isfinite(a);
            const bool b_fin = std::isfinite(b);
            CHECK(a_fin == b_fin);  // finite-ness matches exactly
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
    test_mul_neg_one_with_neg_factor_cancels();
    test_sub_normal_forms();
    test_fold_guard_skips_nan();
    test_fold_guard_skips_inf();
    test_like_terms_collect();
    test_cancellation_keeps_zero_times_term();
    test_like_factors_become_square();
    test_add_constant_chain();
    test_unary_rewrites();
    test_layer2_factors_common_term();
    test_layer2_fallback_on_tiny_limits();
    test_layer1_idempotent_random();
    test_semantics_preserved_random();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
