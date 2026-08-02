// Tests for tree evaluation and dual-number gradients, including a finite-difference
// check of the analytic derivative (required by the project's AD-verification policy).

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
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

bool close(double a, double b, double tol) { return std::fabs(a - b) < tol; }

using namespace rsymbolic;

// Tree for y = a*x + b. Postfix: [a][x][*][b][+]
Tree linear_tree(double a0, double b0) {
    return {constant_node(0, a0), variable_node(0), binary_node(BinaryOp::Mul),
            constant_node(1, b0), binary_node(BinaryOp::Add)};
}

// Tree for y = a*exp(b*x). Postfix: [a][b][x][*][exp][*]
Tree exp_tree(double a0, double b0) {
    return {constant_node(0, a0), constant_node(1, b0), variable_node(0),
            binary_node(BinaryOp::Mul), unary_node(UnaryOp::Exp),
            binary_node(BinaryOp::Mul)};
}

// Compute d(prediction)/d(c_k) at a point by central finite differences.
double finite_diff(const Tree& tree, const std::vector<double>& row,
                   std::vector<double> c, int k, double h) {
    c[k] += h;
    const double fp = evaluate<double>(tree, row.data(), c.data());
    c[k] -= 2.0 * h;
    const double fm = evaluate<double>(tree, row.data(), c.data());
    return (fp - fm) / (2.0 * h);
}

// Compute d(prediction)/d(c_k) via a dual-number pass.
double dual_grad(const Tree& tree, const std::vector<double>& row,
                 const std::vector<double>& c, int k) {
    std::vector<Dual> dc(c.size());
    for (std::size_t j = 0; j < c.size(); ++j) {
        dc[j] = Dual(c[j], static_cast<int>(j) == k ? 1.0 : 0.0);
    }
    return evaluate<Dual>(tree, row.data(), dc.data()).deriv;
}

void test_linear_value() {
    const Tree tree = linear_tree(2.5, 1.7);
    const std::vector<double> row = {2.0};  // x = 2
    const std::vector<double> c = {2.5, 1.7};
    const double y = evaluate<double>(tree, row.data(), c.data());
    CHECK(close(y, 2.5 * 2.0 + 1.7, 1e-12));  // 6.7
}

void test_exp_value() {
    const Tree tree = exp_tree(2.0, 0.3);
    const std::vector<double> row = {1.0};  // x = 1
    const std::vector<double> c = {2.0, 0.3};
    const double y = evaluate<double>(tree, row.data(), c.data());
    CHECK(close(y, 2.0 * std::exp(0.3), 1e-12));
}

// Verify dual gradients match finite differences for both models at several points.
void test_gradient_matches_finite_difference() {
    const double h = 1e-6;
    const double tol = 1e-5;

    {
        const Tree tree = linear_tree(2.5, 1.7);
        const std::vector<double> c = {2.5, 1.7};
        for (double x : {-1.0, 0.0, 3.0}) {
            const std::vector<double> row = {x};
            for (int k = 0; k < 2; ++k) {
                CHECK(close(dual_grad(tree, row, c, k),
                            finite_diff(tree, row, c, k, h), tol));
            }
        }
    }
    {
        const Tree tree = exp_tree(2.0, 0.3);
        const std::vector<double> c = {2.0, 0.3};
        for (double x : {0.0, 1.0, 2.5}) {
            const std::vector<double> row = {x};
            for (int k = 0; k < 2; ++k) {
                CHECK(close(dual_grad(tree, row, c, k),
                            finite_diff(tree, row, c, k, h), tol));
            }
        }
    }
}

void test_count_and_initial_constants() {
    const Tree tree = exp_tree(2.0, 0.3);
    CHECK(count_constants(tree) == 2);
    const std::vector<double> init = initial_constants(tree);
    CHECK(init.size() == 2);
    CHECK(close(init[0], 2.0, 1e-12));
    CHECK(close(init[1], 0.3, 1e-12));
}

// Tree for y = a * square(x).  Postfix: [a][x][square][*]
Tree square_tree(double a0) {
    return {constant_node(0, a0), variable_node(0), unary_node(UnaryOp::Square),
            binary_node(BinaryOp::Mul)};
}

// Tree for y = x ^ c.  Postfix: [x][c][pow]
Tree pow_tree(double c0) {
    return {variable_node(0), constant_node(0, c0), binary_node(BinaryOp::Pow)};
}

void test_square_value_and_gradient() {
    const Tree tree = square_tree(3.0);
    const std::vector<double> c = {3.0};
    // y = 3*x^2 at x=2 -> 12
    const std::vector<double> row = {2.0};
    CHECK(close(evaluate<double>(tree, row.data(), c.data()), 12.0, 1e-12));

    // d/da = x^2 = 4
    const double h = 1e-6;
    CHECK(close(dual_grad(tree, row, c, 0), finite_diff(tree, row, c, 0, h), 1e-5));
}

// square / inv / neg render as operators, and the renderings are exact.
//
// to_string() prints them as `(a ^ 2)`, `(1 / a)` and `(-a)` rather than as calls
// (tree.hpp). That is only legitimate if reading the string back gives the same numbers,
// so the claim is checked rather than asserted, over the operands where a guarded or
// signed operator can differ — the infinities, the signed zeros, NaN, and a negative.
void test_operator_renderings() {
    CHECK(to_string(square_tree(3.0)) == "(3 * (x0 ^ 2))");
    // Nested and compound arguments keep the grammar's full parenthesisation.
    const Tree nested = {variable_node(0), unary_node(UnaryOp::Square),
                         unary_node(UnaryOp::Square)};
    CHECK(to_string(nested) == "((x0 ^ 2) ^ 2)");
    const Tree of_sum = {variable_node(0), constant_node(0, 1.5),
                         binary_node(BinaryOp::Add), unary_node(UnaryOp::Square)};
    CHECK(to_string(of_sum) == "((x0 + 1.5) ^ 2)");
    const Tree neg_x = {variable_node(0), unary_node(UnaryOp::Neg)};
    CHECK(to_string(neg_x) == "(-x0)");
    const Tree double_neg = {variable_node(0), unary_node(UnaryOp::Neg),
                             unary_node(UnaryOp::Neg)};
    CHECK(to_string(double_neg) == "(-(-x0))");
    const Tree inv_x = {variable_node(0), unary_node(UnaryOp::Inv)};
    CHECK(to_string(inv_x) == "(1 / x0)");

    // The parentheses around a negation are load bearing, not cosmetic: `-x0 ^ 2` would
    // parse as -(x0^2) in Python and R, which is not square(neg(x0)).
    const Tree square_of_neg = {variable_node(0), unary_node(UnaryOp::Neg),
                                unary_node(UnaryOp::Square)};
    CHECK(to_string(square_of_neg) == "((-x0) ^ 2)");

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double operands[] = {-3.0, -0.0, 0.0, 2.5, inf, -inf, nan};
    for (const double x : operands) {
        const double as_square = rsymbolic::square(x);
        const double as_pow = rsymbolic::pow(x, 2.0);
        CHECK(std::isnan(as_square) ? std::isnan(as_pow) : as_square == as_pow);
        // inv(a) is `1.0 / a` and Div is `a / b`, both unguarded: the same IEEE operation,
        // signed zeros and infinities included.
        const double as_inv = rsymbolic::recip(x);
        const double as_div = detail::apply_binary<double>(BinaryOp::Div, 1.0, x);
        CHECK(std::isnan(as_inv) ? std::isnan(as_div) : as_inv == as_div);
    }
}

// y = a * inv(x): value, AD-vs-finite-difference, and the rendered string.
void test_inv_value_and_gradient() {
    const Tree tree = {constant_node(0, 3.0), variable_node(0),
                       unary_node(UnaryOp::Inv), binary_node(BinaryOp::Mul)};
    const std::vector<double> c = {3.0};
    const std::vector<double> row = {2.0};  // 3 * (1/2) = 1.5
    CHECK(close(evaluate<double>(tree, row.data(), c.data()), 1.5, 1e-12));

    const double h = 1e-6;
    CHECK(close(dual_grad(tree, row, c, 0), finite_diff(tree, row, c, 0, h), 1e-6));

    CHECK(to_string(tree) == "(3 * (1 / x0))");
}

// y = a * erf(x) + sinh(x) + cosh(x): values, AD-vs-finite-difference, rendered string.
void test_erf_sinh_cosh_value_and_gradient() {
    const Tree tree = {constant_node(0, 3.0), variable_node(0),
                       unary_node(UnaryOp::Erf), binary_node(BinaryOp::Mul),
                       variable_node(0), unary_node(UnaryOp::Sinh),
                       binary_node(BinaryOp::Add),
                       variable_node(0), unary_node(UnaryOp::Cosh),
                       binary_node(BinaryOp::Add)};
    const std::vector<double> c = {3.0};
    const std::vector<double> row = {0.5};
    const double want = 3.0 * std::erf(0.5) + std::sinh(0.5) + std::cosh(0.5);
    CHECK(close(evaluate<double>(tree, row.data(), c.data()), want, 1e-12));

    const double h = 1e-6;
    CHECK(close(dual_grad(tree, row, c, 0), finite_diff(tree, row, c, 0, h), 1e-6));

    CHECK(to_string(tree) == "(((3 * erf(x0)) + sinh(x0)) + cosh(x0))");
}

void test_pow_value_and_gradient() {
    const Tree tree = pow_tree(3.0);
    const std::vector<double> c = {3.0};
    // y = x^3 at x=2 -> 8
    const std::vector<double> row = {2.0};
    CHECK(close(evaluate<double>(tree, row.data(), c.data()), 8.0, 1e-9));

    // d/dc = x^c * ln(x) at x=2,c=3 -> 8*ln2
    const double h = 1e-6;
    CHECK(close(dual_grad(tree, row, c, 0), finite_diff(tree, row, c, 0, h), 1e-4));
}

// The protected-operator semantics, asserted directly against SymbolicRegression.jl's
// Operators.jl (docs/69). These are load-bearing for PySR parity: out of domain the
// value must be NaN, because a non-finite loss is how a candidate gets REJECTED. An
// earlier version of this test asserted the opposite for pow (that x<0 with a
// non-integer exponent is finite), which is what let the divergence survive.
void test_safe_operator_semantics() {
    // safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
    CHECK(std::isnan(rsymbolic::sqrt(-1.0)));
    CHECK(std::isnan(rsymbolic::sqrt(-1e-300)));
    CHECK(std::isnan(rsymbolic::sqrt(std::numeric_limits<double>::quiet_NaN())));
    CHECK(rsymbolic::sqrt(0.0) == 0.0);
    CHECK(close(rsymbolic::sqrt(4.0), 2.0, 1e-15));

    // safe_log(x) = x > 0 ? log(x) : NaN. docs/69 left log unguarded, calling it
    // "equivalent in effect" because log(x<=0) is non-finite either way. It is not:
    // rejection happens on the final loss, and a downstream operator maps -Inf back to
    // a finite value (exp(log(0)) == 0), so the candidate survived where SR.jl rejects
    // it. See the tree-level cases in test_log_out_of_domain_is_not_rescued (docs/77).
    CHECK(std::isnan(rsymbolic::log(0.0)));
    CHECK(std::isnan(rsymbolic::log(-0.0)));   // -0.0 <= 0.0 is true
    CHECK(std::isnan(rsymbolic::log(-1.0)));
    CHECK(std::isnan(rsymbolic::log(-std::numeric_limits<double>::infinity())));
    CHECK(std::isnan(rsymbolic::log(std::numeric_limits<double>::quiet_NaN())));
    CHECK(close(rsymbolic::log(1.0), 0.0, 1e-15));
    CHECK(close(rsymbolic::log(std::exp(1.0)), 1.0, 1e-15));
    // in domain: plain log
    CHECK(rsymbolic::log(std::numeric_limits<double>::infinity()) ==
          std::numeric_limits<double>::infinity());

    // safe_pow: plain IEEE pow, except 0^negative is NaN rather than +-Inf.
    CHECK(std::isnan(rsymbolic::pow(0.0, -1.0)));    // SR.jl: isinteger, y<0, x==0
    CHECK(std::isnan(rsymbolic::pow(0.0, -0.5)));    // SR.jl: y<0 && x<=0
    CHECK(std::isnan(rsymbolic::pow(-2.0, 0.5)));    // y>0 non-integer, x<0
    CHECK(std::isnan(rsymbolic::pow(-2.0, -0.5)));   // y<0, x<0
    // A NEAR-integer exponent is not an integer exponent. The old implementation
    // rounded within 1e-6 and answered 4.0 here; SR.jl rejects.
    CHECK(std::isnan(rsymbolic::pow(-2.0, 2.0000001)));
    CHECK(close(rsymbolic::pow(-2.0, 3.0), -8.0, 1e-12));   // integer exponent: real
    CHECK(close(rsymbolic::pow(-2.0, 2.0), 4.0, 1e-12));
    CHECK(rsymbolic::pow(0.0, 2.0) == 0.0);
    CHECK(rsymbolic::pow(0.0, 0.0) == 1.0);                 // Julia: 0.0^0.0 == 1.0
    CHECK(close(rsymbolic::pow(2.0, 0.5), std::sqrt(2.0), 1e-12));

    // The infinities. These are the cases a "simplify the branch table" rewrite gets
    // wrong (docs/69 §4.1): IEEE pow answers +-Inf / +-0 for a -Inf base, but SR.jl's
    // `x < 0` and `x <= 0` guards catch -Inf and return NaN. Verified against Julia.
    const double inf = std::numeric_limits<double>::infinity();
    CHECK(std::isnan(rsymbolic::pow(-inf, 0.5)));    // IEEE pow would say +Inf
    CHECK(std::isnan(rsymbolic::pow(-inf, 1.5)));
    CHECK(std::isnan(rsymbolic::pow(-inf, -0.5)));   // IEEE pow would say +0
    CHECK(std::isnan(rsymbolic::pow(-inf, -1.5)));
    CHECK(std::isnan(rsymbolic::pow(-inf, 2.0000001)));
    CHECK(rsymbolic::pow(inf, 2.0) == inf);          // positive base: plain pow
    CHECK(std::isnan(rsymbolic::pow(2.0, std::numeric_limits<double>::quiet_NaN())));
    // isinteger(Inf) is false in Julia, so the isfinite test in the branch is load
    // bearing: without it floor(Inf) == Inf would take the integer path and skip the
    // x < 0 guard. Julia: safe_pow(-3.0, Inf) = NaN, safe_pow(2.0, Inf) = Inf.
    CHECK(std::isnan(rsymbolic::pow(-3.0, inf)));
    CHECK(rsymbolic::pow(2.0, inf) == inf);

    // square is unguarded and total: square(-3) = 9, no NaN.
    {
        const Tree tree = square_tree(1.0);
        const std::vector<double> c = {1.0};
        const std::vector<double> row = {-3.0};
        const double v = evaluate<double>(tree, row.data(), c.data());
        CHECK(std::isfinite(v));
        CHECK(close(v, 9.0, 1e-12));
    }
    // The same semantics must arrive through the tree evaluator, not just the free
    // function: pow(-2, 2) = 4 stays finite, pow(-2, 1.5) is now NaN.
    {
        const Tree tree = pow_tree(2.0);
        const std::vector<double> c = {2.0};
        const std::vector<double> row = {-2.0};
        CHECK(close(evaluate<double>(tree, row.data(), c.data()), 4.0, 1e-9));
    }
    {
        const Tree tree = pow_tree(1.5);
        const std::vector<double> c = {1.5};
        const std::vector<double> row = {-2.0};
        CHECK(std::isnan(evaluate<double>(tree, row.data(), c.data())));
    }
    // Sqrt through the tree evaluator, the path that used to disagree with the
    // shipped SoA evaluator for a negative argument.
    {
        const Tree tree = {variable_node(0), unary_node(UnaryOp::Sqrt)};
        const std::vector<double> c = {};
        const std::vector<double> neg = {-4.0};
        const std::vector<double> pos = {9.0};
        CHECK(std::isnan(evaluate<double>(tree, neg.data(), c.data())));
        CHECK(close(evaluate<double>(tree, pos.data(), c.data()), 3.0, 1e-12));
    }
}

// The defect an unguarded log actually caused (docs/77). It is NOT that log(0) is
// non-finite — it always was — but that -Inf is not a fixed point of the operator set,
// so the very next node could turn it back into a finite value and the candidate would
// be scored and kept. SR.jl's safe_log emits NaN, and NaN survives all four of these.
//
// `log` and `exp` are both DEFAULT operators, so `exp(log(x))` over data containing an
// exact zero is a default-path divergence, not an opt-in corner.
void test_log_out_of_domain_is_not_rescued() {
    const std::vector<double> c = {};
    const std::vector<double> zero = {0.0};
    const std::vector<double> neg = {-1.0};

    const UnaryOp rescuers[] = {UnaryOp::Exp, UnaryOp::Tanh, UnaryOp::Inv, UnaryOp::Neg};
    for (UnaryOp outer : rescuers) {
        const Tree tree = {variable_node(0), unary_node(UnaryOp::Log), unary_node(outer)};
        CHECK(std::isnan(evaluate<double>(tree, zero.data(), c.data())));
        CHECK(std::isnan(evaluate<double>(tree, neg.data(), c.data())));
    }

    // In domain the composition is untouched: exp(log(4)) == 4.
    {
        const Tree tree = {variable_node(0), unary_node(UnaryOp::Log),
                           unary_node(UnaryOp::Exp)};
        const std::vector<double> pos = {4.0};
        CHECK(close(evaluate<double>(tree, pos.data(), c.data()), 4.0, 1e-12));
    }
}

}  // namespace

int main() {
    test_linear_value();
    test_exp_value();
    test_gradient_matches_finite_difference();
    test_count_and_initial_constants();
    test_square_value_and_gradient();
    test_operator_renderings();
    test_inv_value_and_gradient();
    test_erf_sinh_cosh_value_and_gradient();
    test_pow_value_and_gradient();
    test_safe_operator_semantics();
    test_log_out_of_domain_is_not_rescued();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
