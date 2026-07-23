// Tests for tree evaluation and dual-number gradients, including a finite-difference
// check of the analytic derivative (required by the project's AD-verification policy).

#include <cmath>
#include <cstdio>
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

// y = a * inv(x): value, AD-vs-finite-difference, and the rendered string.
void test_inv_value_and_gradient() {
    const Tree tree = {constant_node(0, 3.0), variable_node(0),
                       unary_node(UnaryOp::Inv), binary_node(BinaryOp::Mul)};
    const std::vector<double> c = {3.0};
    const std::vector<double> row = {2.0};  // 3 * (1/2) = 1.5
    CHECK(close(evaluate<double>(tree, row.data(), c.data()), 1.5, 1e-12));

    const double h = 1e-6;
    CHECK(close(dual_grad(tree, row, c, 0), finite_diff(tree, row, c, 0, h), 1e-6));

    CHECK(to_string(tree) == "(3 * inv(x0))");
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

// Verify square and pow guarded values are finite (not NaN) at boundary inputs.
void test_safe_boundaries() {
    // square(-3) = 9, no NaN
    {
        const Tree tree = square_tree(1.0);
        const std::vector<double> c = {1.0};
        const std::vector<double> row = {-3.0};
        const double v = evaluate<double>(tree, row.data(), c.data());
        CHECK(std::isfinite(v));
        CHECK(close(v, 9.0, 1e-12));
    }
    // pow(-2, 2) = 4 (integer exponent, negative base)
    {
        const Tree tree = pow_tree(2.0);
        const std::vector<double> c = {2.0};
        const std::vector<double> row = {-2.0};
        const double v = evaluate<double>(tree, row.data(), c.data());
        CHECK(std::isfinite(v));
        CHECK(close(v, 4.0, 1e-9));
    }
    // pow(-2, 1.5) -> guarded (not NaN)
    {
        const Tree tree = pow_tree(1.5);
        const std::vector<double> c = {1.5};
        const std::vector<double> row = {-2.0};
        const double v = evaluate<double>(tree, row.data(), c.data());
        CHECK(std::isfinite(v));
    }
}

}  // namespace

int main() {
    test_linear_value();
    test_exp_value();
    test_gradient_matches_finite_difference();
    test_count_and_initial_constants();
    test_square_value_and_gradient();
    test_inv_value_and_gradient();
    test_pow_value_and_gradient();
    test_safe_boundaries();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
