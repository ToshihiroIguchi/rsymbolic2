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

}  // namespace

int main() {
    test_linear_value();
    test_exp_value();
    test_gradient_matches_finite_difference();
    test_count_and_initial_constants();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
