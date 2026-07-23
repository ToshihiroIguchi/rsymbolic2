// Tests for the LaTeX serializer (rsymbolic/expression/latex.hpp): precedence-aware
// minimal parenthesization, constant rendering, and degenerate values. The infix
// to_string() remains the evaluatable representation; LaTeX is display-only.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "rsymbolic/expression/latex.hpp"
#include "rsymbolic/expression/node.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check_eq(const std::string& actual, const std::string& expected,
              const char* file, int line) {
    ++g_checks;
    if (actual != expected) {
        ++g_failures;
        std::printf("FAIL (%s:%d)\n  expected: %s\n  actual:   %s\n",
                    file, line, expected.c_str(), actual.c_str());
    }
}

#define CHECK_EQ(actual, expected) check_eq((actual), (expected), __FILE__, __LINE__)

using namespace rsymbolic;

// Postfix helpers for readable tree literals.
Node c(double v) { return constant_node(0, v); }
Node x(int i) { return variable_node(i); }
Node u(UnaryOp op) { return unary_node(op); }
Node b(BinaryOp op) { return binary_node(op); }

void test_atoms() {
    CHECK_EQ(to_latex({x(0)}), "x_{0}");
    CHECK_EQ(to_latex({x(12)}), "x_{12}");
    CHECK_EQ(to_latex({c(2.5)}), "2.5");
    CHECK_EQ(to_latex({}), "");
}

void test_binary_precedence() {
    // (x0 + 1) * x1 needs parentheses; x0 + 1 * x1 does not.
    CHECK_EQ(to_latex({x(0), c(1.0), b(BinaryOp::Add), x(1), b(BinaryOp::Mul)}),
             "\\left( x_{0} + 1 \\right) \\cdot x_{1}");
    CHECK_EQ(to_latex({x(0), c(1.0), x(1), b(BinaryOp::Mul), b(BinaryOp::Add)}),
             "x_{0} + 1 \\cdot x_{1}");

    // Subtraction parenthesizes an addition-level right operand only.
    CHECK_EQ(to_latex({x(0), x(1), c(1.0), b(BinaryOp::Add), b(BinaryOp::Sub)}),
             "x_{0} - \\left( x_{1} + 1 \\right)");
    CHECK_EQ(to_latex({x(0), x(1), c(2.0), b(BinaryOp::Mul), b(BinaryOp::Sub)}),
             "x_{0} - x_{1} \\cdot 2");

    // Division renders as \frac with unparenthesized operands.
    CHECK_EQ(to_latex({x(0), c(1.0), b(BinaryOp::Add), x(1), b(BinaryOp::Div)}),
             "\\frac{x_{0} + 1}{x_{1}}");

    // Power parenthesizes a non-atomic base; the exponent stays bare in braces.
    CHECK_EQ(to_latex({x(0), c(1.0), b(BinaryOp::Add), c(2.0), b(BinaryOp::Pow)}),
             "\\left( x_{0} + 1 \\right)^{2}");
    CHECK_EQ(to_latex({x(0), c(2.0), b(BinaryOp::Pow)}), "x_{0}^{2}");
    // (e^{x0})^{x1}: an exponential base must be parenthesized.
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Exp), x(1), b(BinaryOp::Pow)}),
             "\\left( e^{x_{0}} \\right)^{x_{1}}");
}

void test_unary_operators() {
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Sqrt)}), "\\sqrt{x_{0}}");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Sqrt), u(UnaryOp::Sqrt)}),
             "\\sqrt{\\sqrt{x_{0}}}");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Abs)}), "\\left| x_{0} \\right|");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Log)}), "\\log\\left( x_{0} \\right)");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Sin)}), "\\sin\\left( x_{0} \\right)");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Cos)}), "\\cos\\left( x_{0} \\right)");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Tanh)}), "\\tanh\\left( x_{0} \\right)");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Exp)}), "e^{x_{0}}");

    // neg parenthesizes addition-level children only; square parenthesizes
    // everything below an atom.
    CHECK_EQ(to_latex({x(0), x(1), b(BinaryOp::Add), u(UnaryOp::Neg)}),
             "-\\left( x_{0} + x_{1} \\right)");
    CHECK_EQ(to_latex({x(0), c(2.0), b(BinaryOp::Mul), u(UnaryOp::Neg)}),
             "-x_{0} \\cdot 2");
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Square)}), "x_{0}^{2}");
    // inv renders as a fraction; \frac braces group, so no operand parentheses.
    CHECK_EQ(to_latex({x(0), u(UnaryOp::Inv)}), "\\frac{1}{x_{0}}");
    CHECK_EQ(to_latex({x(0), x(1), b(BinaryOp::Add), u(UnaryOp::Inv)}),
             "\\frac{1}{x_{0} + x_{1}}");
    CHECK_EQ(to_latex({x(0), x(1), b(BinaryOp::Add), u(UnaryOp::Square)}),
             "\\left( x_{0} + x_{1} \\right)^{2}");
}

void test_constants() {
    // A negative constant binds like an addition-level fragment.
    CHECK_EQ(to_latex({x(0), c(-2.5), b(BinaryOp::Mul)}),
             "x_{0} \\cdot \\left( -2.5 \\right)");
    // "a + -2" never appears; the negative operand is parenthesized.
    CHECK_EQ(to_latex({x(0), c(-2.0), b(BinaryOp::Add)}),
             "x_{0} + \\left( -2 \\right)");
    // Scientific notation splits into m \cdot 10^{k}.
    CHECK_EQ(to_latex({c(2.5e-7)}), "2.5 \\cdot 10^{-7}");
    CHECK_EQ(to_latex({c(1.0e300)}), "1 \\cdot 10^{300}");
    // Precision parameter controls significant digits.
    CHECK_EQ(to_latex({c(3.14159265)}, 3), "3.14");
    // Non-finite values render without crashing.
    CHECK_EQ(to_latex({c(std::numeric_limits<double>::infinity())}), "\\infty");
    CHECK_EQ(to_latex({c(-std::numeric_limits<double>::infinity())}), "-\\infty");
    CHECK_EQ(to_latex({c(std::numeric_limits<double>::quiet_NaN())}),
             "\\mathrm{NaN}");
}

}  // namespace

int main() {
    test_atoms();
    test_binary_precedence();
    test_unary_operators();
    test_constants();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
