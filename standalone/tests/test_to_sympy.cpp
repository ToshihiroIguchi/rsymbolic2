// Tests for the SymPy serializer (rsymbolic/expression/sympy.hpp): precedence-aware
// minimal parenthesization in Python syntax, and the four tokens to_string() emits that
// Python does not accept — square(), inv(), neg() and `^`.
//
// The strings asserted here are checked for SEMANTIC correctness (that sympify() parses
// them to the same function) by python/tests/test_sympy_export.py, which round-trips a
// fitted front through SymPy and compares against predict(). This file pins the syntax.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/sympy.hpp"

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
    CHECK_EQ(to_sympy({x(0)}), "x0");
    CHECK_EQ(to_sympy({x(12)}), "x12");
    CHECK_EQ(to_sympy({c(2.5)}), "2.5");
    CHECK_EQ(to_sympy({}), "");
}

void test_binary_precedence() {
    CHECK_EQ(to_sympy({x(0), c(1.0), b(BinaryOp::Add), x(1), b(BinaryOp::Mul)}),
             "(x0 + 1)*x1");
    CHECK_EQ(to_sympy({x(0), c(1.0), x(1), b(BinaryOp::Mul), b(BinaryOp::Add)}),
             "x0 + 1*x1");

    CHECK_EQ(to_sympy({x(0), x(1), c(1.0), b(BinaryOp::Add), b(BinaryOp::Sub)}),
             "x0 - (x1 + 1)");
    CHECK_EQ(to_sympy({x(0), x(1), c(2.0), b(BinaryOp::Mul), b(BinaryOp::Sub)}),
             "x0 - x1*2");

    // `/` is left-associative: a Mul-level divisor must be parenthesized, a Mul-level
    // dividend must not (x0*x1/x2 is already correct).
    CHECK_EQ(to_sympy({x(0), x(1), x(2), b(BinaryOp::Mul), b(BinaryOp::Div)}),
             "x0/(x1*x2)");
    CHECK_EQ(to_sympy({x(0), x(1), b(BinaryOp::Mul), x(2), b(BinaryOp::Div)}),
             "x0*x1/x2");
    CHECK_EQ(to_sympy({x(0), c(1.0), b(BinaryOp::Add), x(1), b(BinaryOp::Div)}),
             "(x0 + 1)/x1");

    // `**` is right-associative, so BOTH a non-atomic base and a non-atomic exponent
    // need parentheses: x0**2**3 would be x0**(2**3).
    CHECK_EQ(to_sympy({x(0), c(2.0), b(BinaryOp::Pow)}), "x0**2");
    CHECK_EQ(to_sympy({x(0), c(1.0), b(BinaryOp::Add), c(2.0), b(BinaryOp::Pow)}),
             "(x0 + 1)**2");
    CHECK_EQ(to_sympy({x(0), c(2.0), b(BinaryOp::Pow), c(3.0), b(BinaryOp::Pow)}),
             "(x0**2)**3");
    CHECK_EQ(to_sympy({x(0), x(1), c(1.0), b(BinaryOp::Add), b(BinaryOp::Pow)}),
             "x0**(x1 + 1)");
    // A negative exponent is Add-level, so it is parenthesized rather than left as
    // the (legal but unreadable) x0**-1.
    CHECK_EQ(to_sympy({x(0), c(-1.0), b(BinaryOp::Pow)}), "x0**(-1)");
    // A function call is atomic and needs no parentheses as a base.
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Exp), x(1), b(BinaryOp::Pow)}), "exp(x0)**x1");
}

void test_unary_operators() {
    // The nine operators whose SymPy name is their own name.
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Exp)}), "exp(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Log)}), "log(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Sin)}), "sin(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Cos)}), "cos(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Sqrt)}), "sqrt(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Tanh)}), "tanh(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Sinh)}), "sinh(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Cosh)}), "cosh(x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Erf)}), "erf(x0)");
    // `abs`, not `Abs`: Python's builtin abs() calls Symbol.__abs__ and yields SymPy's Abs,
    // so the lowercase form is valid under sympify AND under plain eval/NumPy, where `Abs`
    // would not be defined.
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Abs)}), "abs(x0)");

    // The three that have no Python function at all — the reason this serializer exists.
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Square)}), "x0**2");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Inv)}), "1/x0");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Neg)}), "-x0");

    // square parenthesizes anything below an atom, because -3**2 is -9 in Python.
    CHECK_EQ(to_sympy({x(0), x(1), b(BinaryOp::Add), u(UnaryOp::Square)}), "(x0 + x1)**2");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Neg), u(UnaryOp::Square)}), "(-x0)**2");
    CHECK_EQ(to_sympy({c(-3.0), u(UnaryOp::Square)}), "(-3)**2");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Sin), u(UnaryOp::Square)}), "sin(x0)**2");

    // inv keeps Mul precedence: 1/x0*x1 is x1/x0, but z/(1/x0) must keep its parens.
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Inv), x(1), b(BinaryOp::Mul)}), "1/x0*x1");
    CHECK_EQ(to_sympy({x(0), x(1), b(BinaryOp::Mul), u(UnaryOp::Inv)}), "1/(x0*x1)");
    CHECK_EQ(to_sympy({x(1), x(0), u(UnaryOp::Inv), b(BinaryOp::Div)}), "x1/(1/x0)");
    CHECK_EQ(to_sympy({x(0), u(UnaryOp::Inv), u(UnaryOp::Square)}), "(1/x0)**2");

    // neg parenthesizes addition-level children only.
    CHECK_EQ(to_sympy({x(0), x(1), b(BinaryOp::Add), u(UnaryOp::Neg)}), "-(x0 + x1)");
    CHECK_EQ(to_sympy({x(0), c(2.0), b(BinaryOp::Mul), u(UnaryOp::Neg)}), "-x0*2");
}

void test_constants() {
    // A negative constant binds like an addition-level fragment.
    CHECK_EQ(to_sympy({x(0), c(-2.5), b(BinaryOp::Mul)}), "x0*(-2.5)");
    CHECK_EQ(to_sympy({x(0), c(-2.0), b(BinaryOp::Add)}), "x0 + (-2)");
    // %g scientific notation is already a valid Python literal — no rewriting needed.
    CHECK_EQ(to_sympy({c(2.5e-7)}), "2.5e-07");
    CHECK_EQ(to_sympy({c(3.14159265)}, 3), "3.14");
    // Non-finite values use SymPy's spellings, not Python's float('nan')/float('inf').
    CHECK_EQ(to_sympy({c(std::numeric_limits<double>::infinity())}), "oo");
    CHECK_EQ(to_sympy({c(-std::numeric_limits<double>::infinity())}), "-oo");
    CHECK_EQ(to_sympy({c(std::numeric_limits<double>::quiet_NaN())}), "nan");
    // A negative infinity still binds as addition-level.
    CHECK_EQ(to_sympy({x(0), c(-std::numeric_limits<double>::infinity()),
                       b(BinaryOp::Mul)}),
             "x0*(-oo)");
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
