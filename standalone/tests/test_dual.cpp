// Unit tests for the Dual number type (forward-mode AD primitives).

#include <cmath>
#include <cstdio>

#include "rsymbolic/expression/dual.hpp"

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

bool close(double a, double b, double tol = 1e-9) { return std::fabs(a - b) < tol; }

using rsymbolic::Dual;

// d/dx (x*x) = 2x at x = 3 -> 6
void test_product_rule() {
    Dual x(3.0, 1.0);
    Dual y = x * x;
    CHECK(close(y.value, 9.0));
    CHECK(close(y.deriv, 6.0));
}

// d/dx (1/x) = -1/x^2 at x = 2 -> -0.25
void test_quotient_rule() {
    Dual x(2.0, 1.0);
    Dual y = Dual(1.0) / x;
    CHECK(close(y.value, 0.5));
    CHECK(close(y.deriv, -0.25));
}

// d/dx exp(2x) = 2 exp(2x) at x = 0.5 -> 2*e
void test_exp_chain_rule() {
    Dual x(0.5, 1.0);
    Dual y = exp(Dual(2.0) * x);
    CHECK(close(y.value, std::exp(1.0)));
    CHECK(close(y.deriv, 2.0 * std::exp(1.0)));
}

// d/dx (a*exp(b*x)) w.r.t. b at fixed x: a*x*exp(b*x).
// Here we differentiate w.r.t. b, so seed b, keep x and a as constants.
void test_partial_wrt_one_variable() {
    const double a = 2.0;
    const double x = 1.5;
    Dual b(0.3, 1.0);  // seed derivative w.r.t. b
    Dual y = Dual(a) * exp(b * Dual(x));
    CHECK(close(y.value, a * std::exp(0.3 * x)));
    CHECK(close(y.deriv, a * x * std::exp(0.3 * x)));
}

}  // namespace

int main() {
    test_product_rule();
    test_quotient_rule();
    test_exp_chain_rule();
    test_partial_wrt_one_variable();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
