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

// d/dx (x^2) = 2x at x = 3 -> value=9, deriv=6
void test_square() {
    Dual x(3.0, 1.0);
    Dual y = rsymbolic::square(x);
    CHECK(close(y.value, 9.0));
    CHECK(close(y.deriv, 6.0));

    // negative input: value still works, no NaN
    Dual xn(-2.0, 1.0);
    Dual yn = rsymbolic::square(xn);
    CHECK(close(yn.value, 4.0));
    CHECK(close(yn.deriv, -4.0));
}

// safe_pow(x, y): standard branch x>0
// d/dx (x^3) = 3x^2 at x=2 -> 12; d/dy (2^y) = 2^y * ln2 at y=3 -> 8*ln2
void test_pow_std_branch() {
    // Partial w.r.t. base (seed base, fix exponent)
    Dual base(2.0, 1.0);
    Dual exp_arg(3.0, 0.0);
    Dual p = rsymbolic::pow(base, exp_arg);
    CHECK(close(p.value, 8.0));
    CHECK(close(p.deriv, 12.0));  // 3 * 2^2

    // Partial w.r.t. exponent (fix base, seed exponent)
    Dual base2(2.0, 0.0);
    Dual exp_arg2(3.0, 1.0);
    Dual p2 = rsymbolic::pow(base2, exp_arg2);
    CHECK(close(p2.value, 8.0));
    CHECK(close(p2.deriv, 8.0 * std::log(2.0)));  // 2^3 * ln2
}

// safe_pow guard: x <= 0 returns finite (not NaN), derivative is 0
void test_pow_guarded() {
    // x=0, y>0 -> value=0
    Dual z = rsymbolic::pow(Dual(0.0, 1.0), Dual(2.0, 0.0));
    CHECK(std::isfinite(z.value));
    CHECK(z.value == 0.0);
    CHECK(std::isfinite(z.deriv));

    // x<0, integer y: sign-correct ((-2)^3 = -8)
    Dual neg = rsymbolic::pow(Dual(-2.0, 0.0), Dual(3.0, 0.0));
    CHECK(close(neg.value, -8.0));
    CHECK(std::isfinite(neg.deriv));

    // x<0, non-integer y -> 0 (not NaN)
    Dual bad = rsymbolic::pow(Dual(-2.0, 1.0), Dual(1.5, 0.0));
    CHECK(std::isfinite(bad.value));
    CHECK(std::isfinite(bad.deriv));
}

}  // namespace

int main() {
    test_product_rule();
    test_quotient_rule();
    test_exp_chain_rule();
    test_partial_wrt_one_variable();
    test_square();
    test_pow_std_branch();
    test_pow_guarded();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
