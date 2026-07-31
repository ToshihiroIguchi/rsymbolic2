// Unit tests for the Dual number type (forward-mode AD primitives).

#include <cmath>
#include <cstdio>
#include <initializer_list>

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

// d/dx (1/x) = -1/x^2 at x = 2 -> value=0.5, deriv=-0.25; verified against a central
// finite difference (the standing rule for a newly added operator).
void test_inv() {
    Dual x(2.0, 1.0);
    Dual y = rsymbolic::recip(x);
    CHECK(close(y.value, 0.5));
    CHECK(close(y.deriv, -0.25));

    // negative input: 1/x is defined and smooth away from 0
    Dual xn(-4.0, 1.0);
    Dual yn = rsymbolic::recip(xn);
    CHECK(close(yn.value, -0.25));
    CHECK(close(yn.deriv, -1.0 / 16.0));

    // AD vs central finite difference
    const double x0 = 1.3;
    const double h = 1e-6;
    const double fd = (rsymbolic::recip(x0 + h) - rsymbolic::recip(x0 - h)) / (2.0 * h);
    CHECK(std::fabs(rsymbolic::recip(Dual(x0, 1.0)).deriv - fd) < 1e-6);
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

    // x<0, non-integer y -> NaN, matching SR.jl safe_pow (docs/69). A non-finite value
    // is how the candidate gets rejected; this used to return 0, which let expressions
    // survive that PySR discards.
    Dual bad = rsymbolic::pow(Dual(-2.0, 1.0), Dual(1.5, 0.0));
    CHECK(std::isnan(bad.value));
    CHECK(std::isfinite(bad.deriv));  // derivative stays 0 off the x>0 branch

    // x==0, y<0 -> NaN (IEEE pow would give +Inf; SR.jl special-cases this one)
    Dual zneg = rsymbolic::pow(Dual(0.0, 1.0), Dual(-1.0, 0.0));
    CHECK(std::isnan(zneg.value));
}

// erf/sinh/cosh: AD vs central finite differences at several points (the standing rule
// for a newly added operator), plus the odd/even symmetry the display simplifier relies on.
void test_erf_sinh_cosh() {
    const double h = 1e-6;
    for (double x0 : {-2.4, -0.7, 0.0, 0.35, 1.9}) {
        const double fd_erf =
            (rsymbolic::erf(Dual(x0 + h)).value - rsymbolic::erf(Dual(x0 - h)).value) / (2 * h);
        const double fd_sinh =
            (std::sinh(x0 + h) - std::sinh(x0 - h)) / (2 * h);
        const double fd_cosh =
            (std::cosh(x0 + h) - std::cosh(x0 - h)) / (2 * h);
        CHECK(std::fabs(rsymbolic::erf(Dual(x0, 1.0)).deriv - fd_erf) < 1e-6);
        CHECK(std::fabs(rsymbolic::sinh(Dual(x0, 1.0)).deriv - fd_sinh) < 1e-6);
        CHECK(std::fabs(rsymbolic::cosh(Dual(x0, 1.0)).deriv - fd_cosh) < 1e-6);
    }

    // Values against the library functions, and the known erf landmarks.
    CHECK(close(rsymbolic::erf(Dual(0.0, 1.0)).value, 0.0));
    CHECK(close(rsymbolic::erf(Dual(0.0, 1.0)).deriv, 1.1283791670955126));  // 2/sqrt(pi)
    CHECK(close(rsymbolic::erf(Dual(1.0, 0.0)).value, 0.8427007929497149));
    CHECK(close(rsymbolic::sinh(Dual(1.5, 0.0)).value, std::sinh(1.5)));
    CHECK(close(rsymbolic::cosh(Dual(1.5, 0.0)).value, std::cosh(1.5)));

    // cosh(x)^2 - sinh(x)^2 == 1 (a check that is independent of the library values).
    const double c = rsymbolic::cosh(Dual(0.9, 0.0)).value;
    const double s = rsymbolic::sinh(Dual(0.9, 0.0)).value;
    CHECK(close(c * c - s * s, 1.0));

    // Exact odd/even symmetry: the display simplifier folds neg through these
    // (display_simplify.cpp), which is only sound if libm is exactly antisymmetric.
    CHECK(rsymbolic::erf(Dual(-0.73)).value == -rsymbolic::erf(Dual(0.73)).value);
    CHECK(rsymbolic::sinh(Dual(-0.73)).value == -rsymbolic::sinh(Dual(0.73)).value);
    CHECK(rsymbolic::cosh(Dual(-0.73)).value == rsymbolic::cosh(Dual(0.73)).value);

    // Unguarded like exp: an overflowing argument yields Inf rather than a silent clamp,
    // and the loss finiteness guard is what rejects such a candidate.
    CHECK(std::isinf(rsymbolic::cosh(Dual(1000.0, 1.0)).value));
    // erf saturates instead of overflowing — it is bounded, so no guard is possible.
    CHECK(close(rsymbolic::erf(Dual(30.0, 1.0)).value, 1.0));
    CHECK(rsymbolic::erf(Dual(30.0, 1.0)).deriv == 0.0);
}

}  // namespace

int main() {
    test_product_rule();
    test_quotient_rule();
    test_exp_chain_rule();
    test_partial_wrt_one_variable();
    test_square();
    test_inv();
    test_erf_sinh_cosh();
    test_pow_std_branch();
    test_pow_guarded();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
