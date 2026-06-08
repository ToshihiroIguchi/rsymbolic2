#pragma once

#include <cmath>

namespace rsymbolic {

// Forward-mode automatic differentiation via a first-order dual number.
//
// A Dual carries a value and a single directional derivative. To obtain the partial
// derivative of an expression with respect to constant c_k, evaluate the expression
// with c_k seeded as Dual{value, 1} and every other constant as Dual{value, 0}; the
// result's `deriv` is df/dc_k. The Jacobian for k constants is built with k such
// passes (see least_squares_problem.hpp).
//
// A double implicitly converts to a Dual with zero derivative, so variables and
// literals mix naturally in expressions.
struct Dual {
    double value = 0.0;
    double deriv = 0.0;

    Dual() = default;
    Dual(double v) : value(v), deriv(0.0) {}        // NOLINT: implicit on purpose
    Dual(double v, double d) : value(v), deriv(d) {}
};

inline Dual operator+(const Dual& a, const Dual& b) {
    return {a.value + b.value, a.deriv + b.deriv};
}

inline Dual operator-(const Dual& a, const Dual& b) {
    return {a.value - b.value, a.deriv - b.deriv};
}

inline Dual operator-(const Dual& a) { return {-a.value, -a.deriv}; }

inline Dual operator*(const Dual& a, const Dual& b) {
    return {a.value * b.value, a.deriv * b.value + a.value * b.deriv};
}

inline Dual operator/(const Dual& a, const Dual& b) {
    const double inv = 1.0 / b.value;
    return {a.value * inv, (a.deriv * b.value - a.value * b.deriv) * inv * inv};
}

inline Dual exp(const Dual& a) {
    const double e = std::exp(a.value);
    return {e, a.deriv * e};
}

inline Dual log(const Dual& a) {
    return {std::log(a.value), a.deriv / a.value};
}

inline Dual sin(const Dual& a) {
    return {std::sin(a.value), a.deriv * std::cos(a.value)};
}

inline Dual cos(const Dual& a) {
    return {std::cos(a.value), -a.deriv * std::sin(a.value)};
}

// safe: negative input → value=0, deriv=0 (prevents NaN from poisoning the LM solver)
inline Dual sqrt(const Dual& a) {
    const double s = std::sqrt(a.value > 0.0 ? a.value : 0.0);
    return {s, s > 0.0 ? a.deriv / (2.0 * s) : 0.0};
}

inline Dual tanh(const Dual& a) {
    const double t = std::tanh(a.value);
    return {t, a.deriv * (1.0 - t * t)};
}

// subgradient at 0 is 0 (standard convention)
inline Dual abs(const Dual& a) {
    return {std::abs(a.value),
            a.value > 0.0 ? a.deriv : a.value < 0.0 ? -a.deriv : 0.0};
}

// square(x) = x^2; derivative 2x — no domain restriction.
inline double square(double a) { return a * a; }
inline Dual square(const Dual& a) {
    return {a.value * a.value, 2.0 * a.value * a.deriv};
}

// safe_pow for plain doubles: same branch logic as the Dual overload so the value
// path and the AD path always agree. Named pow() so ADL in apply_binary<double> picks
// this instead of std::pow, giving both the same guarded semantics.
inline double pow(double x, double y) {
    if (x > 0.0) return std::exp(y * std::log(x));
    if (x == 0.0 && y > 0.0) return 0.0;
    if (x < 0.0) {
        const double yr = std::round(y);
        if (std::fabs(y - yr) < 1e-6) return std::pow(x, yr);
    }
    return 0.0;
}

// safe_pow: x^y with guarded behaviour when x <= 0 to prevent NaN from poisoning
// the LM solver. Mirrors SymbolicRegression.jl safe_pow semantics:
//   x > 0          : exp(y * log(x))        [standard]
//   x == 0, y > 0  : 0                      [continuous limit]
//   x < 0, y integer (within tol): real-valued result with correct sign
//   otherwise      : 0                      [undefined; treat as 0 to stay finite]
// Derivatives:
//   dp/dx = y * x^(y-1)        [guarded; 0 when safe value is 0]
//   dp/dy = x^y * log(x)       [only when x > 0; 0 otherwise]
inline Dual pow(const Dual& base, const Dual& exp_arg) {
    const double x = base.value;
    const double y = exp_arg.value;

    // Evaluate the safe value and decide which branch we're on.
    double p;
    bool std_branch = false;  // true iff x > 0
    if (x > 0.0) {
        p = std::exp(y * std::log(x));
        std_branch = true;
    } else if (x == 0.0 && y > 0.0) {
        p = 0.0;
    } else if (x < 0.0) {
        // Use the rounded exponent if it is close enough to an integer.
        const double yr = std::round(y);
        if (std::fabs(y - yr) < 1e-6) {
            p = std::pow(x, yr);  // std::pow handles negative base with integer exp
        } else {
            p = 0.0;
        }
    } else {
        p = 0.0;  // x == 0, y <= 0: undefined; return 0
    }

    // Derivatives.
    double dp = 0.0;
    if (std_branch) {
        // dp/dx contribution: y * x^(y-1) * base.deriv
        const double dpdx = y * std::exp((y - 1.0) * std::log(x));
        // dp/dy contribution: x^y * log(x) * exp_arg.deriv
        const double dpdy = p * std::log(x);
        dp = dpdx * base.deriv + dpdy * exp_arg.deriv;
    }
    // On guarded branches: derivative is 0 (function is flat/undefined there).

    return {p, dp};
}

}  // namespace rsymbolic
