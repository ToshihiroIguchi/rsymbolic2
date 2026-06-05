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

}  // namespace rsymbolic
