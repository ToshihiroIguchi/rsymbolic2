// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cmath>
#include <limits>

#include "rsymbolic/platform/libm.hpp"  // libm::exp/log/sin/cos/erf/pow == std:: off MinGW

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
    const double e = libm::exp(a.value);
    return {e, a.deriv * e};
}

inline Dual log(const Dual& a) {
    return {libm::log(a.value), a.deriv / a.value};
}

inline Dual sin(const Dual& a) {
    return {libm::sin(a.value), a.deriv * libm::cos(a.value)};
}

inline Dual cos(const Dual& a) {
    return {libm::cos(a.value), -a.deriv * libm::sin(a.value)};
}

// safe_sqrt, mirroring SymbolicRegression.jl Operators.jl:
//     safe_sqrt(x) = x >= 0 ? sqrt(x) : NaN
// The NaN is the point: it makes the loss non-finite, and sse_current() turns a
// non-finite residual into kInf, which rejects the candidate — exactly SR.jl's
// NaN-cost rejection. This codebase used to return 0 here instead, on the reasoning
// that NaN would poison the LM solver. It does not: self_lm_optimizer.cpp clamps
// non-finite residuals and Jacobian entries to kLargeResidual, so the solver simply
// steps away. Returning 0 meanwhile let expressions survive that PySR discards, and
// made predict() answer a silently wrong finite number outside the model's domain
// (docs/69).
//
// Written as !(x >= 0) rather than (x < 0) so a NaN argument also yields NaN, which
// is what `x >= zero(x) ? ... : NaN` does in Julia.
inline double sqrt(double x) {
    if (!(x >= 0.0)) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(x);
}

inline Dual sqrt(const Dual& a) {
    if (!(a.value >= 0.0)) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {nan, nan};
    }
    const double s = std::sqrt(a.value);
    // deriv at exactly 0 is +Inf mathematically; 0 is kept here (unchanged) so the
    // Jacobian stays finite on a value that is itself finite and legitimate.
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

// inv(x) = 1/x; derivative -1/x^2. UNGUARDED, exactly like operator/ above: a zero
// argument yields +-Inf and the loss finiteness guard rejects the candidate, which is the
// established convention for division in this codebase (and matches SymbolicRegression.jl's
// `inv`). Named recip() rather than inv() because operator/ declares a local `inv`.
inline double recip(double a) { return 1.0 / a; }
inline Dual recip(const Dual& a) {
    const double r = 1.0 / a.value;
    return {r, -a.deriv * r * r};
}

// 2/sqrt(pi), the factor in d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2). Written as a literal
// because <cmath> guarantees no pi constant portably; multi_dual.hpp and soa_eval.hpp use
// THIS one (they include this header) so the three paths stay bit-identical by construction.
inline constexpr double kTwoOverSqrtPi = 1.1283791670955125739;

// erf(x): the Gauss error function. Unguarded because none is possible or needed — erf is
// defined, smooth and bounded on all of R (SymbolicRegression.jl's `erf` is equally plain).
inline Dual erf(const Dual& a) {
    const double d = kTwoOverSqrtPi * libm::exp(-a.value * a.value);
    return {libm::erf(a.value), a.deriv * d};
}

// sinh/cosh: unguarded like exp — a large argument overflows to +-Inf and the loss
// finiteness guard rejects the candidate, which is this codebase's convention for an
// operator whose *real function* is defined everywhere but can leave the double range.
inline Dual sinh(const Dual& a) {
    return {std::sinh(a.value), a.deriv * std::cosh(a.value)};
}

inline Dual cosh(const Dual& a) {
    return {std::cosh(a.value), a.deriv * std::sinh(a.value)};
}

// safe_pow, a LITERAL transcription of SymbolicRegression.jl Operators.jl:
//
//     if isinteger(y)
//         y < 0 && iszero(x) && return NaN
//     else
//         y > 0 && x <  0 && return NaN
//         y < 0 && x <= 0 && return NaN
//     end
//     return x^y
//
// Transcribed rather than simplified on purpose. The obvious simplification — "IEEE pow
// already returns NaN for a negative base with a non-integer exponent, so the only
// exception is 0^negative" — is WRONG at the infinities: pow(-Inf, 0.5) is +Inf and
// pow(-Inf, -1.5) is +0 under IEEE, while SR.jl's `x < 0` / `x <= 0` guards catch -Inf
// and return NaN. That divergence was found by diffing this function against Julia over
// a grid of edge cases (docs/69 §4.1), not by reading the code.
//
// `isinteger` must exclude the infinities: std::floor(Inf) == Inf, so the isfinite test
// is load bearing (Julia's isinteger(Inf) is false).
//
// This replaces a hand-rolled table that returned **0** wherever SR.jl returns NaN, and
// that accepted a *near*-integer exponent within 1e-6 (so pow(-2, 2.0000001) answered 4
// where SR.jl rejects). Both let candidates survive that PySR discards; see docs/69 and
// difference #12 in docs/29, now closed.
//
// Named pow() so ADL in apply_binary<double> picks this instead of std::pow.
inline double pow(double x, double y) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(y) && std::floor(y) == y) {  // isinteger(y)
        if (y < 0.0 && x == 0.0) return nan;
    } else {
        if (y > 0.0 && x < 0.0) return nan;
        if (y < 0.0 && x <= 0.0) return nan;
    }
    return libm::pow(x, y);
}

// safe_pow for Duals: the value is exactly the double overload above, so the value path
// and the AD path can never disagree. Derivatives are carried only on the x > 0 branch:
//   dp/dx = y * x^(y-1),  dp/dy = x^y * log(x)
// Everywhere else the derivative is 0. That is an implementation choice, not a claim
// about the mathematics — where the value is NaN the candidate is already rejected on
// loss, and where it is finite but x <= 0 (a negative base with an integer exponent)
// dp/dy would be log of a negative number anyway.
inline Dual pow(const Dual& base, const Dual& exp_arg) {
    const double x = base.value;
    const double y = exp_arg.value;
    const double p = pow(x, y);

    double dp = 0.0;
    if (x > 0.0) {
        // dp/dx contribution: y * x^(y-1) * base.deriv
        const double dpdx = y * libm::exp((y - 1.0) * libm::log(x));
        // dp/dy contribution: x^y * log(x) * exp_arg.deriv
        const double dpdy = p * libm::log(x);
        dp = dpdx * base.deriv + dpdy * exp_arg.deriv;
    }

    return {p, dp};
}

}  // namespace rsymbolic
