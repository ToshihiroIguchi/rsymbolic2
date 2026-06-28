// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <array>
#include <cmath>

namespace rsymbolic {

// Vector-mode (batched) forward-mode automatic differentiation.
//
// MultiDual<N> carries a value plus an N-wide gradient block, so a single evaluation
// of an expression computes the value AND its partial derivatives with respect to N
// constants at once. The Jacobian of an expression with k constants is then built in
// ceil(k/N) passes instead of k single-direction passes (least_squares_problem.hpp),
// removing the redundant value/transcendental recomputation and tree traversals that
// the per-constant scalar Dual passes incurred — the dominant search cost (docs/30).
//
// Bit-for-bit equivalence with scalar Dual: every value is computed with the SAME
// floating-point operations as dual.hpp, and each gradient lane uses the SAME
// per-direction formula as Dual::deriv. So each Jacobian entry is identical to the
// scalar k-pass result down to the last bit, which means the whole search is unchanged
// (only faster). This is asserted by test_multi_dual.cpp. When editing, mirror dual.hpp
// EXACTLY: in particular, where dual.hpp divides by a per-op scalar (log, sqrt) the
// lane must also divide — replacing it with a precomputed reciprocal multiply would
// change the low bit and break the equivalence guarantee.
//
// A double implicitly converts to a zero-gradient MultiDual, so variables and literals
// mix naturally in expressions (the std::array<double, N> grad is value-initialised to
// zero by the default member initialiser).
//
// No heap allocation: the gradient is a fixed-size std::array, so MultiDual temporaries
// and the evaluation stack (std::vector<MultiDual<N>>) reuse storage exactly as the
// scalar path does, preserving the allocation-free hot path (docs/23 §4, docs/25).

// Jacobian gradient-block width: number of constant directions differentiated per
// evaluation pass. The optimizer notes k (constants per tree) is "typically <= 5,
// <= ~10 safe" (self_lm_optimizer.hpp), so 8 makes almost every tree a single pass.
// Tunable: results are bit-identical for any N, so this only trades wasted lanes (k<N)
// against pass count (k>N). The default (8) was confirmed against a 4/8/16 sweep on the
// Feynman gate workload (docs/30). Overridable at build time with
// -DRSYMBOLIC2_JAC_BLOCK_WIDTH=<N> for re-measurement.
#ifndef RSYMBOLIC2_JAC_BLOCK_WIDTH
#define RSYMBOLIC2_JAC_BLOCK_WIDTH 8
#endif
inline constexpr int kJacobianBlockWidth = RSYMBOLIC2_JAC_BLOCK_WIDTH;

template <int N>
struct MultiDual {
    double value = 0.0;
    std::array<double, N> grad{};  // zero-initialised gradient block

    MultiDual() = default;
    MultiDual(double v) : value(v) {}  // NOLINT: implicit on purpose (zero gradient)
    MultiDual(double v, const std::array<double, N>& g) : value(v), grad(g) {}
};

template <int N>
inline MultiDual<N> operator+(const MultiDual<N>& a, const MultiDual<N>& b) {
    MultiDual<N> r;
    r.value = a.value + b.value;
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] + b.grad[c];
    return r;
}

template <int N>
inline MultiDual<N> operator-(const MultiDual<N>& a, const MultiDual<N>& b) {
    MultiDual<N> r;
    r.value = a.value - b.value;
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] - b.grad[c];
    return r;
}

template <int N>
inline MultiDual<N> operator-(const MultiDual<N>& a) {
    MultiDual<N> r;
    r.value = -a.value;
    for (int c = 0; c < N; ++c) r.grad[c] = -a.grad[c];
    return r;
}

template <int N>
inline MultiDual<N> operator*(const MultiDual<N>& a, const MultiDual<N>& b) {
    MultiDual<N> r;
    r.value = a.value * b.value;
    // d(ab) = a.deriv * b.value + a.value * b.deriv   (same order as dual.hpp)
    for (int c = 0; c < N; ++c)
        r.grad[c] = a.grad[c] * b.value + a.value * b.grad[c];
    return r;
}

template <int N>
inline MultiDual<N> operator/(const MultiDual<N>& a, const MultiDual<N>& b) {
    const double inv = 1.0 / b.value;
    MultiDual<N> r;
    r.value = a.value * inv;
    // (a.deriv * b.value - a.value * b.deriv) * inv * inv   (same as dual.hpp)
    for (int c = 0; c < N; ++c)
        r.grad[c] = (a.grad[c] * b.value - a.value * b.grad[c]) * inv * inv;
    return r;
}

template <int N>
inline MultiDual<N> exp(const MultiDual<N>& a) {
    const double e = std::exp(a.value);
    MultiDual<N> r;
    r.value = e;
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] * e;
    return r;
}

template <int N>
inline MultiDual<N> log(const MultiDual<N>& a) {
    MultiDual<N> r;
    r.value = std::log(a.value);
    // dual.hpp uses division (a.deriv / a.value); keep division to match bit-for-bit.
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] / a.value;
    return r;
}

template <int N>
inline MultiDual<N> sin(const MultiDual<N>& a) {
    const double cosv = std::cos(a.value);
    MultiDual<N> r;
    r.value = std::sin(a.value);
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] * cosv;
    return r;
}

template <int N>
inline MultiDual<N> cos(const MultiDual<N>& a) {
    const double msinv = -std::sin(a.value);
    MultiDual<N> r;
    r.value = std::cos(a.value);
    // dual.hpp: -a.deriv * sin(value); -grad*sin == grad*(-sin) bit-for-bit (exact sign).
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] * msinv;
    return r;
}

// safe: negative input -> value=0, deriv=0 (prevents NaN poisoning the LM solver).
template <int N>
inline MultiDual<N> sqrt(const MultiDual<N>& a) {
    const double s = std::sqrt(a.value > 0.0 ? a.value : 0.0);
    MultiDual<N> r;
    r.value = s;
    // dual.hpp uses division (a.deriv / (2*s)); keep division to match bit-for-bit.
    for (int c = 0; c < N; ++c) r.grad[c] = s > 0.0 ? a.grad[c] / (2.0 * s) : 0.0;
    return r;
}

template <int N>
inline MultiDual<N> tanh(const MultiDual<N>& a) {
    const double t = std::tanh(a.value);
    const double d = 1.0 - t * t;
    MultiDual<N> r;
    r.value = t;
    for (int c = 0; c < N; ++c) r.grad[c] = a.grad[c] * d;
    return r;
}

// subgradient at 0 is 0 (standard convention).
template <int N>
inline MultiDual<N> abs(const MultiDual<N>& a) {
    MultiDual<N> r;
    r.value = std::abs(a.value);
    for (int c = 0; c < N; ++c)
        r.grad[c] = a.value > 0.0 ? a.grad[c] : a.value < 0.0 ? -a.grad[c] : 0.0;
    return r;
}

// square(x) = x^2; derivative 2x — no domain restriction.
template <int N>
inline MultiDual<N> square(const MultiDual<N>& a) {
    const double two_v = 2.0 * a.value;
    MultiDual<N> r;
    r.value = a.value * a.value;
    // dual.hpp: 2.0 * a.value * a.deriv = (2*value) * deriv  (left-assoc, same bits).
    for (int c = 0; c < N; ++c) r.grad[c] = two_v * a.grad[c];
    return r;
}

// safe_pow: identical value/guard logic and derivative formulas to dual.hpp's
// pow(Dual, Dual). The scalar coefficients (p, dpdx, dpdy) are computed once and
// applied per lane, so every result is bit-identical to the scalar k-pass version.
template <int N>
inline MultiDual<N> pow(const MultiDual<N>& base, const MultiDual<N>& exp_arg) {
    const double x = base.value;
    const double y = exp_arg.value;

    double p;
    bool std_branch = false;  // true iff x > 0
    if (x > 0.0) {
        p = std::exp(y * std::log(x));
        std_branch = true;
    } else if (x == 0.0 && y > 0.0) {
        p = 0.0;
    } else if (x < 0.0) {
        const double yr = std::round(y);
        if (std::fabs(y - yr) < 1e-6) {
            p = std::pow(x, yr);
        } else {
            p = 0.0;
        }
    } else {
        p = 0.0;
    }

    MultiDual<N> r;
    r.value = p;
    if (std_branch) {
        const double dpdx = y * std::exp((y - 1.0) * std::log(x));
        const double dpdy = p * std::log(x);
        for (int c = 0; c < N; ++c)
            r.grad[c] = dpdx * base.grad[c] + dpdy * exp_arg.grad[c];
    }
    // On guarded branches the derivative is 0 — grad is already zero-initialised.

    return r;
}

}  // namespace rsymbolic
