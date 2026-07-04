// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <array>
#include <cstdint>

namespace rsymbolic {

// A physical dimension: a vector of rational exponents over the 7 SI base dimensions.
//
// Used by the opt-in dimensional-analysis feature (X_units / y_units /
// dimensional_constraint_penalty, PySR-compatible; see units/dimensional_analysis.hpp).
// Only the exponents matter for dimensional consistency — the magnitude/scale of a unit
// (e.g. the "kilo" in "km") is irrelevant and is discarded by the parser.
//
// Representation: a fixed-denominator rational, exactly like DynamicQuantities.jl's
// FixedRational{Int32, 25200} (the type PySR's Julia backend uses). Each exponent is an
// int32 numerator over the compile-time denominator DEN = 25200 = 2^4 * 3^2 * 5^2 * 7,
// which is divisible by 2 and 3, so square roots (/2) and any integer power (*k) stay
// exact and equality is an exact integer-array comparison. This inherits PySR's exact
// representability domain (and its limits) rather than approximating with doubles, which
// would make the ==/+/- dimension checks unreliable.
struct Dimension {
    // Axis order: mass, length, time, current, temperature, amount, luminosity.
    static constexpr int N = 7;
    static constexpr std::int32_t DEN = 25200;

    std::array<std::int32_t, N> num{};  // exponent i is num[i] / DEN; all-zero = dimensionless

    bool operator==(const Dimension& o) const { return num == o.num; }
    bool operator!=(const Dimension& o) const { return num != o.num; }

    bool dimensionless() const {
        for (const std::int32_t v : num)
            if (v != 0) return false;
        return true;
    }

    // Multiplying two quantities adds their exponents; dividing subtracts them.
    Dimension operator+(const Dimension& o) const {
        Dimension r;
        for (int i = 0; i < N; ++i) r.num[i] = num[i] + o.num[i];
        return r;
    }
    Dimension operator-(const Dimension& o) const {
        Dimension r;
        for (int i = 0; i < N; ++i) r.num[i] = num[i] - o.num[i];
        return r;
    }
    // Reciprocal (1 / quantity): negate every exponent.
    Dimension operator-() const {
        Dimension r;
        for (int i = 0; i < N; ++i) r.num[i] = -num[i];
        return r;
    }

    // Raise to an integer power k (e.g. square: scaled(2)). Always exact.
    Dimension scaled(int k) const {
        Dimension r;
        for (int i = 0; i < N; ++i) r.num[i] = num[i] * k;
        return r;
    }

    // Take the k-th root (e.g. sqrt: rooted(2)). Exact only when every exponent is
    // divisible by k; returns false otherwise (the caller treats that as a violation,
    // matching DynamicQuantities' representability limit).
    bool rooted(int k, Dimension& out) const {
        Dimension r;
        for (int i = 0; i < N; ++i) {
            if (num[i] % k != 0) return false;
            r.num[i] = num[i] / k;
        }
        out = r;
        return true;
    }

    // Raise to the rational power p/q (used by the parser for exponents like m^(1/2)).
    // Exact only when p*exponent is divisible by q for every axis; returns false
    // otherwise. q must be non-zero (the parser guarantees this).
    bool pow_rational(int p, int q, Dimension& out) const {
        Dimension r;
        for (int i = 0; i < N; ++i) {
            const long long v = static_cast<long long>(num[i]) * p;
            if (v % q != 0) return false;
            r.num[i] = static_cast<std::int32_t>(v / q);
        }
        out = r;
        return true;
    }
};

// The unit dimension for base axis `axis` (0..6), i.e. exponent 1 on that axis.
inline Dimension base_dim(int axis) {
    Dimension d;
    d.num[static_cast<std::size_t>(axis)] = Dimension::DEN;
    return d;
}

}  // namespace rsymbolic
