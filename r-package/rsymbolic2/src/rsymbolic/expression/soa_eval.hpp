// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "rsymbolic/expression/dual.hpp"        // rsymbolic::pow(double,double)
#include "rsymbolic/expression/multi_dual.hpp"  // kJacobianBlockWidth (for callers)
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Struct-of-arrays (SoA), point-batched evaluation of a postfix tree.
//
// The scalar evaluate<T>() in tree.hpp walks the tree once per data point, one value at
// a time through a postfix stack. That layout cannot vectorise: the per-point control
// flow (switch + stack churn) dominates and the SIMD lanes go unused. The evaluators
// here instead process a *tile of P points per node*, so each operator becomes a tight
// loop over P contiguous doubles — the form the compiler can auto-vectorise for +-*/ and
// sqrt, and which eliminates the per-point interpreter dispatch even for the libm
// transcendentals (docs/30 "SoA point-batched evaluator").
//
// BIT-EXACTNESS (load-bearing). Every result is bit-identical to the scalar path:
//   - residual: same op, same order as tree.hpp::detail::apply_unary/apply_binary
//     (double) for each point;
//   - Jacobian: same value path AND same per-direction derivative formula as
//     multi_dual.hpp for each point and each gradient lane.
// Points are independent and the per-node op order within a point is unchanged, so moving
// the point loop inside each node changes nothing numerically. This is the guarantee that
// the search is provably unchanged (only faster); test_soa_eval.cpp asserts it strictly
// against the production scalar `evaluate<double>` and the production MultiDual closure.
//
// Layout. Inputs are column-major: cols[j][i] is feature j of point i. Leaf nodes copy
// their tile slice into a fresh writable segment so the later in-place unary/binary writes
// never corrupt the shared input columns. Scratch (pool + stack + coeff) is caller-owned
// and reused across calls — no allocation on the hot path once sized.

namespace detail {

// --- residual (plain double) kernels: mirror apply_unary/apply_binary<double> ---------

inline void soa_res_unary(UnaryOp op, double* a, std::size_t P) {
    switch (op) {
        case UnaryOp::Neg:    for (std::size_t p = 0; p < P; ++p) a[p] = -a[p]; break;
        case UnaryOp::Exp:    for (std::size_t p = 0; p < P; ++p) a[p] = std::exp(a[p]); break;
        case UnaryOp::Log:    for (std::size_t p = 0; p < P; ++p) a[p] = std::log(a[p]); break;
        case UnaryOp::Sin:    for (std::size_t p = 0; p < P; ++p) a[p] = std::sin(a[p]); break;
        case UnaryOp::Cos:    for (std::size_t p = 0; p < P; ++p) a[p] = std::cos(a[p]); break;
        case UnaryOp::Sqrt:   for (std::size_t p = 0; p < P; ++p)
                                  a[p] = std::sqrt(a[p] > 0.0 ? a[p] : 0.0); break;  // safe, == dual.hpp
        case UnaryOp::Tanh:   for (std::size_t p = 0; p < P; ++p) a[p] = std::tanh(a[p]); break;
        case UnaryOp::Abs:    for (std::size_t p = 0; p < P; ++p) a[p] = std::abs(a[p]); break;
        case UnaryOp::Square: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] * a[p]; break;
    }
}

inline void soa_res_binary(BinaryOp op, double* a, const double* b, std::size_t P) {
    switch (op) {
        case BinaryOp::Add: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] + b[p]; break;
        case BinaryOp::Sub: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] - b[p]; break;
        case BinaryOp::Mul: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] * b[p]; break;
        case BinaryOp::Div: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] / b[p]; break;
        case BinaryOp::Pow: for (std::size_t p = 0; p < P; ++p) a[p] = rsymbolic::pow(a[p], b[p]); break;
    }
}

}  // namespace detail

// Evaluate the tree's value over points [lo, lo+P) in column-major input `cols`.
// Returns a pointer to the length-P result column inside `pool` (valid until the next
// call that reuses `pool`). `pool` and `stk` are reused scratch; sized on first use.
inline const double* evaluate_soa_residual(
    const Tree& tree,
    const std::vector<std::vector<double>>& cols,
    const double* constants,
    std::size_t lo, std::size_t P,
    std::vector<double>& pool, std::vector<double*>& stk) {
    pool.resize(tree.size() * P);
    stk.clear();
    std::size_t next_seg = 0;
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant: {
                double* s = pool.data() + (next_seg++) * P;
                const double v = constants[node.index];
                for (std::size_t p = 0; p < P; ++p) s[p] = v;
                stk.push_back(s);
                break;
            }
            case NodeKind::Variable: {
                double* s = pool.data() + (next_seg++) * P;
                const double* src = cols[static_cast<std::size_t>(node.index)].data() + lo;
                std::memcpy(s, src, P * sizeof(double));
                stk.push_back(s);
                break;
            }
            case NodeKind::Unary:
                detail::soa_res_unary(node.uop, stk.back(), P);
                break;
            case NodeKind::Binary: {
                double* b = stk.back(); stk.pop_back();
                double* a = stk.back();
                detail::soa_res_binary(node.bop, a, b, P);
                break;
            }
        }
    }
    return stk.back();
}

namespace detail {

// --- Jacobian (vector-mode dual) kernels --------------------------------------------
//
// A segment holds (1+N) contiguous length-P columns: value at offset 0, gradient lane c
// at offset (1+c)*P. Every per-element formula and its evaluation order match
// multi_dual.hpp EXACTLY (e.g. div/log/sqrt divide, never multiply by a reciprocal), so
// the per-entry result is bit-identical to the scalar k-pass / MultiDual path. The value
// column is updated LAST in every op so the gradient lanes read the pre-op value, exactly
// as the struct-based MultiDual reads `a.value` while producing `r.grad`.

template <int N>
inline void soa_jac_unary(UnaryOp op, double* s, std::size_t P, double* coeff) {
    auto val = s;
    auto g = [&](int c) { return s + static_cast<std::size_t>(1 + c) * P; };
    switch (op) {
        case UnaryOp::Neg:
            for (int c = 0; c < N; ++c) { double* gc = g(c); for (std::size_t p = 0; p < P; ++p) gc[p] = -gc[p]; }
            for (std::size_t p = 0; p < P; ++p) val[p] = -val[p];
            break;
        case UnaryOp::Exp:
            for (std::size_t p = 0; p < P; ++p) coeff[p] = std::exp(val[p]);
            for (int c = 0; c < N; ++c) { double* gc = g(c); for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] * coeff[p]; }
            for (std::size_t p = 0; p < P; ++p) val[p] = coeff[p];
            break;
        case UnaryOp::Log:
            for (int c = 0; c < N; ++c) { double* gc = g(c); for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] / val[p]; }
            for (std::size_t p = 0; p < P; ++p) val[p] = std::log(val[p]);
            break;
        case UnaryOp::Sin:
            for (std::size_t p = 0; p < P; ++p) coeff[p] = std::cos(val[p]);
            for (int c = 0; c < N; ++c) { double* gc = g(c); for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] * coeff[p]; }
            for (std::size_t p = 0; p < P; ++p) val[p] = std::sin(val[p]);
            break;
        case UnaryOp::Cos:
            for (std::size_t p = 0; p < P; ++p) coeff[p] = -std::sin(val[p]);
            for (int c = 0; c < N; ++c) { double* gc = g(c); for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] * coeff[p]; }
            for (std::size_t p = 0; p < P; ++p) val[p] = std::cos(val[p]);
            break;
        case UnaryOp::Sqrt:
            for (std::size_t p = 0; p < P; ++p) coeff[p] = std::sqrt(val[p] > 0.0 ? val[p] : 0.0);
            for (int c = 0; c < N; ++c) { double* gc = g(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = coeff[p] > 0.0 ? gc[p] / (2.0 * coeff[p]) : 0.0; }
            for (std::size_t p = 0; p < P; ++p) val[p] = coeff[p];
            break;
        case UnaryOp::Tanh:
            for (std::size_t p = 0; p < P; ++p) coeff[p] = std::tanh(val[p]);
            for (int c = 0; c < N; ++c) { double* gc = g(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] * (1.0 - coeff[p] * coeff[p]); }
            for (std::size_t p = 0; p < P; ++p) val[p] = coeff[p];
            break;
        case UnaryOp::Abs:
            for (int c = 0; c < N; ++c) { double* gc = g(c);
                for (std::size_t p = 0; p < P; ++p)
                    gc[p] = val[p] > 0.0 ? gc[p] : val[p] < 0.0 ? -gc[p] : 0.0; }
            for (std::size_t p = 0; p < P; ++p) val[p] = std::abs(val[p]);
            break;
        case UnaryOp::Square:
            for (int c = 0; c < N; ++c) { double* gc = g(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = (2.0 * val[p]) * gc[p]; }
            for (std::size_t p = 0; p < P; ++p) val[p] = val[p] * val[p];
            break;
    }
}

template <int N>
inline void soa_jac_binary(BinaryOp op, double* a, const double* b, std::size_t P, double* coeff) {
    auto av = a;
    auto ag = [&](int c) { return a + static_cast<std::size_t>(1 + c) * P; };
    const double* bv = b;
    auto bg = [&](int c) { return b + static_cast<std::size_t>(1 + c) * P; };
    switch (op) {
        case BinaryOp::Add:
            for (int c = 0; c < N; ++c) { double* gc = ag(c); const double* hc = bg(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] + hc[p]; }
            for (std::size_t p = 0; p < P; ++p) av[p] = av[p] + bv[p];
            break;
        case BinaryOp::Sub:
            for (int c = 0; c < N; ++c) { double* gc = ag(c); const double* hc = bg(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] - hc[p]; }
            for (std::size_t p = 0; p < P; ++p) av[p] = av[p] - bv[p];
            break;
        case BinaryOp::Mul:
            // r.grad[c] = a.grad[c]*b.value + a.value*b.grad[c]   (av/bv pre-op)
            for (int c = 0; c < N; ++c) { double* gc = ag(c); const double* hc = bg(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = gc[p] * bv[p] + av[p] * hc[p]; }
            for (std::size_t p = 0; p < P; ++p) av[p] = av[p] * bv[p];
            break;
        case BinaryOp::Div:
            // inv = 1/b.value; r.grad = (a.grad*b.value - a.value*b.grad)*inv*inv
            for (std::size_t p = 0; p < P; ++p) coeff[p] = 1.0 / bv[p];
            for (int c = 0; c < N; ++c) { double* gc = ag(c); const double* hc = bg(c);
                for (std::size_t p = 0; p < P; ++p)
                    gc[p] = (gc[p] * bv[p] - av[p] * hc[p]) * coeff[p] * coeff[p]; }
            for (std::size_t p = 0; p < P; ++p) av[p] = av[p] * coeff[p];
            break;
        case BinaryOp::Pow: {
            // Mirror multi_dual.hpp::pow: scalar coeffs p_val/dpdx/dpdy per point, then
            // r.grad[c] = dpdx*base.grad[c] + dpdy*exp.grad[c]; guarded branches -> 0.
            double* pv   = coeff;                 // p_val
            double* dpdx = coeff + P;             // dp/dx
            double* dpdy = coeff + 2 * P;         // dp/dy
            for (std::size_t p = 0; p < P; ++p) {
                const double x = av[p];
                const double y = bv[p];
                double pval, dx = 0.0, dy = 0.0;
                if (x > 0.0) {
                    pval = std::exp(y * std::log(x));
                    dx = y * std::exp((y - 1.0) * std::log(x));
                    dy = pval * std::log(x);
                } else if (x == 0.0 && y > 0.0) {
                    pval = 0.0;
                } else if (x < 0.0) {
                    const double yr = std::round(y);
                    pval = (std::fabs(y - yr) < 1e-6) ? std::pow(x, yr) : 0.0;
                } else {
                    pval = 0.0;
                }
                pv[p] = pval; dpdx[p] = dx; dpdy[p] = dy;
            }
            for (int c = 0; c < N; ++c) { double* gc = ag(c); const double* hc = bg(c);
                for (std::size_t p = 0; p < P; ++p) gc[p] = dpdx[p] * gc[p] + dpdy[p] * hc[p]; }
            for (std::size_t p = 0; p < P; ++p) av[p] = pv[p];
            break;
        }
    }
}

}  // namespace detail

// Evaluate one Jacobian block over points [lo, lo+P): differentiates the N constant
// directions block..block+N-1 in a single pass. Returns a pointer to the result
// segment's base in `pool`; the value column is at offset 0 and gradient lane c (i.e.
// d/d c_(block+c)) at offset (1+c)*P. Reads exactly w = min(N, k-block) meaningful
// lanes; inactive lanes (block+c >= k) carry zero throughout, matching the production
// MultiDual closure.
//
// `pool` must hold tree.size()*(1+N)*P doubles; `coeff` must hold 3*P doubles (pow uses
// three coefficient columns). Both are caller-owned scratch, sized on first use.
template <int N>
inline const double* evaluate_soa_jacobian(
    const Tree& tree,
    const std::vector<std::vector<double>>& cols,
    const double* params, int k, int block,
    std::size_t lo, std::size_t P,
    std::vector<double>& pool, std::vector<double*>& stk, std::vector<double>& coeff) {
    const std::size_t seg_stride = static_cast<std::size_t>(1 + N) * P;
    pool.resize(tree.size() * seg_stride);
    if (coeff.size() < 3 * P) coeff.resize(3 * P);
    stk.clear();
    std::size_t next_seg = 0;
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant: {
                double* s = pool.data() + (next_seg++) * seg_stride;
                const double v = params[node.index];
                for (std::size_t p = 0; p < P; ++p) s[p] = v;
                // Seed: lane c carries unit derivative iff this constant is c_(block+c).
                for (int c = 0; c < N; ++c) {
                    double* gc = s + static_cast<std::size_t>(1 + c) * P;
                    const double seed = (node.index == block + c) ? 1.0 : 0.0;
                    for (std::size_t p = 0; p < P; ++p) gc[p] = seed;
                }
                stk.push_back(s);
                break;
            }
            case NodeKind::Variable: {
                double* s = pool.data() + (next_seg++) * seg_stride;
                const double* src = cols[static_cast<std::size_t>(node.index)].data() + lo;
                std::memcpy(s, src, P * sizeof(double));  // value column
                for (int c = 0; c < N; ++c) {             // zero gradient
                    double* gc = s + static_cast<std::size_t>(1 + c) * P;
                    for (std::size_t p = 0; p < P; ++p) gc[p] = 0.0;
                }
                stk.push_back(s);
                break;
            }
            case NodeKind::Unary:
                detail::soa_jac_unary<N>(node.uop, stk.back(), P, coeff.data());
                break;
            case NodeKind::Binary: {
                double* b = stk.back(); stk.pop_back();
                double* a = stk.back();
                detail::soa_jac_binary<N>(node.bop, a, b, P, coeff.data());
                break;
            }
        }
    }
    (void)k;  // k bounds the meaningful lanes for the caller; unused here by design.
    return stk.back();
}

}  // namespace rsymbolic
