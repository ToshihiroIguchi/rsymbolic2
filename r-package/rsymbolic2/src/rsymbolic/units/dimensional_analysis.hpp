// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/units/dimension.hpp"

// Opt-in dimensional analysis, PySR-compatible (X_units / y_units /
// dimensional_constraint_penalty / dimensionless_constants_only). Off by default: with no
// units declared the search is byte-identical to the PySR-parity default (docs/46).
//
// The algorithm mirrors SymbolicRegression.jl's DimensionalAnalysis.jl: a bottom-up pass
// carrying a WildcardDim per node. A "wildcard" is a subtree whose dimension is not yet
// fixed (a free constant that can adopt whatever dimension makes its context consistent).
// `violates` taints a subtree and short-circuits upward. The check is structure-only (it
// needs no data row) and runs once per expression, O(nodes).

namespace rsymbolic {

struct WildcardDim {
    Dimension dim;          // the subtree's dimension (meaningful only when !wildcard)
    bool wildcard = false;  // true if the dimension is still free to unify
    bool violates = false;  // true once a dimensional inconsistency is seen below here
};

namespace units_detail {

inline WildcardDim eval_unary(UnaryOp op, const WildcardDim& a) {
    WildcardDim r;
    r.wildcard = a.wildcard;
    r.violates = a.violates;
    switch (op) {
        case UnaryOp::Neg:
        case UnaryOp::Abs:
            r.dim = a.dim;  // dimension-preserving
            break;
        case UnaryOp::Square:
            r.dim = a.dim.scaled(2);  // exponents double
            break;
        case UnaryOp::Sqrt: {
            Dimension out;
            if (a.dim.rooted(2, out)) {
                r.dim = out;  // exponents halve
            } else {
                r.violates = true;  // non-representable half-exponent
            }
            break;
        }
        // Transcendental / generic unary: the argument must be dimensionless (or a
        // wildcard, coerced to dimensionless), and the result is dimensionless.
        case UnaryOp::Exp:
        case UnaryOp::Log:
        case UnaryOp::Sin:
        case UnaryOp::Cos:
        case UnaryOp::Tanh:
            if (!(a.wildcard || a.dim.dimensionless())) r.violates = true;
            r.dim = Dimension{};  // dimensionless
            r.wildcard = false;   // output of a transcendental is concretely dimensionless
            break;
    }
    return r;
}

inline WildcardDim eval_binary(BinaryOp op, const WildcardDim& l, const WildcardDim& r) {
    WildcardDim out;
    out.violates = l.violates || r.violates;
    if (out.violates) return out;

    switch (op) {
        case BinaryOp::Mul:
            out.dim = l.dim + r.dim;  // multiply: add exponents
            out.wildcard = l.wildcard || r.wildcard;
            break;
        case BinaryOp::Div:
            out.dim = l.dim - r.dim;  // divide: subtract exponents
            out.wildcard = l.wildcard || r.wildcard;
            break;
        case BinaryOp::Add:
        case BinaryOp::Sub:
            if (l.wildcard && r.wildcard) {
                out.dim = l.dim;  // both free: keep one, stay wildcard
                out.wildcard = true;
            } else if (l.wildcard) {
                out.dim = r.dim;  // coerce the free side to the concrete side
                out.wildcard = false;
            } else if (r.wildcard) {
                out.dim = l.dim;
                out.wildcard = false;
            } else if (l.dim == r.dim) {
                out.dim = l.dim;
                out.wildcard = false;
            } else {
                out.violates = true;  // adding incompatible dimensions
            }
            break;
        case BinaryOp::Pow:
            // Generic power: both base and exponent must be dimensionless (or wildcard);
            // the result is dimensionless. (Dimensioned integer powers are expressed via
            // the Square unary op, which is handled dimension-aware above.)
            if (!((l.wildcard || l.dim.dimensionless()) &&
                  (r.wildcard || r.dim.dimensionless()))) {
                out.violates = true;
            }
            out.dim = Dimension{};
            out.wildcard = false;
            break;
    }
    return out;
}

}  // namespace units_detail

// True if `tree` violates the declared dimensional constraints. `x_units` must cover every
// feature index the tree references (size == num_features); `y_units` is the required
// dimension of the whole expression (unset => only internal consistency is checked).
// `allow_wildcards` == !dimensionless_constants_only.
inline bool violates_dimensions(const Tree& tree,
                                const std::vector<Dimension>& x_units,
                                const std::optional<Dimension>& y_units,
                                bool allow_wildcards) {
    std::vector<WildcardDim> stack;
    stack.reserve(tree.size());
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant:
                stack.push_back(WildcardDim{Dimension{}, allow_wildcards, false});
                break;
            case NodeKind::Variable: {
                WildcardDim w;
                const std::size_t idx = static_cast<std::size_t>(node.index);
                if (idx < x_units.size()) w.dim = x_units[idx];  // else dimensionless (guard)
                stack.push_back(w);
                break;
            }
            case NodeKind::Unary:
                stack.back() = units_detail::eval_unary(node.uop, stack.back());
                break;
            case NodeKind::Binary: {
                const WildcardDim b = stack.back();
                stack.pop_back();
                stack.back() = units_detail::eval_binary(node.bop, stack.back(), b);
                break;
            }
        }
    }
    if (stack.empty()) return false;
    const WildcardDim& root = stack.back();
    if (root.violates) return true;
    if (y_units.has_value() && !root.wildcard && root.dim != *y_units) return true;
    return false;
}

// The per-search dimensional-analysis context, built once in run_evolution. When
// `enabled` is false, add_dim_penalty is a no-op and the loss path is byte-identical to
// the units-off default.
struct DimAnalysis {
    bool enabled = false;
    std::vector<Dimension> x_units;      // size == num_features when enabled
    std::optional<Dimension> y_units;    // required output dimension (unset = arbitrary)
    bool allow_wildcards = true;         // = !dimensionless_constants_only
    double penalty_sse = 0.0;            // dimensional_constraint_penalty * W (see below)
};

// Add the flat dimensional penalty to an SSE-scale loss when the expression violates its
// declared units, mirroring PySR (the penalty rides inside the loss so it also keeps
// violators out of the hall of fame / final model, not just out of selection).
//
// Normalization: PySR adds `dimensional_constraint_penalty` to the per-sample loss (MSE
// scale) before dividing by the baseline (= SST/n). rsymbolic2 works in SSE and divides by
// SST, so the equivalent additive term in SSE units is `penalty * W`, with W = n (sum of
// weights when weighted). Then loss/SST gains `penalty*W/SST`, matching PySR's
// `penalty*n/baseline`. penalty_sse already folds in W (built in run_evolution).
inline double add_dim_penalty(double base_loss, const Tree& tree, const DimAnalysis& da) {
    if (!da.enabled || !std::isfinite(base_loss)) return base_loss;
    return violates_dimensions(tree, da.x_units, da.y_units, da.allow_wildcards)
               ? base_loss + da.penalty_sse
               : base_loss;
}

}  // namespace rsymbolic
