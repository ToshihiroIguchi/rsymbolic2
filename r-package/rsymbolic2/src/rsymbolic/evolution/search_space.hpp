// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <optional>
#include <vector>

#include "rsymbolic/evolution/macro_op.hpp"
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/units/dimension.hpp"

namespace rsymbolic {

// Describes the building blocks the search may use and the shape of random trees.
// Kept deliberately small for the walking skeleton: the default operator set is
// {+, -, *} with no unary operators, which is enough to express (and recover) simple
// targets like y = a*x + b without risking non-finite values from division or exp.
struct SearchSpace {
    std::vector<UnaryOp> unary_ops;
    std::vector<BinaryOp> binary_ops{BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    int num_features = 1;        // number of input variables (x0, x1, ...)
    double const_min = -2.0;     // range for randomly generated constants
    double const_max = 2.0;
    int max_depth = 30;          // PySR maxdepth (None -> maxsize = 30); docs/28 §A
    int max_nodes = 30;          // PySR maxsize = 30 (soft cap on tree size); docs/28 §A
    double terminal_prob = 0.3;  // probability of stopping at a leaf before max_depth
    double const_prob = 0.5;     // probability a leaf is a constant (vs a variable)

    // Opt-in user-defined macro operators (docs/57): single-argument templates over the
    // primitive operators, expanded when a growth mutation creates a unary node. Empty (the
    // default) is the off-switch — the unary alphabet is then exactly `unary_ops` and the
    // search is bit-identical to the PySR-parity default.
    std::vector<MacroOp> macro_ops;

    // Opt-in dimensional analysis (PySR X_units / y_units / dimensionless_constants_only;
    // docs/46). All default-off: `x_units` empty means the feature is disabled and the
    // search is byte-identical to the units-off PySR-parity default.
    std::vector<Dimension> x_units;        // empty = off; else size == num_features
    std::optional<Dimension> y_units;      // required output dimension (unset = arbitrary)
    bool dimensionless_constants_only = false;
};

}  // namespace rsymbolic
