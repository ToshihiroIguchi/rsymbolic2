// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Bounded equality saturation over an e-graph (egg-style: hash-consing + union-find +
// e-matching; Willsey et al., POPL 2021), used as the second layer of the DISPLAY-ONLY
// simplifier (docs/54). Never called from the search loop; the search's own parity
// simplifier (simplify.hpp) is untouched by anything in this file.
//
// The rewrite-rule set lives in egraph.cpp. Every rule is an exact identity over the
// reals on the expression's evaluation domain, restricted further by the display
// layer's floating-point policy (docs/54): no rule may turn a NaN/Inf evaluation into
// a finite one or vice versa (so x*0 -> 0, exp(log x) -> x, etc. are excluded), and
// constant folding only fires when the folded value is finite. Rules that merely
// reassociate/redistribute floating-point operations (drift by a rounding step, the
// same caveat simplify()'s combine pass already carries) are allowed.
//
// All limits are hard caps; saturation is best-effort within them. Iteration and
// e-node caps are deterministic; the wall-clock cap is a safety net that is not
// expected to bind at these sizes (hitting it is the only non-deterministic stop).
struct EGraphLimits {
    int max_iterations = 10;   // equality-saturation iterations
    int max_enodes = 10000;    // e-graph size cap (distinct canonical e-nodes)
    double max_millis = 10.0;  // wall-clock safety net per expression
};

struct EGraphResult {
    Tree tree;               // minimum-node-count equivalent of the input (when ok)
    int iterations = 0;      // saturation iterations actually run
    int enodes = 0;          // canonical e-nodes at stop
    bool saturated = false;  // true: reached a rewrite fixpoint within the limits
    bool ok = false;         // extraction produced a tree (false only on empty input)
};

// Saturate the e-graph seeded with `tree` under `limits`, then extract the equivalent
// tree with the fewest nodes (= SR complexity). The result is never larger than the
// input (the input itself is in the graph). Constants in the result are re-indexed
// 0..k-1. The caller (display_simplify) decides whether to adopt the result or fall
// back to its Layer-1 form.
EGraphResult egraph_simplify(const Tree& tree, const EGraphLimits& limits);

}  // namespace rsymbolic
