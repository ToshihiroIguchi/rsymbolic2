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
// All limits are hard caps; saturation is best-effort within them. Both caps are counts,
// which is the point: they make the stop a function of the INPUT TREE ALONE, so the same
// tree always renders the same way (docs/66). An earlier revision also carried a
// wall-clock safety net, and it bound often enough at these sizes to make
// `expression_simplified` differ between two runs of the same fixed-seed search on a busy
// machine. Do not reintroduce a clock reading, a deadline, or any other stop condition
// that a second run could evaluate differently; bound the cost with the counts below.
//
// The counts have to carry the cost bound alone, so they are set from the measured tail
// rather than from what looks generous (docs/66 §3, bench_simplify). Matching is
// superlinear in the class count, so the previous e-node cap of 10000 had a heavy tail —
// 615 ms on the worst of 2000 random 60-node trees, which the 10 ms net was silently
// truncating. Cutting it to 2000 bounds the worst case at ~23 ms, a 27x improvement, and
// costs almost nothing: Layer-2 adoption is unchanged at 30 and 60 nodes and drops 0.6
// percentage points at 120. The two caps are NOT interchangeable here — tightening
// max_iterations instead (to 6) buys less and costs ~2.6 points of adoption, because the
// tail is driven by e-matching over the class count, not by the iteration count.
// Display simplification runs once per Pareto member at finalisation, so the ceiling that
// matters is (front size) x this.
struct EGraphLimits {
    int max_iterations = 10;  // equality-saturation iterations
    int max_enodes = 2000;    // e-graph size cap (distinct canonical e-nodes)
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
