// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Cost and reproducibility probe for the display-only simplifier (docs/54, docs/66).
//
// Two questions, one harness:
//
//  1. **Cost.** display_simplify()'s Layer-2 e-graph is bounded by an iteration cap and an
//     e-node cap. Removing the wall-clock cap (docs/66) makes those caps the only stops, so
//     the worst-case per-call time has to be known rather than assumed — especially as the
//     tree grows, since `maxsize` is user-settable and the e-node cap does not bind at the
//     default size. Reports the per-call distribution (p50/p90/p99/max), not just a mean.
//
//  2. **Reproducibility.** Every tree is simplified twice and the two renderings are
//     compared. Under a wall-clock budget this mismatch count is nonzero on a busy machine
//     (that is the bug docs/66 fixes); with iteration/e-node caps alone it must be 0.
//
// The limits are set field-by-field and never mention a time budget, so this file compiles
// against both the pre-fix and post-fix EGraphLimits — which is what lets the same binary
// source produce the before and after rows of the docs/66 table.
//
// Usage (from the build directory):
//   ./standalone/bench_simplify                      # 2000 trees x 30 nodes, {10, 10000}
//   ./standalone/bench_simplify 2000 120 10 10000 7  # trees nodes iters enodes seed
//
// Trees come from gen_random_tree_fixed_size over the full operator set, so `nodes` is the
// exact node count, matching how `maxsize` bounds what the display layer is ever handed.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/simplification/display_simplify.hpp"

namespace {

using rsymbolic::EGraphLimits;
using rsymbolic::SearchSpace;
using rsymbolic::Tree;

// Every operator the core knows, so the rule set is exercised as widely as possible.
SearchSpace full_space(int nodes) {
    SearchSpace space;
    space.binary_ops = {rsymbolic::BinaryOp::Add, rsymbolic::BinaryOp::Sub,
                        rsymbolic::BinaryOp::Mul, rsymbolic::BinaryOp::Div,
                        rsymbolic::BinaryOp::Pow};
    space.unary_ops = {rsymbolic::UnaryOp::Neg,    rsymbolic::UnaryOp::Sin,
                       rsymbolic::UnaryOp::Cos,    rsymbolic::UnaryOp::Exp,
                       rsymbolic::UnaryOp::Log,    rsymbolic::UnaryOp::Sqrt,
                       rsymbolic::UnaryOp::Square, rsymbolic::UnaryOp::Abs,
                       rsymbolic::UnaryOp::Tanh,   rsymbolic::UnaryOp::Erf,
                       rsymbolic::UnaryOp::Sinh,   rsymbolic::UnaryOp::Cosh,
                       rsymbolic::UnaryOp::Inv};
    space.num_features = 3;
    space.max_nodes = nodes;
    space.max_depth = nodes;  // size-driven generation; depth must not bind first
    return space;
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const std::size_t i = static_cast<std::size_t>(q * static_cast<double>(v.size() - 1));
    return v[i];
}

int arg_int(int argc, char** argv, int i, int fallback) {
    return (argc > i) ? std::atoi(argv[i]) : fallback;
}

}  // namespace

int main(int argc, char** argv) {
    const int n_trees = arg_int(argc, argv, 1, 2000);
    const int nodes = arg_int(argc, argv, 2, 30);
    const int iters = arg_int(argc, argv, 3, 10);
    const int enodes = arg_int(argc, argv, 4, 10000);
    const int seed = arg_int(argc, argv, 5, 20260728);

    EGraphLimits limits;
    limits.max_iterations = iters;
    limits.max_enodes = enodes;

    const SearchSpace space = full_space(nodes);
    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));

    std::vector<double> millis;
    millis.reserve(static_cast<std::size_t>(n_trees));
    int adopted = 0;
    int mismatches = 0;
    double total_ms = 0.0;
    double total_out_nodes = 0.0;

    for (int t = 0; t < n_trees; ++t) {
        const Tree tree = rsymbolic::gen_random_tree_fixed_size(nodes, space, rng);

        rsymbolic::DisplaySimplifyStats stats;
        const auto t0 = std::chrono::steady_clock::now();
        const Tree s = rsymbolic::display_simplify(tree, &stats, limits);
        const std::chrono::duration<double, std::milli> dt =
            std::chrono::steady_clock::now() - t0;
        millis.push_back(dt.count());
        total_ms += dt.count();

        // Quality is measured as what LAYER 2 contributes, not as "did the tree shrink":
        // Layer 1 alone shrinks nearly every random tree, so a raw shrink rate is ~100%
        // regardless of the caps and cannot discriminate between them. Adoption rate and
        // mean output size are what actually move when the caps are tightened.
        if (stats.layer2_adopted) ++adopted;
        total_out_nodes += static_cast<double>(s.size());

        // Reproducibility probe: the same input must render identically every time.
        const Tree again = rsymbolic::display_simplify(tree, nullptr, limits);
        if (rsymbolic::to_string(s) != rsymbolic::to_string(again)) ++mismatches;
    }

    std::printf("trees        : %d x %d nodes (seed %d)\n", n_trees, nodes, seed);
    std::printf("limits       : max_iterations=%d max_enodes=%d\n", iters, enodes);
    std::printf("per call ms  : p50 %.4f  p90 %.4f  p99 %.4f  max %.4f\n",
                quantile(millis, 0.50), quantile(millis, 0.90), quantile(millis, 0.99),
                quantile(millis, 1.00));
    std::printf("total ms     : %.1f\n", total_ms);
    std::printf("layer2 adopt : %.1f%% (%d/%d)\n",
                100.0 * adopted / (n_trees > 0 ? n_trees : 1), adopted, n_trees);
    std::printf("mean out nodes: %.3f (from %d)\n",
                total_out_nodes / (n_trees > 0 ? n_trees : 1), nodes);
    std::printf("mismatches   : %d  (must be 0: same input, same rendering)\n", mismatches);
    return mismatches == 0 ? 0 : 1;
}
