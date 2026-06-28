// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/evolution/crossover.hpp"

#include <utility>

namespace rsymbolic {

namespace {

// Splice donor[donor_start..donor_end] into base in place of base[cut_start..cut_end],
// preserving postfix order. The caller re-indexes constants. Net size:
//   base.size() - (cut_end - cut_start + 1) + (donor_end - donor_start + 1).
Tree splice_subtree(const Tree& base, std::size_t cut_start, std::size_t cut_end,
                    const Tree& donor, std::size_t donor_start, std::size_t donor_end) {
    Tree child;
    child.reserve(base.size() - (cut_end - cut_start + 1) +
                  (donor_end - donor_start + 1));
    child.insert(child.end(), base.begin(), base.begin() + cut_start);
    child.insert(child.end(), donor.begin() + donor_start, donor.begin() + donor_end + 1);
    child.insert(child.end(), base.begin() + cut_end + 1, base.end());
    return child;
}

}  // namespace

Tree subtree_crossover(const Tree& pa, const Tree& pb,
                       std::mt19937_64& rng, int max_nodes) {
    if (pa.empty() || pb.empty()) return pa;

    // Select the crossover point in pa: the node at cut_end is the root of the subtree
    // that will be replaced, and [cut_start, cut_end] is its full range.
    const std::size_t cut_end =
        std::uniform_int_distribution<std::size_t>(0, pa.size() - 1)(rng);
    const std::size_t cut_start = subtree_start(pa, cut_end);

    // Select the donor subtree root in pb: [donor_start, donor_end].
    const std::size_t donor_end =
        std::uniform_int_distribution<std::size_t>(0, pb.size() - 1)(rng);
    const std::size_t donor_start = subtree_start(pb, donor_end);

    Tree child = splice_subtree(pa, cut_start, cut_end, pb, donor_start, donor_end);
    if (static_cast<int>(child.size()) > max_nodes) return pa;

    reindex_constants(child);
    return child;
}

CrossoverChildren subtree_crossover_pair(const Tree& pa, const Tree& pb,
                                         std::mt19937_64& rng, int max_nodes,
                                         int max_tries) {
    if (pa.empty() || pb.empty()) return {Tree{}, Tree{}, false};

    std::uniform_int_distribution<std::size_t> pick_a(0, pa.size() - 1);
    std::uniform_int_distribution<std::size_t> pick_b(0, pb.size() - 1);

    for (int attempt = 0; attempt < max_tries; ++attempt) {
        // One pair of cut points, swapped both ways (SR.jl crossover_trees).
        const std::size_t a_end   = pick_a(rng);
        const std::size_t a_start = subtree_start(pa, a_end);
        const std::size_t b_end   = pick_b(rng);
        const std::size_t b_start = subtree_start(pb, b_end);

        Tree c1 = splice_subtree(pa, a_start, a_end, pb, b_start, b_end);  // pa: a-subtree <- b
        Tree c2 = splice_subtree(pb, b_start, b_end, pa, a_start, a_end);  // pb: b-subtree <- a

        // SR.jl breeds until BOTH children satisfy the constraints (here: the size cap).
        if (static_cast<int>(c1.size()) <= max_nodes &&
            static_cast<int>(c2.size()) <= max_nodes) {
            reindex_constants(c1);
            reindex_constants(c2);
            return {std::move(c1), std::move(c2), true};
        }
    }
    return {Tree{}, Tree{}, false};
}

}  // namespace rsymbolic
