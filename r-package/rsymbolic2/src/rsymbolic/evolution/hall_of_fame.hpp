// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// A population member: a fitted expression with its loss and complexity.
struct PopMember {
    Tree tree;
    double loss = 0.0;     // sum of squared residuals after constant optimization
    int complexity = 0;    // node count
    // Monotonic age stamp for regularized-evolution replacement (SR.jl reg_evol_cycle
    // replaces the OLDEST population member, i.e. the smallest birth). Assigned from a
    // per-island counter; ignored by the hall of fame and migration, which rank by
    // loss/cost. See evolutionary_search.cpp.
    std::uint64_t birth = 0;
};

// Tracks the best member found at each complexity level and exposes the non-dominated
// (Pareto) front of accuracy vs. complexity. Acts as the elitist archive: the best
// expression survives even if it is later removed from the working population.
class HallOfFame {
public:
    // Insert a candidate, keeping only the lowest-loss member at each complexity.
    void update(const PopMember& member);

    // Merge all entries from another HallOfFame into this one.
    void merge(const HallOfFame& other);

    bool empty() const { return by_complexity_.empty(); }

    // The single lowest-loss member across all complexities.
    PopMember best() const;

    // The non-dominated front: ascending complexity with strictly decreasing loss.
    std::vector<PopMember> pareto_front() const;

    // All archived members (one per complexity level), in ascending complexity order.
    // Used by the periodic constant re-optimization pass, which re-fits each member and
    // re-inserts the improved copy via update().
    std::vector<PopMember> members() const;

private:
    std::map<int, PopMember> by_complexity_;
};

// PySR model_selection criterion for picking the recommended member off the Pareto front.
// Mirrors PySRRegressor(model_selection=...): "best" (default), "accuracy", "score".
enum class ModelSelection {
    Best,      // highest score among members within 1.5x of the most accurate loss
    Accuracy,  // lowest-loss member (the most accurate, usually most complex)
    Score,     // highest score over the whole front (no accuracy filter)
};

// Per-member selection score for a Pareto `front` (ascending complexity, strictly
// decreasing loss, as returned by HallOfFame::pareto_front). scores[i] is the drop in
// log-loss per unit of added complexity relative to the next-simpler member:
//   scores[0] = 0.0 (no simpler predecessor; PySR convention),
//   scores[i] = (log(max(loss[i-1], 1e-300)) - log(max(loss[i], 1e-300))) / dc,
//   scores[i] = NaN when dc = complexity[i] - complexity[i-1] <= 0 (malformed front).
// Losses are floored at 1e-300 so a zero-loss member scores large-but-finite rather
// than +inf (PySR displays inf there; we keep the value select_best actually ranks by).
std::vector<double> pareto_scores(const std::vector<PopMember>& front);

// Index of the "recommended" member on a Pareto `front` (ascending complexity, strictly
// decreasing loss, as returned by HallOfFame::pareto_front), per PySR's model_selection:
//   * Score    — the member maximising the drop in log-loss per unit of added complexity
//                relative to the next-simpler member (the accuracy/complexity "knee").
//   * Best     — the same score, but considered only among members whose loss is within
//                1.5x of the front's most accurate (lowest) loss; PySR's default.
//   * Accuracy — the lowest-loss member (front.back()).
// Returns 0 for an empty or single-element front. Default mode is Best (PySR default).
std::size_t select_best(const std::vector<PopMember>& front,
                        ModelSelection mode = ModelSelection::Best);

}  // namespace rsymbolic
