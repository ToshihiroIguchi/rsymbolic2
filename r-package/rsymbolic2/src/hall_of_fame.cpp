// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/evolution/hall_of_fame.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace rsymbolic {

void HallOfFame::merge(const HallOfFame& other) {
    for (const auto& entry : other.by_complexity_) {
        update(entry.second);
    }
}

void HallOfFame::update(const PopMember& member) {
    auto it = by_complexity_.find(member.complexity);
    if (it == by_complexity_.end() || member.loss < it->second.loss) {
        by_complexity_[member.complexity] = member;
    }
}

PopMember HallOfFame::best() const {
    if (by_complexity_.empty()) {
        throw std::runtime_error("HallOfFame::best() called on an empty hall of fame");
    }
    const PopMember* best = nullptr;
    for (const auto& entry : by_complexity_) {
        if (best == nullptr || entry.second.loss < best->loss) {
            best = &entry.second;
        }
    }
    return *best;
}

std::vector<PopMember> HallOfFame::members() const {
    std::vector<PopMember> out;
    out.reserve(by_complexity_.size());
    for (const auto& entry : by_complexity_) out.push_back(entry.second);
    return out;
}

std::vector<PopMember> HallOfFame::pareto_front() const {
    // Iterate complexities ascending; keep an entry only if its loss strictly improves
    // on the best loss seen at any smaller complexity. This yields the non-dominated
    // set (lower complexity is better, lower loss is better).
    std::vector<PopMember> front;
    double best_loss = std::numeric_limits<double>::infinity();
    for (const auto& entry : by_complexity_) {
        if (entry.second.loss < best_loss) {
            front.push_back(entry.second);
            best_loss = entry.second.loss;
        }
    }
    return front;
}

std::vector<double> pareto_scores(const std::vector<PopMember>& front) {
    // Floor losses before taking logs so an exact (zero-loss) fit yields a large,
    // finite score rather than +inf, and so a degenerate zero never divides.
    constexpr double kLossFloor = 1e-300;

    std::vector<double> scores(front.size(), 0.0);
    for (std::size_t i = 1; i < front.size(); ++i) {
        const int dc = front[i].complexity - front[i - 1].complexity;
        if (dc <= 0) {
            scores[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        const double prev = std::max(front[i - 1].loss, kLossFloor);
        const double cur  = std::max(front[i].loss, kLossFloor);
        scores[i] = (std::log(prev) - std::log(cur)) / static_cast<double>(dc);
    }
    return scores;
}

std::size_t select_best(const std::vector<PopMember>& front, ModelSelection mode) {
    if (front.size() <= 1) return 0;

    // Accuracy: the lowest-loss member. The front is strictly decreasing in loss, so the
    // most accurate is the last (most complex) entry. PySR model_selection="accuracy".
    if (mode == ModelSelection::Accuracy) return front.size() - 1;

    // PySR "best" considers only members whose loss is within 1.5x of the most accurate
    // (lowest) loss; "score" considers the entire front. The front's lowest loss is its
    // last entry. The threshold is +inf for "score" so every member qualifies.
    const double min_loss  = front.back().loss;
    const double threshold = (mode == ModelSelection::Best)
                                 ? 1.5 * min_loss
                                 : std::numeric_limits<double>::infinity();

    // Rank by the same per-member scores exposed to callers, so the recommended index
    // and the displayed score column agree by construction. scores[0] is 0.0 by
    // convention and NaN marks a malformed (non-ascending) step; both lose every
    // `score > best_score` comparison, matching the previous skip behaviour.
    const std::vector<double> scores = pareto_scores(front);
    std::size_t best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < front.size(); ++i) {
        if (front[i].loss > threshold) continue;  // outside the accuracy band ("best")
        if (scores[i] > best_score) { best_score = scores[i]; best_idx = i; }
    }
    return best_idx;
}

}  // namespace rsymbolic
