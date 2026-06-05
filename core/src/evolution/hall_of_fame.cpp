#include "rsymbolic/evolution/hall_of_fame.hpp"

#include <limits>
#include <stdexcept>

namespace rsymbolic {

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

}  // namespace rsymbolic
