#pragma once

#include <map>
#include <vector>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// A population member: a fitted expression with its loss and complexity.
struct PopMember {
    Tree tree;
    double loss = 0.0;     // sum of squared residuals after constant optimization
    int complexity = 0;    // node count
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

private:
    std::map<int, PopMember> by_complexity_;
};

}  // namespace rsymbolic
