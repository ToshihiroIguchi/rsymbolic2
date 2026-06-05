#pragma once

#include <random>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Subtree crossover: replace a random subtree in parent_a with a random subtree from
// parent_b.  The crossover point (root of the replaced subtree) and the donor point
// (root of the inserted subtree) are both drawn uniformly over all nodes in their
// respective trees.
//
// If the resulting tree would exceed max_nodes, parent_a is returned unchanged (the
// size guard acts as the standard "bloat control" in GP crossover).
//
// Constant indices are reassigned to 0..k-1 after the structural edit.
Tree subtree_crossover(const Tree& parent_a, const Tree& parent_b,
                       std::mt19937_64& rng, int max_nodes);

}  // namespace rsymbolic
