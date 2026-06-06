#pragma once

#include <random>

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Generate a random expression tree within the given search space. Constant nodes are
// assigned contiguous parameter indices (0..k-1) in left-to-right order, matching the
// convention assumed by count_constants / evaluate.
Tree generate_random_tree(const SearchSpace& space, std::mt19937_64& rng);

}  // namespace rsymbolic
