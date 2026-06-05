#pragma once

#include <random>

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Perturb a randomly chosen constant's value. Returns false if there are no constants.
// Structure and parameter indices are preserved.
bool mutate_constant(Tree& tree, std::mt19937_64& rng, double scale);

// Replace a randomly chosen operator with another of the same arity from the search
// space. Returns false if there are no operator nodes. Structure is preserved.
bool mutate_operator(Tree& tree, const SearchSpace& space, std::mt19937_64& rng);

// Replace the entire tree with a freshly generated random tree. Always succeeds and is
// the source of structural novelty in the minimal mutation set.
void randomize_tree(Tree& tree, const SearchSpace& space, std::mt19937_64& rng);

// Apply a single mutation, chosen among the feasible kinds (constant / operator /
// randomize). Randomize is always feasible; the others depend on the tree's content.
void mutate(Tree& tree, const SearchSpace& space, std::mt19937_64& rng,
            double const_perturb_scale = 0.5);

}  // namespace rsymbolic
