#include "rsymbolic/evolution/crossover.hpp"

namespace rsymbolic {

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

    const std::size_t cut_len   = cut_end   - cut_start   + 1;
    const std::size_t donor_len = donor_end - donor_start + 1;
    const std::size_t child_size = pa.size() - cut_len + donor_len;

    if (static_cast<int>(child_size) > max_nodes) return pa;

    Tree child;
    child.reserve(child_size);
    child.insert(child.end(), pa.begin(), pa.begin() + cut_start);
    child.insert(child.end(), pb.begin() + donor_start, pb.begin() + donor_end + 1);
    child.insert(child.end(), pa.begin() + cut_end + 1, pa.end());

    reindex_constants(child);
    return child;
}

}  // namespace rsymbolic
