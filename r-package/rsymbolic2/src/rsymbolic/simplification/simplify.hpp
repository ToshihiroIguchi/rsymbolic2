#pragma once

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Algebraic simplification of an expression tree. This is the minimal, semantics-
// preserving rule set for the walking skeleton:
//   - constant folding: a subtree of only constants collapses to one constant
//   - identity elimination: x+0, x-0, x*1, x/1 -> x; x*0 -> 0
//   - double negation: neg(neg(x)) -> x
//   - unary folding: op(const) -> const
//
// It does NOT canonicalize or reassociate (e.g. it will not merge two constants that
// are separated by a variable across associative operators); reaching a globally
// minimal form is deferred to a later e-graph-based pass. Constant nodes in the result
// are re-indexed to a contiguous 0..k-1 range. The returned tree evaluates to the same
// value as the input (up to floating-point rounding of folded constants).
Tree simplify(const Tree& tree);

}  // namespace rsymbolic
