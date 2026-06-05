#pragma once

#include <vector>

#include "rsymbolic/expression/node.hpp"

namespace rsymbolic {

// Describes the building blocks the search may use and the shape of random trees.
// Kept deliberately small for the walking skeleton: the default operator set is
// {+, -, *} with no unary operators, which is enough to express (and recover) simple
// targets like y = a*x + b without risking non-finite values from division or exp.
struct SearchSpace {
    std::vector<UnaryOp> unary_ops;
    std::vector<BinaryOp> binary_ops{BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    int num_features = 1;        // number of input variables (x0, x1, ...)
    double const_min = -2.0;     // range for randomly generated constants
    double const_max = 2.0;
    int max_depth = 3;           // maximum tree depth when generating
    double terminal_prob = 0.3;  // probability of stopping at a leaf before max_depth
    double const_prob = 0.5;     // probability a leaf is a constant (vs a variable)
};

}  // namespace rsymbolic
