#pragma once

#include <cstdint>

namespace rsymbolic {

enum class NodeKind : std::uint8_t {
    Constant,  // a tunable constant; `index` selects its slot in the constants vector
    Variable,  // an input feature; `index` selects the column of the data row
    Unary,     // a unary operator (`uop`)
    Binary,    // a binary operator (`bop`)
};

enum class UnaryOp : std::uint8_t { Neg, Exp, Log, Sin, Cos, Sqrt, Tanh, Abs };

enum class BinaryOp : std::uint8_t { Add, Sub, Mul, Div };

// A single node of an expression tree. Trees are stored as a flat vector in postfix
// (reverse Polish) order, so a stack-based evaluator can run with no pointer chasing
// (see tree.hpp). This is the minimal node needed for the walking skeleton; it will be
// extended as more operators and features are required.
struct Node {
    NodeKind kind = NodeKind::Constant;
    int index = 0;          // Constant: parameter slot; Variable: feature column
    double value = 0.0;     // Constant: initial value (ignored for other kinds)
    UnaryOp uop = UnaryOp::Neg;
    BinaryOp bop = BinaryOp::Add;
};

// Readable builders for assembling trees in postfix order.
inline Node constant_node(int parameter_index, double initial_value) {
    Node n;
    n.kind = NodeKind::Constant;
    n.index = parameter_index;
    n.value = initial_value;
    return n;
}

inline Node variable_node(int feature_index) {
    Node n;
    n.kind = NodeKind::Variable;
    n.index = feature_index;
    return n;
}

inline Node unary_node(UnaryOp op) {
    Node n;
    n.kind = NodeKind::Unary;
    n.uop = op;
    return n;
}

inline Node binary_node(BinaryOp op) {
    Node n;
    n.kind = NodeKind::Binary;
    n.bop = op;
    return n;
}

}  // namespace rsymbolic
