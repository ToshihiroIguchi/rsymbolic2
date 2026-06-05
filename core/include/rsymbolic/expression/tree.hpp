#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/node.hpp"

namespace rsymbolic {

// An expression tree: a flat vector of nodes in postfix order.
using Tree = std::vector<Node>;

namespace detail {

template <typename T>
T apply_unary(UnaryOp op, const T& a) {
    using std::cos;
    using std::exp;
    using std::log;
    using std::sin;
    switch (op) {
        case UnaryOp::Neg:
            return -a;
        case UnaryOp::Exp:
            return exp(a);  // ADL: rsymbolic::exp for Dual, std::exp for double
        case UnaryOp::Log:
            return log(a);
        case UnaryOp::Sin:
            return sin(a);
        case UnaryOp::Cos:
            return cos(a);
    }
    return a;  // unreachable
}

template <typename T>
T apply_binary(BinaryOp op, const T& a, const T& b) {
    switch (op) {
        case BinaryOp::Add:
            return a + b;
        case BinaryOp::Sub:
            return a - b;
        case BinaryOp::Mul:
            return a * b;
        case BinaryOp::Div:
            return a / b;
    }
    return a;  // unreachable
}

}  // namespace detail

// Evaluate a postfix tree.
//
//   T          - double for a plain value; Dual for a value + one directional
//                derivative (forward-mode AD).
//   row        - the input feature values for one data point (indexed by Variable
//                nodes); always plain doubles.
//   constants  - the tunable constant values (indexed by Constant nodes), as T so a
//                constant can be seeded with a derivative in the Dual case.
//
// Assumes a well-formed tree (every operator has its operands available on the stack).
template <typename T>
T evaluate(const Tree& tree, const double* row, const T* constants) {
    std::vector<T> stack;
    stack.reserve(tree.size());
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant:
                stack.push_back(constants[node.index]);
                break;
            case NodeKind::Variable:
                stack.push_back(T(row[node.index]));
                break;
            case NodeKind::Unary: {
                T a = stack.back();
                stack.back() = detail::apply_unary<T>(node.uop, a);
                break;
            }
            case NodeKind::Binary: {
                T b = stack.back();
                stack.pop_back();
                T a = stack.back();
                stack.back() = detail::apply_binary<T>(node.bop, a, b);
                break;
            }
        }
    }
    return stack.back();
}

// Number of distinct tunable constants (assumes parameter indices are 0..k-1).
inline int count_constants(const Tree& tree) {
    int max_index = -1;
    for (const Node& node : tree) {
        if (node.kind == NodeKind::Constant && node.index > max_index) {
            max_index = node.index;
        }
    }
    return max_index + 1;
}

// The initial constant vector implied by the Constant nodes' `value` fields.
inline std::vector<double> initial_constants(const Tree& tree) {
    std::vector<double> values(static_cast<std::size_t>(count_constants(tree)), 0.0);
    for (const Node& node : tree) {
        if (node.kind == NodeKind::Constant) {
            values[static_cast<std::size_t>(node.index)] = node.value;
        }
    }
    return values;
}

}  // namespace rsymbolic
