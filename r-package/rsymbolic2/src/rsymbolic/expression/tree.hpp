// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/platform/libm.hpp"

namespace rsymbolic {

// An expression tree: a flat vector of nodes in postfix order.
using Tree = std::vector<Node>;

namespace detail {

template <typename T>
T apply_unary(UnaryOp op, const T& a) {
    // The redirected six come from libm:: rather than std:: (rsymbolic/platform/libm.hpp);
    // off MinGW they ARE std::. Dual/MultiDual overloads still win via ADL for those types.
    using libm::cos;
    using libm::erf;
    using libm::exp;
    using libm::log;
    using libm::sin;
    using std::abs;
    using std::cosh;
    using std::sinh;
    using std::tanh;
    // NOTE: no `using std::sqrt`. Sqrt resolves to rsymbolic::sqrt — the safe_sqrt in
    // dual.hpp — for double as well as for Dual/MultiDual, exactly like square/recip/pow
    // below. It used to pull in std::sqrt here, which made this path the one place in the
    // codebase that returned NaN for a negative argument while the shipped SoA evaluator
    // returned 0. Both now return NaN, from one definition (docs/69).
    switch (op) {
        case UnaryOp::Neg:    return -a;
        case UnaryOp::Exp:    return exp(a);    // ADL: rsymbolic:: for Dual, std:: for double
        case UnaryOp::Log:    return log(a);
        case UnaryOp::Sin:    return sin(a);
        case UnaryOp::Cos:    return cos(a);
        case UnaryOp::Sqrt:   return sqrt(a);   // rsymbolic::safe_sqrt (neg→NaN), all T
        case UnaryOp::Tanh:   return tanh(a);
        case UnaryOp::Abs:    return abs(a);
        case UnaryOp::Square: return square(a); // rsymbolic::square via ADL
        case UnaryOp::Inv:    return recip(a);  // rsymbolic::recip via ADL (1/x)
        case UnaryOp::Erf:    return erf(a);    // ADL: rsymbolic:: for Dual, std:: for double
        case UnaryOp::Sinh:   return sinh(a);
        case UnaryOp::Cosh:   return cosh(a);
    }
    return a;  // unreachable
}

template <typename T>
T apply_binary(BinaryOp op, const T& a, const T& b) {
    switch (op) {
        case BinaryOp::Add: return a + b;
        case BinaryOp::Sub: return a - b;
        case BinaryOp::Mul: return a * b;
        case BinaryOp::Div: return a / b;
        case BinaryOp::Pow: return pow(a, b);  // rsymbolic::pow via ADL (safe semantics)
    }
    return a;  // unreachable
}

inline const char* unary_name(UnaryOp op) {
    switch (op) {
        case UnaryOp::Neg:    return "neg";
        case UnaryOp::Exp:    return "exp";
        case UnaryOp::Log:    return "log";
        case UnaryOp::Sin:    return "sin";
        case UnaryOp::Cos:    return "cos";
        case UnaryOp::Sqrt:   return "sqrt";
        case UnaryOp::Tanh:   return "tanh";
        case UnaryOp::Abs:    return "abs";
        case UnaryOp::Square: return "square";
        case UnaryOp::Inv:    return "inv";
        case UnaryOp::Erf:    return "erf";
        case UnaryOp::Sinh:   return "sinh";
        case UnaryOp::Cosh:   return "cosh";
    }
    return "?";
}

inline const char* binary_name(BinaryOp op) {
    switch (op) {
        case BinaryOp::Add: return "+";
        case BinaryOp::Sub: return "-";
        case BinaryOp::Mul: return "*";
        case BinaryOp::Div: return "/";
        case BinaryOp::Pow: return "^";
    }
    return "?";
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
//
// Scratch overload: the evaluation stack is supplied by the caller and reused across
// calls. The stack vector itself carries no information between calls (it is cleared
// on entry); only its heap capacity is reused. This removes the per-call allocation
// that dominated the multi-island hot path (one std::vector<T> allocated per data
// point, m allocations per residual/Jacobian evaluation) — see docs/23 §4. Numerically
// identical to a fresh stack: same operations in the same order.
//
// Thread-safety: the caller owns `stack`; pass a stack local to each thread / closure
// call and never share one across threads. evaluate() is non-recursive, so a single
// stack per in-flight call is sufficient.
template <typename T>
T evaluate(const Tree& tree, const double* row, const T* constants,
           std::vector<T>& stack) {
    stack.clear();
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

// Convenience overload: allocates a fresh scratch stack per call. Kept for one-off
// callers (tests, non-hot paths) that do not manage their own buffer.
template <typename T>
T evaluate(const Tree& tree, const double* row, const T* constants) {
    std::vector<T> stack;
    return evaluate(tree, row, constants, stack);
}

// Number of operands a node consumes.
inline int arity(const Node& node) {
    switch (node.kind) {
        case NodeKind::Constant:
        case NodeKind::Variable:
            return 0;
        case NodeKind::Unary:
            return 1;
        case NodeKind::Binary:
            return 2;
    }
    return 0;
}

// True if the node sequence is a well-formed postfix expression: a stack evaluation
// never underflows and finishes with exactly one value.
inline bool is_valid_postfix(const Tree& tree) {
    long depth = 0;
    for (const Node& node : tree) {
        depth -= arity(node);
        if (depth < 0) return false;
        depth += 1;
    }
    return depth == 1;
}

// Start index of the subtree whose root (last node) is at position `end`. The subtree
// occupies the contiguous range [start, end] in postfix order.
inline std::size_t subtree_start(const Tree& tree, std::size_t end) {
    int slots = 1;
    std::size_t j = end;
    while (true) {
        slots += arity(tree[j]) - 1;  // node fills one slot, needs `arity` more
        if (slots == 0 || j == 0) break;
        --j;
    }
    return j;
}

// Reassign constant parameter indices to a contiguous 0..k-1 range (left to right).
// Call this after any structural edit that may have disturbed the indices.
inline void reindex_constants(Tree& tree) {
    int index = 0;
    for (Node& node : tree) {
        if (node.kind == NodeKind::Constant) node.index = index++;
    }
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

// Write the constant values back into the tree's Constant nodes (e.g. after fitting).
inline void set_constants(Tree& tree, const std::vector<double>& constants) {
    for (Node& node : tree) {
        if (node.kind == NodeKind::Constant) {
            node.value = constants[static_cast<std::size_t>(node.index)];
        }
    }
}

// Render a tree as a human-readable infix string. This is the frozen round-trip form:
// predict() parses it back in R, Python and JavaScript, and every display surface shows
// it, so a change here is a change to all of them at once.
//
// Three unaries render as operators rather than as calls, because notation exists for
// them and `square`/`inv`/`neg` are the engine's internal names, not anything a reader
// writes:
//
//     square(a) -> (a ^ 2)        inv(a) -> (1 / a)        neg(a) -> (-a)
//
// Each reuses syntax this grammar already has — `^`, `/` and unary minus are all in the
// parser and in all three predict() wrappers — so nothing new has to be parsed, and the
// result agrees with to_latex(), which has always printed `a^{2}` and `-a`.
//
// The rewrites are exact, not approximate:
//   * safe_pow's only guard for an integer exponent is `y < 0 && x == 0` (dual.hpp), so
//     pow(x, 2) is x*x for every double including the infinities and NaN — what square()
//     computes, and what the wrappers' `^` computes (they part from safe_pow only at
//     0^-1 and (-Inf)^0.5, neither reachable with exponent 2).
//   * recip(a) is literally `1.0 / a` and Div is `a / b`, both unguarded, so `(1 / a)`
//     and inv(a) are the same IEEE operation.
//   * negation is exact by definition.
//
// Every form is fully parenthesized, like the binaries, so the printer needs no
// precedence tracking and no context is ever misread. The parenthesisation of neg is
// load bearing rather than cosmetic: `-a ^ 2` would parse as -(a^2) in Python and R,
// which is not square(neg(a)).
//
// A Square node and a Pow node over the constant 2 therefore print alike, as do Inv and
// a Div under a 1. That is a property of the string, not a loss: each pair computes the
// same function, and the string has never carried node identity (complexity comes from
// the tree). Parsers still accept `square(...)`, `inv(...)` and `neg(...)` —
// op_names.hpp keeps the names, and the predict wrappers keep the bindings — so strings
// saved by an earlier version continue to evaluate.
inline std::string to_string(const Tree& tree) {
    std::vector<std::string> stack;
    stack.reserve(tree.size());
    char buf[32];
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant:
                std::snprintf(buf, sizeof(buf), "%.6g", node.value);
                stack.emplace_back(buf);
                break;
            case NodeKind::Variable:
                std::snprintf(buf, sizeof(buf), "x%d", node.index);
                stack.emplace_back(buf);
                break;
            case NodeKind::Unary: {
                const std::string a = stack.back();
                switch (node.uop) {
                    case UnaryOp::Square: stack.back() = "(" + a + " ^ 2)"; break;
                    case UnaryOp::Inv:    stack.back() = "(1 / " + a + ")"; break;
                    case UnaryOp::Neg:    stack.back() = "(-" + a + ")"; break;
                    default:
                        stack.back() =
                            std::string(detail::unary_name(node.uop)) + "(" + a + ")";
                        break;
                }
                break;
            }
            case NodeKind::Binary: {
                const std::string b = stack.back();
                stack.pop_back();
                const std::string a = stack.back();
                stack.back() =
                    "(" + a + " " + detail::binary_name(node.bop) + " " + b + ")";
                break;
            }
        }
    }
    return stack.empty() ? std::string() : stack.back();
}

}  // namespace rsymbolic
