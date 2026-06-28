// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/simplification/simplify.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace rsymbolic {

namespace {

// Temporary recursive form used only during simplification.
struct Ast {
    Node node;
    std::vector<Ast> kids;
};

bool is_constant(const Ast& a) {
    return a.node.kind == NodeKind::Constant;
}

Ast make_constant(double value) {
    Ast a;
    a.node = constant_node(0, value);  // index assigned on emit
    return a;
}

Ast make_binary(BinaryOp op, Ast left, Ast right) {
    Ast a;
    a.node = binary_node(op);
    a.kids.push_back(std::move(left));
    a.kids.push_back(std::move(right));
    return a;
}

// Build a recursive AST from a postfix tree.
Ast build(const Tree& tree) {
    std::vector<Ast> stack;
    stack.reserve(tree.size());
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant:
            case NodeKind::Variable:
                stack.push_back(Ast{node, {}});
                break;
            case NodeKind::Unary: {
                Ast child = std::move(stack.back());
                stack.pop_back();
                stack.push_back(Ast{node, {std::move(child)}});
                break;
            }
            case NodeKind::Binary: {
                Ast right = std::move(stack.back());
                stack.pop_back();
                Ast left = std::move(stack.back());
                stack.pop_back();
                stack.push_back(Ast{node, {std::move(left), std::move(right)}});
                break;
            }
        }
    }
    return std::move(stack.back());
}

bool is_binary_op(const Ast& a, BinaryOp op) {
    return a.node.kind == NodeKind::Binary && a.node.bop == op;
}

// Stage 1: constant folding (DynamicExpressions `simplify_tree!`). A node all of whose
// children are constants collapses to one constant, provided the result is finite (the
// `is_valid` guard in SR.jl). No identity/negation/power rewrites: SR.jl does not perform
// them here — that omission is intentional parity (docs/29 §11).
Ast fold(Ast a) {
    for (Ast& kid : a.kids) {
        kid = fold(std::move(kid));
    }

    if (a.node.kind == NodeKind::Unary && is_constant(a.kids[0])) {
        const double v = detail::apply_unary<double>(a.node.uop, a.kids[0].node.value);
        if (std::isfinite(v)) return make_constant(v);
    } else if (a.node.kind == NodeKind::Binary && is_constant(a.kids[0]) &&
               is_constant(a.kids[1])) {
        const double v = detail::apply_binary<double>(a.node.bop, a.kids[0].node.value,
                                                      a.kids[1].node.value);
        if (std::isfinite(v)) return make_constant(v);
    }
    return a;
}

// Stage 2: combine_operators (DynamicExpressions `combine_operators`). Children are
// recursed first so a child's canonicalisation is visible when combining the parent.
// Acts only on a binary node with one constant child (`top_level_constant`):
//   - commutative (+,*): canonicalise the constant to the right, then merge it into a
//     same-operator child that also has a constant operand:
//       (c1 ∘ x) ∘ c2  ->  (op(c1,c2) ∘ x)        [and the mirror with x on the left]
//   - subtraction (-): the four SR.jl rewrites that fold a nested constant.
// Div/Pow are neither, so only their children are recursed. The merged constant is NOT
// finiteness-guarded — SR.jl's combine_operators applies the operator unconditionally
// (only the fold stage guards), and parity outranks the rare non-finite merge.
Ast combine(Ast a) {
    for (Ast& kid : a.kids) {
        kid = combine(std::move(kid));
    }
    if (a.node.kind != NodeKind::Binary) return a;

    const BinaryOp op = a.node.bop;
    Ast L = std::move(a.kids[0]);
    Ast R = std::move(a.kids[1]);
    if (!is_constant(L) && !is_constant(R)) return make_binary(op, std::move(L), std::move(R));

    if (op == BinaryOp::Add || op == BinaryOp::Mul) {  // commutative
        // Canonicalise: put the constant on the right, then assume the var on the left.
        if (is_constant(L)) std::swap(L, R);
        const double topconstant = R.node.value;
        if (is_binary_op(L, op)) {  // (below.l ∘ below.r) ∘ topconstant
            if (is_constant(L.kids[0])) {
                L.kids[0].node.value =
                    detail::apply_binary<double>(op, L.kids[0].node.value, topconstant);
                return L;
            }
            if (is_constant(L.kids[1])) {
                L.kids[1].node.value =
                    detail::apply_binary<double>(op, L.kids[1].node.value, topconstant);
                return L;
            }
        }
        return make_binary(op, std::move(L), std::move(R));  // canonicalised, no merge
    }

    if (op == BinaryOp::Sub) {  // not commutative: the four SR.jl rewrites
        if (is_constant(L) && is_binary_op(R, BinaryOp::Sub)) {
            if (is_constant(R.kids[0])) {
                // (c - (c2 - var)) -> (var - c'),  c' = c2 - c
                const double cprime = R.kids[0].node.value - L.node.value;
                return make_binary(BinaryOp::Sub, std::move(R.kids[1]), make_constant(cprime));
            }
            if (is_constant(R.kids[1])) {
                // (c - (var - c2)) -> (c' - var),  c' = c + c2
                const double cprime = L.node.value + R.kids[1].node.value;
                return make_binary(BinaryOp::Sub, make_constant(cprime), std::move(R.kids[0]));
            }
        } else if (is_constant(R) && is_binary_op(L, BinaryOp::Sub)) {
            if (is_constant(L.kids[0])) {
                // ((c - var) - c2) -> (c' - var),  c' = c - c2
                const double cprime = L.kids[0].node.value - R.node.value;
                return make_binary(BinaryOp::Sub, make_constant(cprime), std::move(L.kids[1]));
            }
            if (is_constant(L.kids[1])) {
                // ((var - c) - c2) -> (var - c'),  c' = c2 + c
                const double cprime = R.node.value + L.kids[1].node.value;
                return make_binary(BinaryOp::Sub, std::move(L.kids[0]), make_constant(cprime));
            }
        }
        return make_binary(BinaryOp::Sub, std::move(L), std::move(R));
    }

    return make_binary(op, std::move(L), std::move(R));  // Div / Pow: no combining
}

// Emit an AST back to a postfix tree, re-indexing constants contiguously.
void emit(const Ast& a, Tree& out, int& const_index) {
    for (const Ast& kid : a.kids) {
        emit(kid, out, const_index);
    }
    Node node = a.node;
    if (node.kind == NodeKind::Constant) {
        node.index = const_index++;
    }
    out.push_back(node);
}

}  // namespace

Tree simplify(const Tree& tree) {
    if (tree.empty()) return {};
    // SR.jl applies simplify_tree! (fold) then combine_operators, in that order: folding
    // first lets combine_operators see fully-reduced constant operands (DynamicExpressions
    // Simplify.jl). The mutation `kSimplify` and the per-iteration population pass both go
    // through here, mirroring SR.jl where both call the same pair.
    Ast ast = combine(fold(build(tree)));
    Tree out;
    out.reserve(tree.size());
    int const_index = 0;
    emit(ast, out, const_index);
    return out;
}

}  // namespace rsymbolic
