// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/simplification/display_simplify.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace rsymbolic {

namespace {

// Temporary recursive form used only during simplification. Deliberately a local
// duplicate of the identical struct in simplify.cpp (see that file's header comment
// and docs/52): this file must have no dependency on the search-loop simplifier.
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

Ast make_unary(UnaryOp op, Ast child) {
    Ast a;
    a.node = unary_node(op);
    a.kids.push_back(std::move(child));
    return a;
}

Ast make_binary(BinaryOp op, Ast left, Ast right) {
    Ast a;
    a.node = binary_node(op);
    a.kids.push_back(std::move(left));
    a.kids.push_back(std::move(right));
    return a;
}

// Build a recursive AST from a postfix tree. Identical to simplify.cpp's build().
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

// --- Rule 2: Add/Sub/Mul constant reassociation --------------------------------------
// A deliberate duplicate of simplify.cpp's combine() single-node logic (the recursion
// into children happens in the caller, rewrite_pass(), not here). See simplify.cpp for
// the derivation; kept byte-for-byte equivalent in behaviour so display_simplify never
// diverges from simplify() on the cases they share.
Ast combine_binary(Ast a) {
    const BinaryOp op = a.node.bop;
    Ast L = std::move(a.kids[0]);
    Ast R = std::move(a.kids[1]);
    if (!is_constant(L) && !is_constant(R)) return make_binary(op, std::move(L), std::move(R));

    if (op == BinaryOp::Add || op == BinaryOp::Mul) {  // commutative
        if (is_constant(L)) std::swap(L, R);
        const double topconstant = R.node.value;
        if (is_binary_op(L, op)) {
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
        return make_binary(op, std::move(L), std::move(R));
    }

    if (op == BinaryOp::Sub) {  // not commutative: the four SR.jl rewrites
        if (is_constant(L) && is_binary_op(R, BinaryOp::Sub)) {
            if (is_constant(R.kids[0])) {
                const double cprime = R.kids[0].node.value - L.node.value;
                return make_binary(BinaryOp::Sub, std::move(R.kids[1]), make_constant(cprime));
            }
            if (is_constant(R.kids[1])) {
                const double cprime = L.node.value + R.kids[1].node.value;
                return make_binary(BinaryOp::Sub, make_constant(cprime), std::move(R.kids[0]));
            }
        } else if (is_constant(R) && is_binary_op(L, BinaryOp::Sub)) {
            if (is_constant(L.kids[0])) {
                const double cprime = L.kids[0].node.value - R.node.value;
                return make_binary(BinaryOp::Sub, make_constant(cprime), std::move(L.kids[1]));
            }
            if (is_constant(L.kids[1])) {
                const double cprime = R.node.value + L.kids[1].node.value;
                return make_binary(BinaryOp::Sub, std::move(L.kids[0]), make_constant(cprime));
            }
        }
        return make_binary(BinaryOp::Sub, std::move(L), std::move(R));
    }

    return make_binary(op, std::move(L), std::move(R));  // Div / Pow: handled elsewhere
}

// --- Rule 3: NEW Div/Mul constant-chain folding (not in SR.jl combine_operators) ------
// Runs on the node AFTER combine_binary(), so for Mul the constant operand (if any) is
// already canonicalised to the right, and any Div/Mul child has already been through
// this same fixed-point pass (post-order: children first). Each branch strictly
// removes 2 nodes (5 nodes -> 3) and is isfinite-guarded.
Ast fold_div_mul_chain(Ast a) {
    if (a.node.kind != NodeKind::Binary) return a;
    const BinaryOp op = a.node.bop;
    if (op != BinaryOp::Div && op != BinaryOp::Mul) return a;

    Ast& L = a.kids[0];
    Ast& R = a.kids[1];

    if (op == BinaryOp::Div) {
        // (t * c1) / c2 -> t * (c1/c2)
        if (is_binary_op(L, BinaryOp::Mul) && is_constant(L.kids[1]) && is_constant(R)) {
            const double v = L.kids[1].node.value / R.node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Mul, std::move(L.kids[0]), make_constant(v));
        }
        // (t / c1) / c2 -> t / (c1*c2)
        if (is_binary_op(L, BinaryOp::Div) && is_constant(L.kids[1]) && is_constant(R)) {
            const double v = L.kids[1].node.value * R.node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Div, std::move(L.kids[0]), make_constant(v));
        }
        // (c1 / t) / c2 -> (c1/c2) / t
        if (is_binary_op(L, BinaryOp::Div) && is_constant(L.kids[0]) && is_constant(R)) {
            const double v = L.kids[0].node.value / R.node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Div, make_constant(v), std::move(L.kids[1]));
        }
        // c1 / (c2 / t) -> (c1/c2) * t
        if (is_constant(L) && is_binary_op(R, BinaryOp::Div) && is_constant(R.kids[0])) {
            const double v = L.node.value / R.kids[0].node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Mul, std::move(R.kids[1]), make_constant(v));
        }
        // c1 / (t * c2) -> (c1/c2) / t
        if (is_constant(L) && is_binary_op(R, BinaryOp::Mul) && is_constant(R.kids[1])) {
            const double v = L.node.value / R.kids[1].node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Div, make_constant(v), std::move(R.kids[0]));
        }
        // c1 / (t / c2) -> (c1*c2) / t
        if (is_constant(L) && is_binary_op(R, BinaryOp::Div) && is_constant(R.kids[1])) {
            const double v = L.node.value * R.kids[1].node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Div, make_constant(v), std::move(R.kids[0]));
        }
    } else {  // Mul
        // (t / c1) * c2 -> t * (c2/c1)
        if (is_binary_op(L, BinaryOp::Div) && is_constant(L.kids[1]) && is_constant(R)) {
            const double v = R.node.value / L.kids[1].node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Mul, std::move(L.kids[0]), make_constant(v));
        }
        // (c1 / t) * c2 -> (c1*c2) / t
        if (is_binary_op(L, BinaryOp::Div) && is_constant(L.kids[0]) && is_constant(R)) {
            const double v = L.kids[0].node.value * R.node.value;
            if (std::isfinite(v)) return make_binary(BinaryOp::Div, make_constant(v), std::move(L.kids[1]));
        }
    }
    return a;
}

// --- Rule 4: identity elimination -----------------------------------------------------
// Runs after rules 2/3 have canonicalised the node, so a literal identity constant (if
// present at all) is always the right-hand child: t+0/0+t and t*1/1*t are both
// canonicalised to (t op c) by combine_binary before this runs (see the header
// comment); t-0 and t/1 never swap operands (Sub/Div are not commutative) so they are
// already in this shape. Deliberately excludes t*0, 0*t (would discard t's NaN/Inf)
// and t^1, t^0 (see display_simplify.hpp for the t^1 exactness finding).
Ast eliminate_identity(Ast a) {
    if (a.node.kind != NodeKind::Binary) return a;
    if (!is_constant(a.kids[1])) return a;
    const BinaryOp op = a.node.bop;
    const double c = a.kids[1].node.value;
    if ((op == BinaryOp::Add || op == BinaryOp::Sub) && c == 0.0) return std::move(a.kids[0]);
    if ((op == BinaryOp::Mul || op == BinaryOp::Div) && c == 1.0) return std::move(a.kids[0]);
    return a;
}

// --- Rule 6: t * (-1) -> neg(t) --------------------------------------------------------
// Runs after rule 4 so a literal -1 (never matching the Mul==1 identity check) is
// still present as the right child when reached.
Ast rewrite_mul_neg(Ast a) {
    if (a.node.kind == NodeKind::Binary && a.node.bop == BinaryOp::Mul &&
        is_constant(a.kids[1]) && a.kids[1].node.value == -1.0) {
        return make_unary(UnaryOp::Neg, std::move(a.kids[0]));
    }
    return a;
}

// One bottom-up fixed-point pass applying rules 1-6, in order, at each node (each rule
// fires at most once per node per pass; a rewrite that creates a new node shape for a
// LATER rule to act on is picked up on the next pass by the caller's fixed-point loop).
Ast rewrite_pass(Ast a) {
    for (Ast& kid : a.kids) {
        kid = rewrite_pass(std::move(kid));
    }

    // Constant / Variable leaves have no children to fold or rewrite; nothing below
    // this point may run on them (combine_binary et al. assume exactly two children).
    if (a.node.kind == NodeKind::Constant || a.node.kind == NodeKind::Variable) return a;

    if (a.node.kind == NodeKind::Unary) {
        // Rule 1: constant-subtree fold, isfinite-guarded (subsumes neg(c) -> literal).
        if (is_constant(a.kids[0])) {
            const double v = detail::apply_unary<double>(a.node.uop, a.kids[0].node.value);
            if (std::isfinite(v)) return make_constant(v);
        }
        // Rule 5: double negation.
        if (a.node.uop == UnaryOp::Neg && a.kids[0].node.kind == NodeKind::Unary &&
            a.kids[0].node.uop == UnaryOp::Neg) {
            return std::move(a.kids[0].kids[0]);
        }
        return a;
    }

    // Binary node. Rule 1: constant-subtree fold, isfinite-guarded.
    if (is_constant(a.kids[0]) && is_constant(a.kids[1])) {
        const double v = detail::apply_binary<double>(a.node.bop, a.kids[0].node.value,
                                                       a.kids[1].node.value);
        if (std::isfinite(v)) return make_constant(v);
    }

    // Rule 2, then rule 3, then rule 4, then rule 6, in order.
    Ast node = combine_binary(std::move(a));
    node = fold_div_mul_chain(std::move(node));
    node = eliminate_identity(std::move(node));
    node = rewrite_mul_neg(std::move(node));
    return node;
}

// Structural equality used by the fixed-point loop's convergence check: exact (no
// tolerance), since a converged tree's node values are bit-identical to the previous
// pass's output when no rule fired.
bool same_tree(const Tree& a, const Tree& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const Node& x = a[i];
        const Node& y = b[i];
        if (x.kind != y.kind) return false;
        switch (x.kind) {
            case NodeKind::Constant:
                if (x.value != y.value) return false;
                break;
            case NodeKind::Variable:
                if (x.index != y.index) return false;
                break;
            case NodeKind::Unary:
                if (x.uop != y.uop) return false;
                break;
            case NodeKind::Binary:
                if (x.bop != y.bop) return false;
                break;
        }
    }
    return true;
}

// Emit an AST back to a postfix tree, re-indexing constants contiguously. Identical to
// simplify.cpp's emit().
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

Tree to_tree(const Ast& a) {
    Tree out;
    int const_index = 0;
    emit(a, out, const_index);
    return out;
}

// Every mutating rule above strictly decreases node count except rule 2's
// canonicalise-only swap (no size change) — so repeated passes form a terminating,
// strictly-non-increasing sequence in the worst case; a handful of passes is enough
// for any realistically bloated candidate to reach its fixed point. 8 is a generous
// margin (tests assert it is never hit).
constexpr int kMaxPasses = 8;

}  // namespace

Tree display_simplify(const Tree& tree, int* passes_used) {
    if (passes_used) *passes_used = 0;
    if (tree.empty()) return {};

    Ast ast = build(tree);
    Tree prev = to_tree(ast);
    for (int pass = 0; pass < kMaxPasses; ++pass) {
        ast = rewrite_pass(std::move(ast));
        Tree cur = to_tree(ast);
        if (passes_used) *passes_used = pass + 1;
        if (same_tree(cur, prev)) return cur;
        prev = std::move(cur);
    }
    return prev;
}

}  // namespace rsymbolic
