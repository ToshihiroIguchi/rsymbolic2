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

Ast simplify_ast(Ast a) {
    for (Ast& kid : a.kids) {
        kid = simplify_ast(std::move(kid));
    }

    if (a.node.kind == NodeKind::Unary) {
        const Ast& child = a.kids[0];
        if (is_constant(child)) {
            const double v = detail::apply_unary<double>(a.node.uop, child.node.value);
            if (std::isfinite(v)) return make_constant(v);
        }
        // neg(neg(x)) -> x
        if (a.node.uop == UnaryOp::Neg && child.node.kind == NodeKind::Unary &&
            child.node.uop == UnaryOp::Neg) {
            return a.kids[0].kids[0];
        }
        return a;
    }

    if (a.node.kind == NodeKind::Binary) {
        const Ast& L = a.kids[0];
        const Ast& R = a.kids[1];
        const bool lc = is_constant(L);
        const bool rc = is_constant(R);

        // Constant folding.
        if (lc && rc) {
            const double v = detail::apply_binary<double>(a.node.bop, L.node.value,
                                                          R.node.value);
            if (std::isfinite(v)) return make_constant(v);
        }

        switch (a.node.bop) {
            case BinaryOp::Add:
                if (lc && L.node.value == 0.0) return a.kids[1];  // 0 + x -> x
                if (rc && R.node.value == 0.0) return a.kids[0];  // x + 0 -> x
                break;
            case BinaryOp::Sub:
                if (rc && R.node.value == 0.0) return a.kids[0];  // x - 0 -> x
                break;
            case BinaryOp::Mul:
                if ((lc && L.node.value == 0.0) || (rc && R.node.value == 0.0)) {
                    return make_constant(0.0);  // x * 0 -> 0
                }
                if (lc && L.node.value == 1.0) return a.kids[1];  // 1 * x -> x
                if (rc && R.node.value == 1.0) return a.kids[0];  // x * 1 -> x
                break;
            case BinaryOp::Div:
                if (rc && R.node.value == 1.0) return a.kids[0];  // x / 1 -> x
                break;
            case BinaryOp::Pow:
                if (rc && R.node.value == 0.0) return make_constant(1.0);  // x^0 -> 1
                if (rc && R.node.value == 1.0) return a.kids[0];           // x^1 -> x
                if (rc && R.node.value == 2.0) {                           // x^2 -> square(x)
                    Ast sq;
                    sq.node = unary_node(UnaryOp::Square);
                    sq.kids.push_back(a.kids[0]);
                    return sq;
                }
                break;
        }
        return a;
    }

    return a;  // Constant or Variable leaf
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
    Ast ast = simplify_ast(build(tree));
    Tree out;
    out.reserve(tree.size());
    int const_index = 0;
    emit(ast, out, const_index);
    return out;
}

}  // namespace rsymbolic
