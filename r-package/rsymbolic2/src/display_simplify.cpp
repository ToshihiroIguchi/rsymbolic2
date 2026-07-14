// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

// Layer 1 of the display-only simplifier (docs/54): deterministic normalisation in
// the style of Cohen 2003's automatic simplification, plus the Layer-2 orchestration
// (egraph.cpp) and its fallback. Deliberately shares no code with simplify.cpp (the
// search-loop parity simplifier): the two must never influence each other.

#include "rsymbolic/simplification/display_simplify.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace rsymbolic {

namespace {

// Temporary recursive form used only during simplification.
struct Ast {
    Node node;
    std::vector<Ast> kids;
};

bool is_const(const Ast& a) { return a.node.kind == NodeKind::Constant; }
bool is_unary(const Ast& a, UnaryOp op) {
    return a.node.kind == NodeKind::Unary && a.node.uop == op;
}
bool is_binary(const Ast& a, BinaryOp op) {
    return a.node.kind == NodeKind::Binary && a.node.bop == op;
}

Ast mk_const(double value) {
    Ast a;
    a.node = constant_node(0, value);  // index assigned on emit
    return a;
}

Ast mk_unary(UnaryOp op, Ast kid) {
    Ast a;
    a.node = unary_node(op);
    a.kids.push_back(std::move(kid));
    return a;
}

Ast mk_binary(BinaryOp op, Ast left, Ast right) {
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

Tree to_tree(const Ast& a) {
    Tree out;
    int const_index = 0;
    emit(a, out, const_index);
    return out;
}

std::uint64_t double_bits(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

// Deterministic total order used to canonicalise commutative operand lists and to
// group like terms/factors. Variables < unaries < binaries < constants — constants
// sort last, matching the constant-on-the-right display convention simplify()'s
// combine pass established.
int cmp(const Ast& x, const Ast& y) {
    const auto rank = [](const Ast& a) {
        switch (a.node.kind) {
            case NodeKind::Variable: return 0;
            case NodeKind::Unary: return 1;
            case NodeKind::Binary: return 2;
            case NodeKind::Constant: return 3;
        }
        return 4;
    };
    const int rx = rank(x), ry = rank(y);
    if (rx != ry) return rx < ry ? -1 : 1;
    switch (x.node.kind) {
        case NodeKind::Variable:
            if (x.node.index != y.node.index) return x.node.index < y.node.index ? -1 : 1;
            return 0;
        case NodeKind::Constant: {
            const std::uint64_t bx = double_bits(x.node.value);
            const std::uint64_t by = double_bits(y.node.value);
            if (x.node.value < y.node.value) return -1;
            if (y.node.value < x.node.value) return 1;
            if (bx != by) return bx < by ? -1 : 1;  // -0.0 vs +0.0, NaN payloads
            return 0;
        }
        case NodeKind::Unary:
            if (x.node.uop != y.node.uop) return x.node.uop < y.node.uop ? -1 : 1;
            return cmp(x.kids[0], y.kids[0]);
        case NodeKind::Binary: {
            if (x.node.bop != y.node.bop) return x.node.bop < y.node.bop ? -1 : 1;
            const int c0 = cmp(x.kids[0], y.kids[0]);
            if (c0 != 0) return c0;
            return cmp(x.kids[1], y.kids[1]);
        }
    }
    return 0;
}

bool ast_equal(const Ast& x, const Ast& y) { return cmp(x, y) == 0; }

Ast normalize(Ast a);  // forward

// Exact negation of an already-normalised subtree. Every branch is IEEE-exact:
// negation never rounds, fl(b - a) == -fl(a - b), and negating a constant operand of
// a canonical product/reciprocal only flips that constant's sign bit.
Ast negate(Ast t) {
    if (is_const(t)) return mk_const(-t.node.value);
    if (is_unary(t, UnaryOp::Neg)) return std::move(t.kids[0]);
    if (is_binary(t, BinaryOp::Sub)) {
        return mk_binary(BinaryOp::Sub, std::move(t.kids[1]), std::move(t.kids[0]));
    }
    if (is_binary(t, BinaryOp::Mul) && is_const(t.kids[1])) {
        t.kids[1].node.value = -t.kids[1].node.value;
        return t;
    }
    if (is_binary(t, BinaryOp::Div) && is_const(t.kids[0])) {
        t.kids[0].node.value = -t.kids[0].node.value;
        return t;
    }
    return mk_unary(UnaryOp::Neg, std::move(t));
}

// Unary application on a normalised child, with the exact/audited local rewrites
// (docs/54): constant folding (isfinite-guarded), negation normal form, abs/square of
// sign-invariant arguments, odd/even functions of neg, sqrt(square t) -> abs t (the
// one drift+overflow-caveat rule in this file). Anything not listed passes through.
Ast norm_unary(UnaryOp op, Ast kid) {
    if (is_const(kid)) {
        const double v = detail::apply_unary<double>(op, kid.node.value);
        if (std::isfinite(v)) return mk_const(v);
        return mk_unary(op, std::move(kid));
    }
    switch (op) {
        case UnaryOp::Neg:
            return negate(std::move(kid));
        case UnaryOp::Abs:
            if (is_unary(kid, UnaryOp::Neg)) {
                return norm_unary(UnaryOp::Abs, std::move(kid.kids[0]));
            }
            // |t| == t when t is non-negative or NaN by construction.
            if (is_unary(kid, UnaryOp::Abs) || is_unary(kid, UnaryOp::Square) ||
                is_unary(kid, UnaryOp::Sqrt) || is_unary(kid, UnaryOp::Exp)) {
                return kid;
            }
            break;
        case UnaryOp::Square:
            if (is_unary(kid, UnaryOp::Neg) || is_unary(kid, UnaryOp::Abs)) {
                return norm_unary(UnaryOp::Square, std::move(kid.kids[0]));
            }
            break;
        case UnaryOp::Sqrt:
            if (is_unary(kid, UnaryOp::Square)) {
                return norm_unary(UnaryOp::Abs, std::move(kid.kids[0]));
            }
            break;
        case UnaryOp::Sin:
        case UnaryOp::Tanh:
            if (is_unary(kid, UnaryOp::Neg)) {  // odd: pull the negation out
                return negate(mk_unary(op, std::move(kid.kids[0])));
            }
            break;
        case UnaryOp::Cos:
            if (is_unary(kid, UnaryOp::Neg)) {  // even: drop the negation
                return mk_unary(UnaryOp::Cos, std::move(kid.kids[0]));
            }
            break;
        default:
            break;
    }
    return mk_unary(op, std::move(kid));
}

// Fold a list of constants left-to-right under `op`, guarded: returns false (leave
// the constants unmerged) if any partial result is non-finite.
bool fold_constants(const std::vector<double>& vals, BinaryOp op, double init, double& out) {
    double acc = init;
    for (double v : vals) {
        acc = detail::apply_binary<double>(op, acc, v);
        if (!std::isfinite(acc)) return false;
    }
    out = acc;
    return true;
}

// ---- products ------------------------------------------------------------------

struct ProductParts {
    std::vector<Ast> num, den;      // non-constant factors
    std::vector<double> ncon, dcon; // constant factors, sign of pulled-out negs in ncon
};

// Flatten an already-normalised subtree into numerator/denominator factor lists.
void collect_norm_product(Ast t, bool inv, ProductParts& p) {
    if (is_binary(t, BinaryOp::Mul)) {
        collect_norm_product(std::move(t.kids[0]), inv, p);
        collect_norm_product(std::move(t.kids[1]), inv, p);
    } else if (is_binary(t, BinaryOp::Div)) {
        collect_norm_product(std::move(t.kids[0]), inv, p);
        collect_norm_product(std::move(t.kids[1]), !inv, p);
    } else if (is_unary(t, UnaryOp::Neg)) {
        p.ncon.push_back(-1.0);  // exact sign pull-out
        collect_norm_product(std::move(t.kids[0]), inv, p);
    } else if (is_const(t)) {
        (inv ? p.dcon : p.ncon).push_back(t.node.value);
    } else {
        (inv ? p.den : p.num).push_back(std::move(t));
    }
}

// Flatten a raw Mul/Div/Neg spine, normalising at the leaves (each leaf may itself
// normalise into a canonical product form, which is then flattened further).
void collect_raw_product(Ast t, bool inv, ProductParts& p) {
    if (is_binary(t, BinaryOp::Mul)) {
        collect_raw_product(std::move(t.kids[0]), inv, p);
        collect_raw_product(std::move(t.kids[1]), inv, p);
    } else if (is_binary(t, BinaryOp::Div)) {
        collect_raw_product(std::move(t.kids[0]), inv, p);
        collect_raw_product(std::move(t.kids[1]), !inv, p);
    } else if (is_unary(t, UnaryOp::Neg)) {
        p.ncon.push_back(-1.0);
        collect_raw_product(std::move(t.kids[0]), inv, p);
    } else {
        collect_norm_product(normalize(std::move(t)), inv, p);
    }
}

// Group k structurally-equal factors: pairs become square(f) (bit-exact — square
// evaluates as f*f), recursively, so k=4 yields square(square(f)). The rebuilt
// product's association may differ from the input's (rounding drift, docs/54).
void expand_multiplicity(Ast f, std::size_t k, std::vector<Ast>& out) {
    while (k >= 2) {
        if (k % 2 != 0) out.push_back(f);  // the odd factor stays un-squared
        f = norm_unary(UnaryOp::Square, std::move(f));
        k /= 2;
    }
    out.push_back(std::move(f));
}

// Sort factors canonically and collapse structurally-equal runs via square().
std::vector<Ast> group_factors(std::vector<Ast> fs) {
    std::sort(fs.begin(), fs.end(), [](const Ast& x, const Ast& y) { return cmp(x, y) < 0; });
    std::vector<Ast> out;
    std::size_t i = 0;
    while (i < fs.size()) {
        std::size_t j = i + 1;
        while (j < fs.size() && ast_equal(fs[i], fs[j])) ++j;
        expand_multiplicity(std::move(fs[i]), j - i, out);
        i = j;
    }
    // expand_multiplicity may emit square(f) after a lone f; keep canonical order.
    std::sort(out.begin(), out.end(), [](const Ast& x, const Ast& y) { return cmp(x, y) < 0; });
    return out;
}

Ast left_fold_mul(std::vector<Ast> fs) {
    Ast acc = std::move(fs[0]);
    for (std::size_t i = 1; i < fs.size(); ++i) {
        acc = mk_binary(BinaryOp::Mul, std::move(acc), std::move(fs[i]));
    }
    return acc;
}

// Rebuild a flattened product in canonical display form: (N / D) with a single
// merged constant on the right — N * q, q / D, or (N / D) * q — falling back to
// unmerged constant factors when the merge would be non-finite (docs/54).
Ast rebuild_product(ProductParts p) {
    double cn = 1.0, cd = 1.0;
    if (!fold_constants(p.ncon, BinaryOp::Mul, 1.0, cn)) {
        cn = 1.0;
        for (double v : p.ncon) p.num.push_back(mk_const(v));
    }
    if (!fold_constants(p.dcon, BinaryOp::Mul, 1.0, cd)) {
        cd = 1.0;
        for (double v : p.dcon) p.den.push_back(mk_const(v));
    }
    double q = cn / cd;
    if (!std::isfinite(q)) {  // e.g. division by a merged 0: keep both sides literal
        if (cn != 1.0) p.num.push_back(mk_const(cn));
        if (cd != 1.0) p.den.push_back(mk_const(cd));
        q = cn = cd = 1.0;
    }

    std::vector<Ast> num = group_factors(std::move(p.num));
    std::vector<Ast> den = group_factors(std::move(p.den));

    // Keep `t / c` as a literal division when the numerator carries no constant of its
    // own and nothing else sits in the denominator: dividing by c directly is more
    // precise than multiplying by 1/c and reads better (docs/52's rejection of
    // t/c -> t*(1/c), carried forward). cd == -1 still merges (negate is exact).
    if (cn == 1.0 && cd != 1.0 && cd != -1.0 && den.empty() && !num.empty()) {
        return mk_binary(BinaryOp::Div, left_fold_mul(std::move(num)), mk_const(cd));
    }

    if (num.empty() && den.empty()) return mk_const(q);
    if (den.empty()) {
        Ast n = left_fold_mul(std::move(num));
        if (q == 1.0) return n;
        if (q == -1.0) return negate(std::move(n));
        return mk_binary(BinaryOp::Mul, std::move(n), mk_const(q));
    }
    Ast d = left_fold_mul(std::move(den));
    if (num.empty()) return mk_binary(BinaryOp::Div, mk_const(q), std::move(d));
    Ast base = mk_binary(BinaryOp::Div, left_fold_mul(std::move(num)), std::move(d));
    if (q == 1.0) return base;
    if (q == -1.0) return negate(std::move(base));
    return mk_binary(BinaryOp::Mul, std::move(base), mk_const(q));
}

// ---- sums ------------------------------------------------------------------------

struct SumTerm {
    double coeff = 1.0;
    Ast core;  // never Add/Sub; Div(1, D) stands for a reciprocal (coeff carries c/D's c)
};

// Split a normalised non-sum term into (coefficient, core) so like terms can merge.
SumTerm split_term(Ast t) {
    if (is_unary(t, UnaryOp::Neg)) {
        SumTerm s = split_term(std::move(t.kids[0]));
        s.coeff = -s.coeff;
        return s;
    }
    if (is_binary(t, BinaryOp::Mul) && is_const(t.kids[1])) {
        const double c = t.kids[1].node.value;
        return SumTerm{c, std::move(t.kids[0])};
    }
    if (is_binary(t, BinaryOp::Div) && is_const(t.kids[0])) {
        const double c = t.kids[0].node.value;
        t.kids[0].node.value = 1.0;
        return SumTerm{c, std::move(t)};
    }
    return SumTerm{1.0, std::move(t)};
}

Ast emit_term(double coeff, Ast core) {
    if (is_binary(core, BinaryOp::Div) && is_const(core.kids[0]) &&
        core.kids[0].node.value == 1.0) {
        core.kids[0].node.value = coeff;
        return core;
    }
    if (coeff == 1.0) return core;
    if (coeff == -1.0) return negate(std::move(core));
    return mk_binary(BinaryOp::Mul, std::move(core), mk_const(coeff));
}

void collect_raw_sum(Ast t, double sign, std::vector<SumTerm>& terms,
                     std::vector<double>& consts);

void add_normalized_term(Ast n, double sign, std::vector<SumTerm>& terms,
                         std::vector<double>& consts) {
    if (is_const(n)) {
        consts.push_back(sign * n.node.value);
        return;
    }
    // A leaf can normalise into a Sub (negate() of a Sub swaps operands): flatten it.
    if (is_binary(n, BinaryOp::Add) || is_binary(n, BinaryOp::Sub)) {
        collect_raw_sum(std::move(n), sign, terms, consts);
        return;
    }
    SumTerm s = split_term(std::move(n));
    s.coeff *= sign;  // exact: sign is ±1
    terms.push_back(std::move(s));
}

void collect_raw_sum(Ast t, double sign, std::vector<SumTerm>& terms,
                     std::vector<double>& consts) {
    if (is_binary(t, BinaryOp::Add)) {
        collect_raw_sum(std::move(t.kids[0]), sign, terms, consts);
        collect_raw_sum(std::move(t.kids[1]), sign, terms, consts);
    } else if (is_binary(t, BinaryOp::Sub)) {
        collect_raw_sum(std::move(t.kids[0]), sign, terms, consts);
        collect_raw_sum(std::move(t.kids[1]), -sign, terms, consts);
    } else if (is_unary(t, UnaryOp::Neg)) {
        collect_raw_sum(std::move(t.kids[0]), -sign, terms, consts);
    } else {
        add_normalized_term(normalize(std::move(t)), sign, terms, consts);
    }
}

Ast rebuild_sum(std::vector<SumTerm> terms, std::vector<double> consts) {
    double cacc = 0.0;
    if (!fold_constants(consts, BinaryOp::Add, 0.0, cacc)) {
        cacc = 0.0;
        for (double v : consts) terms.push_back(SumTerm{1.0, mk_const(v)});
    }

    std::sort(terms.begin(), terms.end(),
              [](const SumTerm& x, const SumTerm& y) { return cmp(x.core, y.core) < 0; });

    // Merge like terms. A merge whose coefficient sum is non-finite is skipped (both
    // terms kept). A sum collapsing to coefficient 0 keeps a `core * 0` term — never
    // a bare 0 — so a NaN/Inf that `core` produces still propagates (docs/54).
    std::vector<SumTerm> merged;
    for (SumTerm& t : terms) {
        if (!merged.empty() && ast_equal(merged.back().core, t.core)) {
            const double c = merged.back().coeff + t.coeff;
            if (std::isfinite(c)) {
                merged.back().coeff = (c == 0.0) ? 0.0 : c;  // normalise -0 to +0
                continue;
            }
        }
        merged.push_back(std::move(t));
    }

    if (merged.empty()) return mk_const(cacc);

    // Emission order: canonical order, but positive-coefficient terms lead — a
    // negative leader costs an extra neg() node ((-a) + b) where (b - a) says the
    // same thing one node smaller. When every term is negative, the constant (if any)
    // leads instead: (c - t1 - t2) rather than ((neg(t1) - t2) + c).
    std::stable_partition(merged.begin(), merged.end(),
                          [](const SumTerm& t) { return !(t.coeff < 0.0); });

    Ast acc;
    std::size_t start = 0;
    bool const_emitted = false;
    if (merged[0].coeff < 0.0 && cacc != 0.0) {
        acc = mk_const(cacc);
        const_emitted = true;
    } else {
        acc = emit_term(merged[0].coeff, std::move(merged[0].core));
        start = 1;
    }
    for (std::size_t i = start; i < merged.size(); ++i) {
        if (merged[i].coeff < 0.0) {
            acc = mk_binary(BinaryOp::Sub, std::move(acc),
                            emit_term(-merged[i].coeff, std::move(merged[i].core)));
        } else {
            acc = mk_binary(BinaryOp::Add, std::move(acc),
                            emit_term(merged[i].coeff, std::move(merged[i].core)));
        }
    }
    if (!const_emitted && cacc != 0.0) {  // also drops an exact -0 tail
        if (cacc < 0.0) {
            acc = mk_binary(BinaryOp::Sub, std::move(acc), mk_const(-cacc));
        } else {
            acc = mk_binary(BinaryOp::Add, std::move(acc), mk_const(cacc));
        }
    }
    return acc;
}

// ---- normalisation entry ----------------------------------------------------------

Ast normalize(Ast a) {
    switch (a.node.kind) {
        case NodeKind::Constant:
        case NodeKind::Variable:
            return a;
        case NodeKind::Unary:
            return norm_unary(a.node.uop, normalize(std::move(a.kids[0])));
        case NodeKind::Binary:
            break;
    }
    switch (a.node.bop) {
        case BinaryOp::Add:
        case BinaryOp::Sub: {
            std::vector<SumTerm> terms;
            std::vector<double> consts;
            collect_raw_sum(std::move(a), 1.0, terms, consts);
            return rebuild_sum(std::move(terms), std::move(consts));
        }
        case BinaryOp::Mul:
        case BinaryOp::Div: {
            ProductParts p;
            collect_raw_product(std::move(a), false, p);
            return rebuild_product(std::move(p));
        }
        case BinaryOp::Pow: {
            Ast left = normalize(std::move(a.kids[0]));
            Ast right = normalize(std::move(a.kids[1]));
            if (is_const(left) && is_const(right)) {
                const double v = detail::apply_binary<double>(BinaryOp::Pow, left.node.value,
                                                              right.node.value);
                if (std::isfinite(v)) return mk_const(v);
            }
            // No other Pow rewrite is sound under safe_pow semantics (docs/54).
            return mk_binary(BinaryOp::Pow, std::move(left), std::move(right));
        }
    }
    return a;  // unreachable
}

// Normalisation is applied to a fixpoint (a rewrite such as sqrt(square) -> abs can
// expose one more local rule at the parent). Convergence is fast — every pass is
// non-increasing in node count — but bounded defensively.
constexpr int kMaxNormalizePasses = 4;

bool node_equal(const Node& x, const Node& y) {
    if (x.kind != y.kind) return false;
    switch (x.kind) {
        case NodeKind::Constant: return double_bits(x.value) == double_bits(y.value);
        case NodeKind::Variable: return x.index == y.index;
        case NodeKind::Unary: return x.uop == y.uop;
        case NodeKind::Binary: return x.bop == y.bop;
    }
    return false;
}

bool tree_equal(const Tree& a, const Tree& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!node_equal(a[i], b[i])) return false;
    }
    return true;
}

}  // namespace

Tree display_simplify(const Tree& tree, DisplaySimplifyStats* stats,
                      const EGraphLimits& limits) {
    if (stats) *stats = DisplaySimplifyStats{};
    if (tree.empty()) return {};

    // Layer 1: deterministic normalisation to a fixpoint.
    Tree l1 = tree;
    for (int pass = 0; pass < kMaxNormalizePasses; ++pass) {
        Tree cur = to_tree(normalize(build(l1)));
        if (tree_equal(cur, l1)) break;
        l1 = std::move(cur);
    }

    // Layer 2: bounded equality saturation, adopted only on a strict improvement.
    if (limits.max_iterations > 0 && l1.size() > 2) {
        const EGraphResult er = egraph_simplify(l1, limits);
        if (stats) {
            stats->egraph_iterations = er.iterations;
            stats->egraph_enodes = er.enodes;
            stats->egraph_saturated = er.saturated;
        }
        if (er.ok && er.tree.size() < l1.size()) {
            // Re-normalise the extraction so it lands in the same canonical display
            // form Layer 1 produces (ordering, constant placement). Non-increasing.
            Tree l2 = er.tree;
            for (int pass = 0; pass < kMaxNormalizePasses; ++pass) {
                Tree cur = to_tree(normalize(build(l2)));
                if (tree_equal(cur, l2)) break;
                l2 = std::move(cur);
            }
            if (l2.size() < l1.size()) {
                if (stats) stats->layer2_adopted = true;
                return l2;
            }
        }
    }
    return l1;
}

}  // namespace rsymbolic
