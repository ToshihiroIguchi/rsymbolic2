// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

// Bounded equality saturation for the display-only simplifier (docs/54). Self-
// contained egg-style e-graph (Willsey et al., POPL 2021): hash-consed e-nodes over
// union-find e-classes, congruence closure by repeated re-canonicalisation sweeps
// (simpler than egg's incremental repair and easily fast enough at the <=10k-node
// budget), top-down e-matching of a fixed audited rule set, and minimum-node-count
// extraction. No third-party dependency; never called from the search loop.

#include "rsymbolic/simplification/egraph.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rsymbolic {

namespace {

using Id = std::int32_t;

std::uint64_t double_bits(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

// One operator application (or leaf) whose children are e-class ids. Constants are
// compared by bit pattern so +0.0 / -0.0 stay distinct and folding never conflates
// values the evaluator would distinguish.
struct ENode {
    NodeKind kind = NodeKind::Constant;
    UnaryOp uop = UnaryOp::Neg;
    BinaryOp bop = BinaryOp::Add;
    int var_index = 0;   // Variable
    double value = 0.0;  // Constant
    Id a = -1, b = -1;   // child e-class ids (-1 = unused)

    bool operator==(const ENode& o) const {
        if (kind != o.kind) return false;
        switch (kind) {
            case NodeKind::Constant: return double_bits(value) == double_bits(o.value);
            case NodeKind::Variable: return var_index == o.var_index;
            case NodeKind::Unary: return uop == o.uop && a == o.a;
            case NodeKind::Binary: return bop == o.bop && a == o.a && b == o.b;
        }
        return false;
    }
};

struct ENodeHash {
    std::size_t operator()(const ENode& n) const {
        std::uint64_t h = static_cast<std::uint64_t>(n.kind) + 1;
        const auto mix = [&h](std::uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        switch (n.kind) {
            case NodeKind::Constant: mix(double_bits(n.value)); break;
            case NodeKind::Variable: mix(static_cast<std::uint64_t>(n.var_index)); break;
            case NodeKind::Unary:
                mix(static_cast<std::uint64_t>(n.uop));
                mix(static_cast<std::uint64_t>(n.a));
                break;
            case NodeKind::Binary:
                mix(static_cast<std::uint64_t>(n.bop));
                mix(static_cast<std::uint64_t>(n.a));
                mix(static_cast<std::uint64_t>(n.b));
                break;
        }
        return static_cast<std::size_t>(h);
    }
};

// Total order on e-nodes, used only to make extraction tie-breaks deterministic.
bool enode_less(const ENode& x, const ENode& y) {
    if (x.kind != y.kind) return x.kind < y.kind;
    switch (x.kind) {
        case NodeKind::Constant: return double_bits(x.value) < double_bits(y.value);
        case NodeKind::Variable: return x.var_index < y.var_index;
        case NodeKind::Unary:
            if (x.uop != y.uop) return x.uop < y.uop;
            return x.a < y.a;
        case NodeKind::Binary:
            if (x.bop != y.bop) return x.bop < y.bop;
            if (x.a != y.a) return x.a < y.a;
            return x.b < y.b;
    }
    return false;
}

// ---------------------------------------------------------------------------------
// E-graph: union-find over e-classes, a flat list of every e-node ever added (child
// ids possibly stale), and a memo of canonical e-node -> e-class. Congruence closure
// runs as whole-graph sweeps: re-canonicalise every node, merging classes whose nodes
// collide in the memo, until a sweep changes nothing. The per-class constant analysis
// (isfinite-guarded folding) is propagated in the same sweeps.
class EGraph {
  public:
    Id find(Id x) const {
        while (uf_[static_cast<std::size_t>(x)] != x) x = uf_[static_cast<std::size_t>(x)];
        return x;
    }

    ENode canonical(ENode n) const {
        if (n.a >= 0) n.a = find(n.a);
        if (n.b >= 0) n.b = find(n.b);
        return n;
    }

    Id add(ENode n) {
        n = canonical(n);
        const auto it = memo_.find(n);
        if (it != memo_.end()) return find(it->second);
        const Id id = static_cast<Id>(uf_.size());
        uf_.push_back(id);
        has_const_.push_back(false);
        const_value_.push_back(0.0);
        memo_.emplace(n, id);
        all_nodes_.push_back({n, id});
        double v;
        if (eval_const(n, v)) set_const(id, v);
        return find(id);
    }

    bool merge(Id x, Id y) {
        x = find(x);
        y = find(y);
        if (x == y) return false;
        if (x > y) std::swap(x, y);  // keep the older id as root (deterministic)
        uf_[static_cast<std::size_t>(y)] = x;
        // Analysis join. Two constant classes can merge with values a rounding step
        // apart (drift-class rules folding in different orders); keep the root's.
        if (!has_const_[static_cast<std::size_t>(x)] && has_const_[static_cast<std::size_t>(y)]) {
            has_const_[static_cast<std::size_t>(x)] = true;
            const_value_[static_cast<std::size_t>(x)] = const_value_[static_cast<std::size_t>(y)];
        }
        ++n_unions_;
        return true;
    }

    // Congruence + analysis fixpoint. Cheap at this scale: each sweep is one pass over
    // all_nodes_; sweeps repeat only while some merge happened.
    void rebuild() {
        bool changed = true;
        while (changed) {
            changed = false;
            memo_.clear();
            // Index loop: set_const may append a Constant e-node mid-sweep.
            for (std::size_t i = 0; i < all_nodes_.size(); ++i) {
                const ENode canon = canonical(all_nodes_[i].first);
                const Id cid = find(all_nodes_[i].second);
                const auto it = memo_.find(canon);
                if (it != memo_.end()) {
                    if (merge(cid, it->second)) changed = true;
                    memo_[canon] = find(cid);
                } else {
                    memo_.emplace(canon, cid);
                }
                double v;
                if (eval_const(canon, v) && !has_const_[static_cast<std::size_t>(find(cid))]) {
                    set_const(cid, v);
                    changed = true;
                }
            }
        }
    }

    int num_enodes() const { return static_cast<int>(memo_.size()); }

    bool class_const(Id id, double& out) const {
        id = find(id);
        if (!has_const_[static_cast<std::size_t>(id)]) return false;
        out = const_value_[static_cast<std::size_t>(id)];
        return true;
    }

    std::uint64_t n_unions() const { return n_unions_; }

    // Canonical e-nodes grouped by canonical class, in a deterministic (sorted) order.
    // Valid until the next add/merge.
    std::vector<std::pair<Id, ENode>> sorted_nodes() const {
        std::vector<std::pair<Id, ENode>> out;
        out.reserve(memo_.size());
        for (const auto& kv : memo_) out.push_back({find(kv.second), canonical(kv.first)});
        std::sort(out.begin(), out.end(), [](const auto& p, const auto& q) {
            if (p.first != q.first) return p.first < q.first;
            return enode_less(p.second, q.second);
        });
        out.erase(std::unique(out.begin(), out.end(),
                              [](const auto& p, const auto& q) {
                                  return p.first == q.first && p.second == q.second;
                              }),
                  out.end());
        return out;
    }

    std::size_t num_classes() const { return uf_.size(); }

  private:
    bool eval_const(const ENode& n, double& out) const {
        switch (n.kind) {
            case NodeKind::Constant:
                out = n.value;
                return true;
            case NodeKind::Variable:
                return false;
            case NodeKind::Unary: {
                double a;
                if (!class_const(n.a, a)) return false;
                const double v = detail::apply_unary<double>(n.uop, a);
                if (!std::isfinite(v)) return false;  // FP policy: never fold to NaN/Inf
                out = v;
                return true;
            }
            case NodeKind::Binary: {
                double a, b;
                if (!class_const(n.a, a) || !class_const(n.b, b)) return false;
                const double v = detail::apply_binary<double>(n.bop, a, b);
                if (!std::isfinite(v)) return false;
                out = v;
                return true;
            }
        }
        return false;
    }

    void set_const(Id id, double v) {
        id = find(id);
        if (has_const_[static_cast<std::size_t>(id)]) return;
        has_const_[static_cast<std::size_t>(id)] = true;
        const_value_[static_cast<std::size_t>(id)] = v;
        // egg's "modify": add the literal so extraction can pick it at cost 1.
        ENode k;
        k.kind = NodeKind::Constant;
        k.value = v;
        merge(id, add(k));
    }

    std::vector<Id> uf_;
    std::vector<bool> has_const_;
    std::vector<double> const_value_;
    std::unordered_map<ENode, Id, ENodeHash> memo_;
    std::vector<std::pair<ENode, Id>> all_nodes_;  // as-added (ids may be stale)
    std::uint64_t n_unions_ = 0;
};

// ---------------------------------------------------------------------------------
// Patterns and rules. A pattern is a small tree over pattern variables, exact
// constant literals (bit-compared, so e.g. the add-0 rule matches +0.0 only), and
// operator applications.

struct Pat;
using PP = std::shared_ptr<const Pat>;

struct Pat {
    enum Kind { Var, Lit, Un, Bin } kind = Var;
    int var = 0;
    double lit = 0.0;
    UnaryOp uop = UnaryOp::Neg;
    BinaryOp bop = BinaryOp::Add;
    PP x, y;
};

PP pv(int v) {
    auto p = std::make_shared<Pat>();
    p->kind = Pat::Var;
    p->var = v;
    return p;
}
PP pc(double c) {
    auto p = std::make_shared<Pat>();
    p->kind = Pat::Lit;
    p->lit = c;
    return p;
}
PP pu(UnaryOp op, PP x) {
    auto p = std::make_shared<Pat>();
    p->kind = Pat::Un;
    p->uop = op;
    p->x = std::move(x);
    return p;
}
PP pb(BinaryOp op, PP x, PP y) {
    auto p = std::make_shared<Pat>();
    p->kind = Pat::Bin;
    p->bop = op;
    p->x = std::move(x);
    p->y = std::move(y);
    return p;
}

struct Rule {
    const char* name;
    PP lhs;
    PP rhs;
};

constexpr int kMaxPatVars = 3;

// The audited rule set (docs/54 lists every rule with its floating-point class).
// Directions are spelled out explicitly: saturation has no implicit symmetry.
// Excluded on purpose (docs/54 "Excluded rules"): x*0 -> 0 and x-x -> 0 (discard a
// NaN/Inf that x would produce), x/x -> 1 (NaN at x=0/Inf), exp(log x) -> x (NaN for
// x<0), log(exp x) -> x (Inf-ness changes past the exp overflow threshold),
// square(sqrt x) -> x (NaN for x<0), log/exp/sqrt product-sum expansions (domain
// splits), and every Pow rewrite (safe_pow's 0-fallback for NaN/undefined bases means
// even pow(t,2) <-> square(t) changes NaN behaviour).
std::vector<Rule> build_rules() {
    const PP a = pv(0), b = pv(1), c = pv(2);
    const auto A = [](PP x, PP y) { return pb(BinaryOp::Add, std::move(x), std::move(y)); };
    const auto S = [](PP x, PP y) { return pb(BinaryOp::Sub, std::move(x), std::move(y)); };
    const auto M = [](PP x, PP y) { return pb(BinaryOp::Mul, std::move(x), std::move(y)); };
    const auto D = [](PP x, PP y) { return pb(BinaryOp::Div, std::move(x), std::move(y)); };
    const auto N = [](PP x) { return pu(UnaryOp::Neg, std::move(x)); };
    const auto SQ = [](PP x) { return pu(UnaryOp::Square, std::move(x)); };
    const auto AB = [](PP x) { return pu(UnaryOp::Abs, std::move(x)); };

    std::vector<Rule> r;
    // --- commutativity / associativity (reassociation drift; docs/54 FP class B) ---
    r.push_back({"add-comm", A(a, b), A(b, a)});
    r.push_back({"mul-comm", M(a, b), M(b, a)});
    r.push_back({"add-assoc-l", A(A(a, b), c), A(a, A(b, c))});
    r.push_back({"add-assoc-r", A(a, A(b, c)), A(A(a, b), c)});
    r.push_back({"mul-assoc-l", M(M(a, b), c), M(a, M(b, c))});
    r.push_back({"mul-assoc-r", M(a, M(b, c)), M(M(a, b), c)});
    // --- subtraction / negation (IEEE-exact; class A) ---
    r.push_back({"sub-to-add-neg", S(a, b), A(a, N(b))});
    r.push_back({"add-neg-to-sub", A(a, N(b)), S(a, b)});
    r.push_back({"neg-neg", N(N(a)), a});
    r.push_back({"neg-sub", N(S(a, b)), S(b, a)});
    r.push_back({"sub-neg", S(a, N(b)), A(a, b)});
    r.push_back({"neg-mul-in", N(M(a, b)), M(N(a), b)});
    r.push_back({"neg-mul-out", M(N(a), b), N(M(a, b))});
    r.push_back({"neg-as-mul", N(a), M(a, pc(-1.0))});
    r.push_back({"mul-neg-one", M(a, pc(-1.0)), N(a)});
    r.push_back({"div-neg-num-out", D(N(a), b), N(D(a, b))});
    r.push_back({"div-neg-num-in", N(D(a, b)), D(N(a), b)});
    r.push_back({"div-neg-den", D(a, N(b)), N(D(a, b))});
    // --- division chains (drift; class B) ---
    r.push_back({"div-div-num", D(D(a, b), c), D(a, M(b, c))});
    r.push_back({"div-mul-den", D(a, M(b, c)), D(D(a, b), c)});
    r.push_back({"div-div-den", D(a, D(b, c)), D(M(a, c), b)});
    r.push_back({"mul-div-l", M(D(a, b), c), D(M(a, c), b)});
    r.push_back({"div-of-mul", D(M(a, b), c), M(a, D(b, c))});
    r.push_back({"mul-of-div", M(a, D(b, c)), D(M(a, b), c)});
    // --- identities (exact on the matched literal; class A) ---
    r.push_back({"add-zero", A(a, pc(0.0)), a});
    r.push_back({"sub-zero", S(a, pc(0.0)), a});
    r.push_back({"mul-one", M(a, pc(1.0)), a});
    r.push_back({"div-one", D(a, pc(1.0)), a});
    // --- distributivity / factoring (drift; class B — the size-reduction driver) ---
    r.push_back({"distrib", M(a, A(b, c)), A(M(a, b), M(a, c))});
    r.push_back({"factor", A(M(a, b), M(a, c)), M(a, A(b, c))});
    r.push_back({"factor-one", A(M(a, b), a), M(a, A(b, pc(1.0)))});
    r.push_back({"add-self", A(a, a), M(a, pc(2.0))});
    r.push_back({"div-add-common", A(D(a, c), D(b, c)), D(A(a, b), c)});
    // --- square (docs/54: square(t) evaluates as t*t, so mul-self is bit-exact) ---
    r.push_back({"mul-self", M(a, a), SQ(a)});
    r.push_back({"square-expand", SQ(a), M(a, a)});
    r.push_back({"square-neg", SQ(N(a)), SQ(a)});
    r.push_back({"square-abs", SQ(AB(a)), SQ(a)});
    r.push_back({"square-mul", SQ(M(a, b)), M(SQ(a), SQ(b))});
    r.push_back({"square-mul-join", M(SQ(a), SQ(b)), SQ(M(a, b))});
    r.push_back({"square-div", SQ(D(a, b)), D(SQ(a), SQ(b))});
    r.push_back({"square-div-join", D(SQ(a), SQ(b)), SQ(D(a, b))});
    // --- abs (exact: |.| introduces no rounding; sign-symmetric rounding) ---
    r.push_back({"abs-neg", AB(N(a)), AB(a)});
    r.push_back({"abs-abs", AB(AB(a)), AB(a)});
    r.push_back({"abs-square", AB(SQ(a)), SQ(a)});
    r.push_back({"abs-sqrt", AB(pu(UnaryOp::Sqrt, a)), pu(UnaryOp::Sqrt, a)});
    r.push_back({"abs-exp", AB(pu(UnaryOp::Exp, a)), pu(UnaryOp::Exp, a)});
    r.push_back({"abs-mul", AB(M(a, b)), M(AB(a), AB(b))});
    r.push_back({"abs-mul-join", M(AB(a), AB(b)), AB(M(a, b))});
    r.push_back({"abs-div", AB(D(a, b)), D(AB(a), AB(b))});
    r.push_back({"abs-div-join", D(AB(a), AB(b)), AB(D(a, b))});
    // --- sqrt of a square (drift + overflow caveat, adopted; docs/54) ---
    r.push_back({"sqrt-square", pu(UnaryOp::Sqrt, SQ(a)), AB(a)});
    // --- odd/even unaries (libm sign symmetry; class B) ---
    r.push_back({"sin-neg-out", pu(UnaryOp::Sin, N(a)), N(pu(UnaryOp::Sin, a))});
    r.push_back({"sin-neg-in", N(pu(UnaryOp::Sin, a)), pu(UnaryOp::Sin, N(a))});
    r.push_back({"tanh-neg-out", pu(UnaryOp::Tanh, N(a)), N(pu(UnaryOp::Tanh, a))});
    r.push_back({"tanh-neg-in", N(pu(UnaryOp::Tanh, a)), pu(UnaryOp::Tanh, N(a))});
    r.push_back({"cos-neg", pu(UnaryOp::Cos, N(a)), pu(UnaryOp::Cos, a)});
    return r;
}

const std::vector<Rule>& rules() {
    static const std::vector<Rule> r = build_rules();
    return r;
}

// ---------------------------------------------------------------------------------
// E-matching: top-down over the per-class canonical node lists.

struct Subst {
    std::array<Id, kMaxPatVars> bind{{-1, -1, -1}};
};

using ClassNodes = std::vector<std::vector<ENode>>;  // canonical class id -> nodes

void match_pat(const EGraph& g, const ClassNodes& cn, const Pat& p, Id cid, Subst s,
               std::vector<Subst>& out) {
    switch (p.kind) {
        case Pat::Var: {
            Id& slot = s.bind[static_cast<std::size_t>(p.var)];
            if (slot < 0) {
                slot = cid;
                out.push_back(s);
            } else if (slot == cid) {
                out.push_back(s);
            }
            return;
        }
        case Pat::Lit: {
            double v;
            if (g.class_const(cid, v) && double_bits(v) == double_bits(p.lit)) {
                out.push_back(s);
            }
            return;
        }
        case Pat::Un: {
            for (const ENode& n : cn[static_cast<std::size_t>(cid)]) {
                if (n.kind != NodeKind::Unary || n.uop != p.uop) continue;
                match_pat(g, cn, *p.x, n.a, s, out);
            }
            return;
        }
        case Pat::Bin: {
            for (const ENode& n : cn[static_cast<std::size_t>(cid)]) {
                if (n.kind != NodeKind::Binary || n.bop != p.bop) continue;
                std::vector<Subst> left;
                match_pat(g, cn, *p.x, n.a, s, left);
                for (const Subst& sl : left) {
                    match_pat(g, cn, *p.y, n.b, sl, out);
                }
            }
            return;
        }
    }
}

Id instantiate(EGraph& g, const Pat& p, const Subst& s) {
    switch (p.kind) {
        case Pat::Var:
            return g.find(s.bind[static_cast<std::size_t>(p.var)]);
        case Pat::Lit: {
            ENode n;
            n.kind = NodeKind::Constant;
            n.value = p.lit;
            return g.add(n);
        }
        case Pat::Un: {
            ENode n;
            n.kind = NodeKind::Unary;
            n.uop = p.uop;
            n.a = instantiate(g, *p.x, s);
            return g.add(n);
        }
        case Pat::Bin: {
            ENode n;
            n.kind = NodeKind::Binary;
            n.bop = p.bop;
            n.a = instantiate(g, *p.x, s);
            n.b = instantiate(g, *p.y, s);
            return g.add(n);
        }
    }
    return -1;
}

// ---------------------------------------------------------------------------------
// Extraction: per-class minimum tree size by fixpoint relaxation, deterministic
// tie-break by enode_less over the sorted node list.

void emit_best(const std::vector<ENode>& best, Id cid, Tree& out) {
    const ENode& e = best[static_cast<std::size_t>(cid)];
    switch (e.kind) {
        case NodeKind::Constant:
            out.push_back(constant_node(0, e.value));
            break;
        case NodeKind::Variable:
            out.push_back(variable_node(e.var_index));
            break;
        case NodeKind::Unary:
            emit_best(best, e.a, out);
            out.push_back(unary_node(e.uop));
            break;
        case NodeKind::Binary:
            emit_best(best, e.a, out);
            emit_best(best, e.b, out);
            out.push_back(binary_node(e.bop));
            break;
    }
}

bool extract(const EGraph& g, Id root, Tree& out) {
    const auto nodes = g.sorted_nodes();
    const std::size_t n_classes = g.num_classes();
    const int kInf = 1 << 28;
    std::vector<int> cost(n_classes, kInf);
    std::vector<ENode> best(n_classes);
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& [cid, e] : nodes) {
            int c = 1;
            if (e.a >= 0) c += cost[static_cast<std::size_t>(e.a)];
            if (e.b >= 0) c += cost[static_cast<std::size_t>(e.b)];
            if (c >= kInf) continue;
            const std::size_t ci = static_cast<std::size_t>(cid);
            if (c < cost[ci] || (c == cost[ci] && enode_less(e, best[ci]))) {
                cost[ci] = c;
                best[ci] = e;
                changed = true;
            }
        }
    }
    root = g.find(root);
    if (cost[static_cast<std::size_t>(root)] >= kInf) return false;
    out.clear();
    emit_best(best, root, out);
    reindex_constants(out);
    return true;
}

}  // namespace

EGraphResult egraph_simplify(const Tree& tree, const EGraphLimits& limits) {
    EGraphResult result;
    if (tree.empty()) return result;

    const auto t0 = std::chrono::steady_clock::now();
    const auto over_time = [&]() {
        const std::chrono::duration<double, std::milli> dt =
            std::chrono::steady_clock::now() - t0;
        return dt.count() > limits.max_millis;
    };

    EGraph g;
    // Seed: the postfix tree maps directly onto adds via a stack.
    std::vector<Id> stack;
    stack.reserve(tree.size());
    for (const Node& node : tree) {
        ENode n;
        switch (node.kind) {
            case NodeKind::Constant:
                n.kind = NodeKind::Constant;
                n.value = node.value;
                break;
            case NodeKind::Variable:
                n.kind = NodeKind::Variable;
                n.var_index = node.index;
                break;
            case NodeKind::Unary:
                n.kind = NodeKind::Unary;
                n.uop = node.uop;
                n.a = stack.back();
                stack.pop_back();
                break;
            case NodeKind::Binary:
                n.kind = NodeKind::Binary;
                n.bop = node.bop;
                n.b = stack.back();
                stack.pop_back();
                n.a = stack.back();
                stack.pop_back();
                break;
        }
        stack.push_back(g.add(n));
    }
    const Id root = stack.back();
    g.rebuild();

    for (int iter = 0; iter < limits.max_iterations; ++iter) {
        if (over_time() || g.num_enodes() > limits.max_enodes) break;
        result.iterations = iter + 1;

        // Read phase: collect matches against a stable snapshot.
        ClassNodes cn(g.num_classes());
        for (const auto& [cid, e] : g.sorted_nodes()) {
            cn[static_cast<std::size_t>(cid)].push_back(e);
        }
        std::vector<std::pair<Id, std::pair<const Pat*, Subst>>> matches;
        for (const Rule& rule : rules()) {
            for (std::size_t cid = 0; cid < cn.size(); ++cid) {
                if (cn[cid].empty()) continue;  // non-canonical or empty class
                std::vector<Subst> subs;
                match_pat(g, cn, *rule.lhs, static_cast<Id>(cid), Subst{}, subs);
                for (const Subst& s : subs) {
                    matches.push_back({static_cast<Id>(cid), {rule.rhs.get(), s}});
                }
            }
            if (over_time()) break;
        }

        // Write phase: instantiate + union, respecting the node cap.
        const std::uint64_t unions_before = g.n_unions();
        const int enodes_before = g.num_enodes();
        bool capped = false;
        for (const auto& m : matches) {
            if (g.num_enodes() > limits.max_enodes || over_time()) {
                capped = true;
                break;
            }
            const Id rhs = instantiate(g, *m.second.first, m.second.second);
            g.merge(m.first, rhs);
        }
        g.rebuild();

        if (!capped && g.n_unions() == unions_before && g.num_enodes() == enodes_before) {
            result.saturated = true;
            break;
        }
        if (capped) break;
    }

    result.enodes = g.num_enodes();
    result.ok = extract(g, root, result.tree);
    return result;
}

}  // namespace rsymbolic
