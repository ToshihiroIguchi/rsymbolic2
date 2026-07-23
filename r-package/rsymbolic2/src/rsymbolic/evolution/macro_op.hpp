// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/parse_expression.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// User-defined MACRO OPERATORS: single-argument expression templates built out of the
// existing primitive operators (docs/57).
//
// PySR can accept an arbitrary operator because it has a Julia runtime to compile it.
// rsymbolic2 must not gain a runtime language, so a user-defined operator here is a
// *template*, not a new node kind: `gauss(x) = exp(neg(square(x)))` is stored as a small
// postfix tree with a placeholder where its argument goes, and it is EXPANDED into the
// expression the moment the search creates such a node. The engine's node set therefore
// stays closed, and evaluation, AD, the SoA kernels, to_string/to_latex, both simplifiers,
// dimensional analysis and predict() need no knowledge of macros at all.
//
// Consequences, all deliberate (docs/57 §2):
//   - Complexity is counted AFTER expansion: a 4-node macro costs 4 nodes. A macro biases
//     which structures get proposed, not what parsimony charges for them.
//   - Macros are invisible in the reported expression: results print the expanded
//     primitive form, which is exactly what keeps the frozen expression string
//     (docs/48 D2) evaluatable by every predict() implementation.
//   - Numeric literals in a body become ordinary tunable constants seeded at that value
//     (the engine has no frozen-constant concept), so a macro can also seed a starting
//     point rather than fix a coefficient.
//
// Off by default: an empty macro list leaves the search bit-identical to the PySR-parity
// default (CLAUDE.md "Opt-in high-accuracy options").

// Feature index marking the macro's argument slot inside a body. Never reaches evaluation:
// expand_macro() replaces the placeholder node with the argument subtree, and a body is
// rejected at parse time unless it contains exactly one placeholder.
inline constexpr int kMacroArgIndex = -1;

// The identifier a macro body uses for its argument, shared by every binding so a body
// written for one interface is valid in all of them.
inline constexpr const char* kMacroArgName = "x";

struct MacroOp {
    std::string name;         // user-facing name; diagnostics only, never in output
    Tree body;                // postfix, primitive operators only
    std::size_t placeholder = 0;  // index of the placeholder node within `body`
};

inline bool is_macro_placeholder(const Node& n) {
    return n.kind == NodeKind::Variable && n.index == kMacroArgIndex;
}

// Nodes a macro adds on top of its argument: body size minus the placeholder it consumes.
// A primitive unary operator's equivalent cost is 1.
inline int macro_extra_nodes(const MacroOp& m) {
    return static_cast<int>(m.body.size()) - 1;
}

// Build a macro from its user-written body, validating everything the search relies on.
// `arg_name` is the identifier standing for the macro's argument (conventionally "x").
// `max_nodes` is the search's size cap: a macro that cannot fit under it could never be
// inserted, so it is rejected at configuration time rather than silently never used.
// Returns false with a human-readable `error` (the caller raises in its own idiom).
inline bool make_macro_op(const std::string& name, const std::string& body,
                          const std::string& arg_name, int max_nodes, MacroOp& out,
                          std::string& error) {
    if (name.empty()) {
        error = "macro operator name must not be empty";
        return false;
    }
    UnaryOp clash_u;
    BinaryOp clash_b;
    if (unary_from_name(name, clash_u) || binary_from_name(name, clash_b)) {
        error = "macro operator '" + name + "' shadows a built-in operator";
        return false;
    }
    Tree tree;
    if (!parse_expression(body, arg_name, kMacroArgIndex, tree, error)) {
        error = "macro operator '" + name + "': " + error;
        return false;
    }
    std::size_t placeholders = 0;
    std::size_t placeholder_at = 0;
    for (std::size_t i = 0; i < tree.size(); ++i) {
        if (is_macro_placeholder(tree[i])) {
            ++placeholders;
            placeholder_at = i;
        }
    }
    // Exactly one occurrence: zero would make the macro a constant expression, and two or
    // more would duplicate the argument subtree on expansion (unbounded growth, and the
    // duplicate's constants would be fitted independently — never what the user drew).
    if (placeholders != 1) {
        error = "macro operator '" + name + "': body must use '" + arg_name +
                "' exactly once (found " + std::to_string(placeholders) + ")";
        return false;
    }
    // An operator-free body (just "x") would expand to its own argument: a growth mutation
    // that adds nothing. Harmless but meaningless, and it hides a typo.
    if (tree.size() < 2) {
        error = "macro operator '" + name + "': body must apply at least one operator to '" +
                arg_name + "'";
        return false;
    }
    if (max_nodes > 0 && static_cast<int>(tree.size()) - 1 >= max_nodes) {
        error = "macro operator '" + name + "': body has " +
                std::to_string(tree.size()) + " nodes, which cannot fit under max_nodes = " +
                std::to_string(max_nodes);
        return false;
    }
    out.name = name;
    out.body = std::move(tree);
    out.placeholder = placeholder_at;
    return true;
}

// Expand `m` over `arg`: splice the argument's postfix sequence into the placeholder's
// position. The result is a well-formed postfix tree whose constant indices still need
// re-indexing (every call site in mutation.cpp already calls reindex_constants).
inline Tree expand_macro(const MacroOp& m, const Tree& arg) {
    Tree out;
    out.reserve(m.body.size() - 1 + arg.size());
    out.insert(out.end(), m.body.begin(), m.body.begin() + m.placeholder);
    out.insert(out.end(), arg.begin(), arg.end());
    out.insert(out.end(), m.body.begin() + m.placeholder + 1, m.body.end());
    return out;
}

}  // namespace rsymbolic
