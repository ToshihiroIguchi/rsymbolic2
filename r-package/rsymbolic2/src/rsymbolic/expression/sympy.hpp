// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

namespace detail {

// Operator precedence for minimal-parenthesis Python/SymPy rendering. A fragment with a
// lower level must be parenthesized where a higher one is required. `Pow` sits above `Mul`
// because `**` binds tighter than `*` and `/`, and `Atom` is reserved for fragments that
// need no parentheses anywhere — a bare name, a non-negative literal, or a function call.
enum class SympyPrec { Add = 1, Mul = 2, Pow = 3, Atom = 4 };

inline std::string sympy_paren_if(const std::string& s, SympyPrec prec, SympyPrec min_prec) {
    return static_cast<int>(prec) < static_cast<int>(min_prec) ? "(" + s + ")" : s;
}

// Render a constant as a Python literal. `%g` output (including `1e-05`) is already valid
// Python, so only the non-finite cases need mapping: SymPy parses `nan` and `oo`, where
// Python's own `float('nan')` spelling would not survive sympify(). A negative literal
// carries Add precedence for the same reason a Neg node does — `-3**2` is `-9` in Python,
// so a negative base must be parenthesized.
inline std::pair<std::string, SympyPrec> sympy_constant(double value, int precision) {
    if (std::isnan(value)) return {"nan", SympyPrec::Atom};
    if (std::isinf(value)) {
        return value < 0 ? std::pair<std::string, SympyPrec>{"-oo", SympyPrec::Add}
                         : std::pair<std::string, SympyPrec>{"oo", SympyPrec::Atom};
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, value);
    const std::string s(buf);
    return {s, s[0] == '-' ? SympyPrec::Add : SympyPrec::Atom};
}

}  // namespace detail

// Render a tree as a Python expression that SymPy's `sympify()` parses to the same
// mathematical function. Variables render as x0, x1, ... (matching to_string), which become
// Symbols; callers do not substitute feature names, because a column name is free text and
// `flow rate` is not a Python identifier.
//
// This is a display/export-only companion to to_string(), like to_latex(): the infix string
// stays the evaluatable round-trip representation and is never replaced by this one.
//
// What actually breaks in to_string()'s output, measured against SymPy 1.14 rather than
// assumed: `square(a)`, `inv(a)` and `neg(a)` have no SymPy function, and sympify() turns
// each into an undefined APPLIED FUNCTION instead of raising — a silently wrong expression,
// which is worse than an error. Two things that look equally broken are not: `^` survives
// because sympify() passes convert_xor=True by default (parse_expr on its own does not), and
// `abs` survives because Python's builtin abs() calls Symbol.__abs__ and yields Abs. Both
// are still emitted in Python form here (`**`, and `abs` left as-is), so the result is also
// valid under plain eval(), parse_expr() and NumPy — not only under sympify().
//
// CAVEAT, and it is the important one: this is the MATHEMATICAL form of the expression, not
// the engine's. The core's operators are domain-guarded (docs/69): sqrt, log and pow return
// NaN outside their domain where SymPy returns a complex value or a symbolic expression.
// Use predict() to evaluate; use this to differentiate, simplify or typeset.
inline std::string to_sympy(const Tree& tree, int precision = 6) {
    using detail::SympyPrec;
    using detail::sympy_paren_if;

    std::vector<std::pair<std::string, SympyPrec>> stack;
    stack.reserve(tree.size());
    char buf[32];

    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant:
                stack.push_back(detail::sympy_constant(node.value, precision));
                break;
            case NodeKind::Variable:
                std::snprintf(buf, sizeof(buf), "x%d", node.index);
                stack.emplace_back(buf, SympyPrec::Atom);
                break;
            case NodeKind::Unary: {
                const auto a = stack.back();
                std::pair<std::string, SympyPrec> out;
                switch (node.uop) {
                    case UnaryOp::Neg:
                        out = {"-" + sympy_paren_if(a.first, a.second, SympyPrec::Mul),
                               SympyPrec::Add};
                        break;
                    case UnaryOp::Square:
                        // The base must be an atom: `-3**2` is -9 and `1/x**2` is 1/(x**2).
                        out = {sympy_paren_if(a.first, a.second, SympyPrec::Atom) + "**2",
                               SympyPrec::Pow};
                        break;
                    case UnaryOp::Inv:
                        // Mul precedence, so `1/x*y` (which is y/x) never renders for
                        // inv(x)*y while `z/(1/x)` keeps its parentheses.
                        out = {"1/" + sympy_paren_if(a.first, a.second, SympyPrec::Pow),
                               SympyPrec::Mul};
                        break;
                    default:
                        // Every remaining operator (exp, log, sin, cos, sqrt, tanh, abs,
                        // erf, sinh, cosh) is a function of the same name in SymPy, so the
                        // engine's own name is already valid Python. Call form is atomic:
                        // sin(x)**2 needs no parentheses.
                        out = {std::string(detail::unary_name(node.uop)) + "(" + a.first +
                                   ")",
                               SympyPrec::Atom};
                        break;
                }
                stack.back() = out;
                break;
            }
            case NodeKind::Binary: {
                const auto b = stack.back();
                stack.pop_back();
                const auto a = stack.back();
                std::pair<std::string, SympyPrec> out;
                switch (node.bop) {
                    case BinaryOp::Add: {
                        // `a + -2` parses, but parenthesizing the leading minus keeps the
                        // output readable and matches to_latex's handling.
                        const std::string rhs = (!b.first.empty() && b.first[0] == '-')
                                                    ? "(" + b.first + ")"
                                                    : b.first;
                        out = {a.first + " + " + rhs, SympyPrec::Add};
                        break;
                    }
                    case BinaryOp::Sub:
                        out = {a.first + " - " +
                                   sympy_paren_if(b.first, b.second, SympyPrec::Mul),
                               SympyPrec::Add};
                        break;
                    case BinaryOp::Mul:
                        out = {sympy_paren_if(a.first, a.second, SympyPrec::Mul) + "*" +
                                   sympy_paren_if(b.first, b.second, SympyPrec::Mul),
                               SympyPrec::Mul};
                        break;
                    case BinaryOp::Div:
                        // `/` is left-associative, so a Mul-level divisor must be
                        // parenthesized: a/(b*c) is not a/b*c.
                        out = {sympy_paren_if(a.first, a.second, SympyPrec::Mul) + "/" +
                                   sympy_paren_if(b.first, b.second, SympyPrec::Pow),
                               SympyPrec::Mul};
                        break;
                    case BinaryOp::Pow:
                        // `**` is right-associative, so a Pow-level BASE needs parentheses
                        // too: (x**2)**3 is not x**2**3.
                        out = {sympy_paren_if(a.first, a.second, SympyPrec::Atom) + "**" +
                                   sympy_paren_if(b.first, b.second, SympyPrec::Atom),
                               SympyPrec::Pow};
                        break;
                }
                stack.back() = out;
                break;
            }
        }
    }
    return stack.empty() ? std::string() : stack.back().first;
}

}  // namespace rsymbolic
