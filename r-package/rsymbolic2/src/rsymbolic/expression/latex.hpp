// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

namespace detail {

// Operator precedence levels for minimal-parenthesis LaTeX rendering. A fragment
// with a lower level must be parenthesized when embedded where a higher level is
// required (e.g. an Add-level child of a multiplication).
enum class LatexPrec { Add = 1, Mul = 2, Pow = 3, Atom = 4 };

inline std::string latex_paren(const std::string& s) {
    return "\\left( " + s + " \\right)";
}

inline std::string latex_paren_if(const std::string& s, LatexPrec prec,
                                  LatexPrec min_prec) {
    return static_cast<int>(prec) < static_cast<int>(min_prec) ? latex_paren(s) : s;
}

// Render a constant as LaTeX: plain decimals stay as-is, scientific notation is
// rewritten as m \cdot 10^{k}, and non-finite values become \infty / \mathrm{NaN}
// (degenerate fits can carry them; rendering must never fail). Returns the fragment
// and its precedence (a leading '-' binds like an addition-level expression).
inline std::pair<std::string, LatexPrec> latex_constant(double value, int precision) {
    if (std::isnan(value)) return {"\\mathrm{NaN}", LatexPrec::Atom};
    if (std::isinf(value)) {
        return value < 0 ? std::pair<std::string, LatexPrec>{"-\\infty", LatexPrec::Add}
                         : std::pair<std::string, LatexPrec>{"\\infty", LatexPrec::Atom};
    }

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, value);
    std::string s(buf);

    const std::size_t e = s.find_first_of("eE");
    if (e != std::string::npos) {
        const std::string mantissa = s.substr(0, e);
        const int exponent = std::atoi(s.c_str() + e + 1);
        s = mantissa + " \\cdot 10^{" + std::to_string(exponent) + "}";
        return {s, mantissa[0] == '-' ? LatexPrec::Add : LatexPrec::Mul};
    }
    return {s, s[0] == '-' ? LatexPrec::Add : LatexPrec::Atom};
}

}  // namespace detail

// Render a tree as LaTeX math (no surrounding $). Variables render as x_{i}
// (0-based, matching to_string's x0, x1, ...); callers that carry feature names
// substitute them for the x_{i} tokens. `precision` is the significant digits used
// for constants (the default matches to_string's %.6g).
//
// This is a display-only companion to to_string(): the infix string remains the
// evaluatable round-trip representation; LaTeX is never parsed back.
inline std::string to_latex(const Tree& tree, int precision = 6) {
    using detail::LatexPrec;
    using detail::latex_paren_if;

    std::vector<std::pair<std::string, LatexPrec>> stack;
    stack.reserve(tree.size());
    char buf[32];

    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant:
                stack.push_back(detail::latex_constant(node.value, precision));
                break;
            case NodeKind::Variable:
                std::snprintf(buf, sizeof(buf), "x_{%d}", node.index);
                stack.emplace_back(buf, LatexPrec::Atom);
                break;
            case NodeKind::Unary: {
                const auto a = stack.back();
                std::pair<std::string, LatexPrec> out;
                switch (node.uop) {
                    case UnaryOp::Neg:
                        out = {"-" + latex_paren_if(a.first, a.second, LatexPrec::Mul),
                               LatexPrec::Add};
                        break;
                    case UnaryOp::Exp:
                        out = {"e^{" + a.first + "}", LatexPrec::Pow};
                        break;
                    case UnaryOp::Log:
                        out = {"\\log\\left( " + a.first + " \\right)", LatexPrec::Atom};
                        break;
                    case UnaryOp::Sin:
                        out = {"\\sin\\left( " + a.first + " \\right)", LatexPrec::Atom};
                        break;
                    case UnaryOp::Cos:
                        out = {"\\cos\\left( " + a.first + " \\right)", LatexPrec::Atom};
                        break;
                    case UnaryOp::Sqrt:
                        out = {"\\sqrt{" + a.first + "}", LatexPrec::Atom};
                        break;
                    case UnaryOp::Tanh:
                        out = {"\\tanh\\left( " + a.first + " \\right)", LatexPrec::Atom};
                        break;
                    case UnaryOp::Abs:
                        out = {"\\left| " + a.first + " \\right|", LatexPrec::Atom};
                        break;
                    case UnaryOp::Square:
                        out = {latex_paren_if(a.first, a.second, LatexPrec::Atom) +
                                   "^{2}",
                               LatexPrec::Pow};
                        break;
                    case UnaryOp::Inv:
                        // \frac braces group on their own, like BinaryOp::Div below.
                        out = {"\\frac{1}{" + a.first + "}", LatexPrec::Atom};
                        break;
                }
                stack.back() = out;
                break;
            }
            case NodeKind::Binary: {
                const auto b = stack.back();
                stack.pop_back();
                const auto a = stack.back();
                std::pair<std::string, LatexPrec> out;
                switch (node.bop) {
                    case BinaryOp::Add: {
                        // Parenthesize a leading-minus right operand so "a + -2"
                        // never appears.
                        const std::string rhs = (!b.first.empty() && b.first[0] == '-')
                                                    ? detail::latex_paren(b.first)
                                                    : b.first;
                        out = {a.first + " + " + rhs, LatexPrec::Add};
                        break;
                    }
                    case BinaryOp::Sub:
                        out = {a.first + " - " +
                                   latex_paren_if(b.first, b.second, LatexPrec::Mul),
                               LatexPrec::Add};
                        break;
                    case BinaryOp::Mul:
                        out = {latex_paren_if(a.first, a.second, LatexPrec::Mul) +
                                   " \\cdot " +
                                   latex_paren_if(b.first, b.second, LatexPrec::Mul),
                               LatexPrec::Mul};
                        break;
                    case BinaryOp::Div:
                        // \frac braces are their own grouping; operands never need
                        // parentheses.
                        out = {"\\frac{" + a.first + "}{" + b.first + "}",
                               LatexPrec::Atom};
                        break;
                    case BinaryOp::Pow:
                        out = {latex_paren_if(a.first, a.second, LatexPrec::Atom) +
                                   "^{" + b.first + "}",
                               LatexPrec::Pow};
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
