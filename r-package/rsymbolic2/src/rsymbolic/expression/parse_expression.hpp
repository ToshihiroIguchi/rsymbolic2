// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/op_names.hpp"
#include "rsymbolic/expression/tree.hpp"

namespace rsymbolic {

// Recursive-descent parser for the infix grammar to_string() emits (tree.hpp), producing a
// postfix Tree. Used to turn a user-written macro-operator body into a template tree
// (macro_op.hpp); the R, Python and JS predict() paths each parse the same grammar in their
// own language, so this is the C++ member of that family — not a second expression format.
//
// Grammar (precedence low to high):
//   expr    := term (('+' | '-') term)*
//   term    := factor (('*' | '/') factor)*
//   factor  := unary ('^' factor)?            [right-associative]
//   unary   := '-' unary | primary
//   primary := number | name '(' expr ')' | '(' expr ')' | <argument identifier>
//
// Function names are the shared operator names (op_names.hpp); only unary calls are
// accepted, because a macro body is written in ordinary infix for its binary operators.
// The single identifier `arg_name` is emitted as variable_node(placeholder_index); every
// other identifier is an error, so a macro body cannot reference data columns.
//
// A literal negation is folded into the constant (`-2` is one Constant node, not
// neg(2)) — exact, and it keeps the macro's node cost minimal. Constants are emitted with
// index 0; the caller re-indexes (reindex_constants) once the tree is assembled.
//
// No exceptions: failures report `false` plus a human-readable message, so each binding can
// raise in its own idiom.

namespace parse_detail {

struct Token {
    enum Kind { Number, Ident, Op, LParen, RParen, Comma, End };
    Kind kind = End;
    double number = 0.0;
    std::string text;  // Ident
    char op = '\0';    // Op: + - * / ^
};

inline bool tokenize(const std::string& s, std::vector<Token>& out, std::string& error) {
    std::size_t i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && i + 1 < s.size() &&
             std::isdigit(static_cast<unsigned char>(s[i + 1])))) {
            std::size_t j = i;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) ||
                                    s[j] == '.')) {
                ++j;
            }
            // Exponent part: e/E, optional sign, at least one digit.
            if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
                std::size_t k = j + 1;
                if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
                if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
                    while (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
                        ++k;
                    }
                    j = k;
                }
            }
            Token t;
            t.kind = Token::Number;
            t.number = std::strtod(s.substr(i, j - i).c_str(), nullptr);
            out.push_back(t);
            i = j;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) ||
                                    s[j] == '_')) {
                ++j;
            }
            Token t;
            t.kind = Token::Ident;
            t.text = s.substr(i, j - i);
            out.push_back(t);
            i = j;
            continue;
        }
        Token t;
        if (c == '(') {
            t.kind = Token::LParen;
        } else if (c == ')') {
            t.kind = Token::RParen;
        } else if (c == ',') {
            // Only ever an error, but tokenising it lets the parser say something useful
            // about a two-argument call instead of "unexpected character".
            t.kind = Token::Comma;
        } else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
            t.kind = Token::Op;
            t.op = c;
        } else {
            error = std::string("unexpected character '") + c + "'";
            return false;
        }
        out.push_back(t);
        ++i;
    }
    Token end;
    end.kind = Token::End;
    out.push_back(end);
    return true;
}

// Emits postfix nodes into `out` as each subexpression completes.
struct Parser {
    const std::vector<Token>& tokens;
    std::size_t pos = 0;
    const std::string& arg_name;
    int placeholder_index;
    Tree& out;
    std::string error;

    const Token& peek() const { return tokens[pos]; }

    bool fail(const std::string& message) {
        if (error.empty()) error = message;
        return false;
    }

    bool parse_expr() {
        if (!parse_term()) return false;
        while (peek().kind == Token::Op && (peek().op == '+' || peek().op == '-')) {
            const char op = peek().op;
            ++pos;
            if (!parse_term()) return false;
            out.push_back(binary_node(op == '+' ? BinaryOp::Add : BinaryOp::Sub));
        }
        return true;
    }

    bool parse_term() {
        if (!parse_factor()) return false;
        while (peek().kind == Token::Op && (peek().op == '*' || peek().op == '/')) {
            const char op = peek().op;
            ++pos;
            if (!parse_factor()) return false;
            out.push_back(binary_node(op == '*' ? BinaryOp::Mul : BinaryOp::Div));
        }
        return true;
    }

    bool parse_factor() {
        if (!parse_unary()) return false;
        if (peek().kind == Token::Op && peek().op == '^') {
            ++pos;
            if (!parse_factor()) return false;  // right-associative
            out.push_back(binary_node(BinaryOp::Pow));
        }
        return true;
    }

    bool parse_unary() {
        if (peek().kind == Token::Op && peek().op == '-') {
            ++pos;
            // Fold a literal negation into the constant: exact, and one node instead of two.
            if (peek().kind == Token::Number) {
                out.push_back(constant_node(0, -peek().number));
                ++pos;
                return true;
            }
            if (!parse_unary()) return false;
            out.push_back(unary_node(UnaryOp::Neg));
            return true;
        }
        return parse_primary();
    }

    bool parse_primary() {
        const Token& t = peek();
        if (t.kind == Token::Number) {
            out.push_back(constant_node(0, t.number));
            ++pos;
            return true;
        }
        if (t.kind == Token::LParen) {
            ++pos;
            if (!parse_expr()) return false;
            if (peek().kind != Token::RParen) return fail("expected ')'");
            ++pos;
            return true;
        }
        if (t.kind == Token::Ident) {
            const std::string name = t.text;
            ++pos;
            if (peek().kind == Token::LParen) {
                UnaryOp op;
                if (!unary_from_name(name, op)) {
                    BinaryOp bop;
                    if (binary_from_name(name, bop)) {
                        return fail("'" + name +
                                    "' is a binary operator; write it in infix form");
                    }
                    return fail("unknown function '" + name + "'; use " +
                                unary_op_name_list());
                }
                ++pos;  // '('
                if (!parse_expr()) return false;
                if (peek().kind == Token::Comma) {
                    return fail("'" + name +
                                "' takes one argument; write binary operators in infix form");
                }
                if (peek().kind != Token::RParen) return fail("expected ')'");
                ++pos;
                out.push_back(unary_node(op));
                return true;
            }
            if (!arg_name.empty() && name == arg_name) {
                out.push_back(variable_node(placeholder_index));
                return true;
            }
            return fail("unknown identifier '" + name + "'; the only variable allowed is '" +
                        arg_name + "'");
        }
        return fail("unexpected end of expression");
    }
};

}  // namespace parse_detail

// Parse `text` into a postfix Tree. The identifier `arg_name` is emitted as
// variable_node(placeholder_index); no other variable is accepted. Returns false and fills
// `error` (never partially trusted output) on any syntax or name error.
inline bool parse_expression(const std::string& text, const std::string& arg_name,
                             int placeholder_index, Tree& out, std::string& error) {
    out.clear();
    std::vector<parse_detail::Token> tokens;
    if (!parse_detail::tokenize(text, tokens, error)) return false;
    if (tokens.size() == 1) {  // End only
        error = "empty expression";
        return false;
    }
    parse_detail::Parser parser{tokens, 0, arg_name, placeholder_index, out, {}};
    if (!parser.parse_expr()) {
        error = parser.error;
        out.clear();
        return false;
    }
    if (parser.peek().kind != parse_detail::Token::End) {
        error = "trailing input after a complete expression";
        out.clear();
        return false;
    }
    return true;
}

}  // namespace rsymbolic
