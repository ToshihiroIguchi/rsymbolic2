// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.

#pragma once

#include <string>

#include "rsymbolic/expression/node.hpp"

namespace rsymbolic {

// The user-facing operator names, in one place.
//
// Every binding (R, Python, WebAssembly) accepts operators as strings and used to carry
// its own copy of the name -> enum mapping; the three copies had to be edited in lockstep
// whenever an operator was added. This header is that mapping, shared by all of them.
// Rendering in the other direction stays in tree.hpp (detail::unary_name / binary_name),
// which is what produces the expression strings.
//
// Adding an operator: extend the enum in node.hpp, add one row here, and every binding
// accepts the new name without further edits.

struct UnaryOpName {
    const char* name;
    UnaryOp op;
};

struct BinaryOpName {
    const char* name;
    BinaryOp op;
};

inline constexpr UnaryOpName kUnaryOpNames[] = {
    {"neg", UnaryOp::Neg},     {"exp", UnaryOp::Exp},   {"log", UnaryOp::Log},
    {"sin", UnaryOp::Sin},     {"cos", UnaryOp::Cos},   {"sqrt", UnaryOp::Sqrt},
    {"tanh", UnaryOp::Tanh},   {"abs", UnaryOp::Abs},   {"square", UnaryOp::Square},
    {"inv", UnaryOp::Inv},
};

inline constexpr BinaryOpName kBinaryOpNames[] = {
    {"add", BinaryOp::Add}, {"sub", BinaryOp::Sub}, {"mul", BinaryOp::Mul},
    {"div", BinaryOp::Div}, {"pow", BinaryOp::Pow},
};

// Look up an operator by its user-facing name. Returns false (leaving `out` untouched)
// when the name is not recognised; the caller raises the error in its own idiom.
inline bool unary_from_name(const std::string& s, UnaryOp& out) {
    for (const UnaryOpName& entry : kUnaryOpNames) {
        if (s == entry.name) {
            out = entry.op;
            return true;
        }
    }
    return false;
}

inline bool binary_from_name(const std::string& s, BinaryOp& out) {
    for (const BinaryOpName& entry : kBinaryOpNames) {
        if (s == entry.name) {
            out = entry.op;
            return true;
        }
    }
    return false;
}

// "neg/exp/log/..." — the recognised names, for error messages.
inline std::string unary_op_name_list() {
    std::string out;
    for (const UnaryOpName& entry : kUnaryOpNames) {
        if (!out.empty()) out += "/";
        out += entry.name;
    }
    return out;
}

inline std::string binary_op_name_list() {
    std::string out;
    for (const BinaryOpName& entry : kBinaryOpNames) {
        if (!out.empty()) out += "/";
        out += entry.name;
    }
    return out;
}

}  // namespace rsymbolic
