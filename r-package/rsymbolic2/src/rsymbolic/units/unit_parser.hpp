// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#pragma once

#include <cctype>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "rsymbolic/units/dimension.hpp"

// Parse a DynamicQuantities-style unit string (as accepted by PySR's X_units / y_units)
// into a Dimension. Only the dimension matters here, so unit magnitude/prefixes are
// accepted and discarded (e.g. "km" and "m" parse to the same Dimension).
//
// Supported grammar (recursive descent):
//   expr     := term ( ('*' | '/') term )*
//   term     := factor ( '^' exponent )?
//   factor   := number | symbol | '(' expr ')'
//   exponent := '(' signed_rational ')' | signed_rational
//   signed_rational := ('-'|'+')? uint ( ('/' | '//') uint )?      # Julia '//' allowed
//   number   := decimal literal (treated as dimensionless)
//   symbol   := unit | prefix unit                                 # longest unit match first
//
// Supported symbols: SI base (kg, g, m, s, A, K, mol, cd); common derived units
// (N, J, W, Pa, C, V, Ohm, T, Hz, F, H, Wb, S; rad, sr are dimensionless); the
// dimensionless literal "1"; decimal SI prefixes y..Y (dimensionally inert). The
// UTF-8 symbols "µ" and "Ω" are accepted as aliases of "u" and "Ohm". Products must be
// explicit ("N*m", not "N m"), matching Julia syntax.
//
// Throws std::invalid_argument (with the offending token) on unknown symbols,
// malformed syntax, offset units (°C/°F), or non-representable fractional exponents.

namespace rsymbolic {

namespace units_detail {

inline Dimension make_dim(int mass, int len, int tim, int cur, int tmp, int amt, int lum) {
    Dimension d;
    d.num[0] = mass * Dimension::DEN;
    d.num[1] = len * Dimension::DEN;
    d.num[2] = tim * Dimension::DEN;
    d.num[3] = cur * Dimension::DEN;
    d.num[4] = tmp * Dimension::DEN;
    d.num[5] = amt * Dimension::DEN;
    d.num[6] = lum * Dimension::DEN;
    return d;
}

// Full-symbol lookup (no prefix). Returns false if `w` is not a known unit.
inline bool lookup_unit(const std::string& w, Dimension& out) {
    static const std::map<std::string, Dimension> table = {
        // SI base
        {"kg", make_dim(1, 0, 0, 0, 0, 0, 0)},  {"g", make_dim(1, 0, 0, 0, 0, 0, 0)},
        {"m", make_dim(0, 1, 0, 0, 0, 0, 0)},   {"s", make_dim(0, 0, 1, 0, 0, 0, 0)},
        {"A", make_dim(0, 0, 0, 1, 0, 0, 0)},   {"K", make_dim(0, 0, 0, 0, 1, 0, 0)},
        {"mol", make_dim(0, 0, 0, 0, 0, 1, 0)}, {"cd", make_dim(0, 0, 0, 0, 0, 0, 1)},
        // Derived (Feynman coverage + common electromagnetism)
        {"N", make_dim(1, 1, -2, 0, 0, 0, 0)},  {"J", make_dim(1, 2, -2, 0, 0, 0, 0)},
        {"W", make_dim(1, 2, -3, 0, 0, 0, 0)},  {"Pa", make_dim(1, -1, -2, 0, 0, 0, 0)},
        {"C", make_dim(0, 0, 1, 1, 0, 0, 0)},   {"V", make_dim(1, 2, -3, -1, 0, 0, 0)},
        {"Ohm", make_dim(1, 2, -3, -2, 0, 0, 0)},
        {"T", make_dim(1, 0, -2, -1, 0, 0, 0)}, {"Hz", make_dim(0, 0, -1, 0, 0, 0, 0)},
        {"F", make_dim(-1, -2, 4, 2, 0, 0, 0)}, {"H", make_dim(1, 2, -2, -2, 0, 0, 0)},
        {"Wb", make_dim(1, 2, -2, -1, 0, 0, 0)},
        {"S", make_dim(-1, -2, 3, 2, 0, 0, 0)},
        // Dimensionless "units"
        {"rad", make_dim(0, 0, 0, 0, 0, 0, 0)}, {"sr", make_dim(0, 0, 0, 0, 0, 0, 0)},
    };
    const auto it = table.find(w);
    if (it == table.end()) return false;
    out = it->second;
    return true;
}

// Strip a decimal SI prefix (case-sensitive) leaving a non-empty remainder. Prefixes are
// dimensionally inert, so the caller only needs the remainder. "da" (deca) is the sole
// two-character prefix and is tried first.
inline bool strip_prefix(const std::string& w, std::string& rest) {
    if (w.size() > 2 && w[0] == 'd' && w[1] == 'a') {
        rest = w.substr(2);
        return true;
    }
    static const std::string one = "yzafpnumcdhkMGTPEZY";  // 1-char prefixes, case-sensitive
    if (w.size() > 1 && one.find(w[0]) != std::string::npos) {
        rest = w.substr(1);
        return true;
    }
    return false;
}

// Rewrite the two UTF-8 unit symbols we accept into their ASCII aliases so the rest of
// the parser can work byte-by-byte over ASCII.
inline std::string normalize_utf8(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == 0xC2 && i + 1 < s.size() &&
            static_cast<unsigned char>(s[i + 1]) == 0xB5) {  // µ (micro)
            r += 'u';
            i += 2;
        } else if (c == 0xCE && i + 1 < s.size() &&
                   static_cast<unsigned char>(s[i + 1]) == 0xA9) {  // Ω (ohm)
            r += "Ohm";
            i += 2;
        } else {
            r += static_cast<char>(c);
            ++i;
        }
    }
    return r;
}

struct Parser {
    std::string s;
    std::size_t pos = 0;
    const std::string& original;

    explicit Parser(std::string normalized, const std::string& orig)
        : s(std::move(normalized)), original(orig) {}

    bool at_end() const { return pos >= s.size(); }
    char peek() const { return at_end() ? '\0' : s[pos]; }
    void advance() { ++pos; }
    void skip_ws() {
        while (!at_end() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::invalid_argument("invalid unit '" + original + "': " + msg);
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        advance();
    }

    std::string read_word() {
        const std::size_t start = pos;
        while (!at_end() && std::isalpha(static_cast<unsigned char>(s[pos]))) ++pos;
        return s.substr(start, pos - start);
    }

    // Consume a decimal literal (dimensionless magnitude); its value is discarded.
    void read_number() {
        while (!at_end() &&
               (std::isdigit(static_cast<unsigned char>(s[pos])) || s[pos] == '.'))
            ++pos;
        if (!at_end() && (s[pos] == 'e' || s[pos] == 'E')) {
            ++pos;
            if (!at_end() && (s[pos] == '+' || s[pos] == '-')) ++pos;
            while (!at_end() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
        }
    }

    int read_uint() {
        const std::size_t start = pos;
        while (!at_end() && std::isdigit(static_cast<unsigned char>(s[pos]))) ++pos;
        if (pos == start) fail("expected an integer");
        return std::stoi(s.substr(start, pos - start));
    }

    std::pair<int, int> parse_exponent() {
        skip_ws();
        bool paren = false;
        if (peek() == '(') {
            advance();
            paren = true;
            skip_ws();
        }
        int sign = 1;
        if (peek() == '-') {
            advance();
            sign = -1;
        } else if (peek() == '+') {
            advance();
        }
        const int p = read_uint();
        int q = 1;
        // A rational exponent is only recognised inside parentheses ("m^(1/2)"): outside
        // them a '/' is the division operator ("m^2/s^2" == (m^2)/(s^2)).
        if (paren) {
            skip_ws();
            if (peek() == '/') {
                advance();
                if (peek() == '/') advance();  // Julia rational '//'
                skip_ws();
                q = read_uint();
                if (q == 0) fail("zero denominator in exponent");
            }
            skip_ws();
            expect(')');
        }
        return {sign * p, q};
    }

    Dimension read_symbol() {
        const std::string w = read_word();
        Dimension d;
        if (lookup_unit(w, d)) return d;
        std::string rest;
        if (strip_prefix(w, rest) && lookup_unit(rest, d)) return d;
        fail("unknown unit symbol '" + w + "'");
    }

    Dimension parse_factor() {
        skip_ws();
        const char c = peek();
        if (c == '(') {
            advance();
            const Dimension d = parse_expr();
            skip_ws();
            expect(')');
            return d;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            read_number();
            return Dimension{};  // dimensionless scalar
        }
        if (std::isalpha(static_cast<unsigned char>(c))) return read_symbol();
        fail("expected a unit symbol, number, or '('");
    }

    Dimension parse_term() {
        Dimension base = parse_factor();
        skip_ws();
        if (peek() == '^') {
            advance();
            const std::pair<int, int> e = parse_exponent();
            Dimension out;
            if (!base.pow_rational(e.first, e.second, out))
                fail("non-representable fractional exponent");
            return out;
        }
        return base;
    }

    Dimension parse_expr() {
        Dimension acc = parse_term();
        for (;;) {
            skip_ws();
            const char c = peek();
            if (c == '*') {
                advance();
                acc = acc + parse_term();
            } else if (c == '/') {
                advance();
                acc = acc - parse_term();
            } else {
                break;
            }
        }
        return acc;
    }
};

}  // namespace units_detail

inline Dimension parse_unit(const std::string& s) {
    units_detail::Parser p(units_detail::normalize_utf8(s), s);
    p.skip_ws();
    if (p.at_end()) p.fail("empty unit string");
    const Dimension d = p.parse_expr();
    p.skip_ws();
    if (!p.at_end()) p.fail("unexpected trailing characters");
    return d;
}

}  // namespace rsymbolic
