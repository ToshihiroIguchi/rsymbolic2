// Unit tests for parse_unit() (DynamicQuantities-style unit string -> Dimension).

#include <cstdio>
#include <stdexcept>
#include <string>

#include "rsymbolic/units/unit_parser.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s (%s:%d)\n", expr, file, line);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

using rsymbolic::Dimension;
using rsymbolic::parse_unit;
using rsymbolic::units_detail::make_dim;

bool throws(const std::string& s) {
    try {
        parse_unit(s);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_base_units() {
    CHECK(parse_unit("m") == make_dim(0, 1, 0, 0, 0, 0, 0));
    CHECK(parse_unit("kg") == make_dim(1, 0, 0, 0, 0, 0, 0));
    CHECK(parse_unit("s") == make_dim(0, 0, 1, 0, 0, 0, 0));
    CHECK(parse_unit("cd") == make_dim(0, 0, 0, 0, 0, 0, 1));  // candela, not centi-day
    CHECK(parse_unit("mol") == make_dim(0, 0, 0, 0, 0, 1, 0));
}

void test_dimensionless() {
    CHECK(parse_unit("1").dimensionless());
    CHECK(parse_unit("rad").dimensionless());
    CHECK(parse_unit("2.5").dimensionless());
}

void test_arithmetic() {
    // Joule = kg*m^2/s^2
    CHECK(parse_unit("kg*m^2/s^2") == make_dim(1, 2, -2, 0, 0, 0, 0));
    CHECK(parse_unit("kg*m^2/s^2") == parse_unit("J"));
    // acceleration
    CHECK(parse_unit("m/s^2") == make_dim(0, 1, -2, 0, 0, 0, 0));
    // Newton*meter = Joule
    CHECK(parse_unit("N*m") == parse_unit("J"));
    // negative exponent
    CHECK(parse_unit("s^-1") == parse_unit("Hz"));
    // parentheses + power: (m/s)^2 = m^2/s^2
    CHECK(parse_unit("(m/s)^2") == make_dim(0, 2, -2, 0, 0, 0, 0));
}

void test_rational_exponent() {
    Dimension half = parse_unit("m^(1/2)");
    CHECK(half.num[1] == Dimension::DEN / 2);
    CHECK(half.scaled(2) == parse_unit("m"));
    // Julia '//' rational syntax
    CHECK(parse_unit("m^(1//2)") == half);
}

void test_prefixes_inert() {
    CHECK(parse_unit("km") == parse_unit("m"));   // kilo stripped
    CHECK(parse_unit("ms") == parse_unit("s"));   // milli + second
    CHECK(parse_unit("mg") == parse_unit("kg"));  // milligram: still mass
    CHECK(parse_unit("dam") == parse_unit("m"));  // deca (two-char prefix) + meter
    CHECK(parse_unit("THz") == parse_unit("Hz"));  // tera + hertz (T is also Tesla-unit)
    CHECK(parse_unit("T") == make_dim(1, 0, -2, -1, 0, 0, 0));  // bare T = Tesla, not tera
}

void test_utf8_aliases() {
    CHECK(parse_unit("\xCE\xA9") == parse_unit("Ohm"));  // Ω
    CHECK(parse_unit("\xC2\xB5m") == parse_unit("m"));   // µm (micrometre)
}

void test_errors() {
    CHECK(throws(""));
    CHECK(throws("   "));
    CHECK(throws("foo"));         // unknown symbol
    CHECK(throws("m/"));          // dangling operator
    CHECK(throws("m^"));          // missing exponent
    CHECK(throws("m^(1/0)"));     // zero denominator
    CHECK(throws("(m/s"));        // unbalanced paren
    CHECK(throws("m s"));         // implicit product not allowed
    CHECK(throws("\xC2\xB0" "C"));  // °C offset unit
    CHECK(throws("h"));           // hour not supported (h is only the hecto prefix)
}

}  // namespace

int main() {
    test_base_units();
    test_dimensionless();
    test_arithmetic();
    test_rational_exponent();
    test_prefixes_inert();
    test_utf8_aliases();
    test_errors();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
