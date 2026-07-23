// Known-answer tests for dimensional-analysis propagation and the loss penalty.
//
// The expected violates() outcomes are the physically-correct answers and mirror
// SymbolicRegression.jl's DimensionalAnalysis.jl. The two wildcard fine points (the
// result-wildcard of Pow and of transcendental unaries) are pinned here as the
// conservative reading; the Phase 0 SR.jl screen (docs/46) confirms them against the
// reference engine.

#include <cstdio>
#include <optional>
#include <vector>

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/units/dimensional_analysis.hpp"
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
using rsymbolic::DimAnalysis;
using rsymbolic::Node;
using rsymbolic::Tree;
using rsymbolic::add_dim_penalty;
using rsymbolic::parse_unit;
using rsymbolic::violates_dimensions;
using rsymbolic::BinaryOp;
using rsymbolic::UnaryOp;

std::vector<Dimension> units(std::initializer_list<const char*> us) {
    std::vector<Dimension> v;
    for (const char* u : us) v.push_back(parse_unit(u));
    return v;
}

// F = m * a : [x0, x1, *]  with x0=kg, x1=m/s^2, y=N  -> consistent.
void test_force_ma_consistent() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::variable_node(1),
              rsymbolic::binary_node(BinaryOp::Mul)};
    CHECK(!violates_dimensions(t, units({"kg", "m/s^2"}), parse_unit("N"), true));
    // y_units unset: still internally consistent.
    CHECK(!violates_dimensions(t, units({"kg", "m/s^2"}), std::nullopt, true));
    // wrong y_units (say kg) -> violates.
    CHECK(violates_dimensions(t, units({"kg", "m/s^2"}), parse_unit("kg"), true));
}

// F = m + a : [x0, x1, +]  adding kg to m/s^2 -> violates.
void test_force_ma_add_violates() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::variable_node(1),
              rsymbolic::binary_node(BinaryOp::Add)};
    CHECK(violates_dimensions(t, units({"kg", "m/s^2"}), std::nullopt, true));
}

// sin(x): [x0, sin]. Dimensioned x -> violates; dimensionless x -> ok.
void test_transcendental_requires_dimensionless() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::unary_node(UnaryOp::Sin)};
    CHECK(violates_dimensions(t, units({"m"}), std::nullopt, true));
    CHECK(!violates_dimensions(t, units({"1"}), std::nullopt, true));
    // sin of dimensioned x with y_units set is still a violation.
    CHECK(violates_dimensions(t, units({"m"}), parse_unit("1"), true));
}

// sqrt(square(x)) == x dimensionally: [x0, square, sqrt].
void test_sqrt_square_roundtrip() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::unary_node(UnaryOp::Square),
              rsymbolic::unary_node(UnaryOp::Sqrt)};
    CHECK(!violates_dimensions(t, units({"m"}), parse_unit("m"), true));
    // result is m, so y_units=s mismatches.
    CHECK(violates_dimensions(t, units({"m"}), parse_unit("s"), true));
}

// inv negates the exponents: 1/x0 with x0=s is a frequency (1/s = Hz).
void test_inv_inverts_dimension() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::unary_node(UnaryOp::Inv)};
    CHECK(!violates_dimensions(t, units({"s"}), parse_unit("Hz"), true));
    CHECK(violates_dimensions(t, units({"s"}), parse_unit("s"), true));
    // inv(inv(x)) is back to the original dimension.
    Tree t2 = {rsymbolic::variable_node(0), rsymbolic::unary_node(UnaryOp::Inv),
               rsymbolic::unary_node(UnaryOp::Inv)};
    CHECK(!violates_dimensions(t2, units({"s"}), parse_unit("s"), true));
}

// A bare variable with a y_units mismatch: [x0], x0=m, y=s -> violates.
void test_y_units_mismatch() {
    Tree t = {rsymbolic::variable_node(0)};
    CHECK(violates_dimensions(t, units({"m"}), parse_unit("s"), true));
    CHECK(!violates_dimensions(t, units({"m"}), parse_unit("m"), true));
}

// Wildcard unification: x + c  ([x0, c0, +]).
void test_wildcard_unification() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::constant_node(0, 1.0),
              rsymbolic::binary_node(BinaryOp::Add)};
    // allow_wildcards: the free constant adopts x's dimension -> consistent, root dim = m.
    CHECK(!violates_dimensions(t, units({"m"}), parse_unit("m"), true));
    // dimensionless_constants_only (allow_wildcards=false): c is forced dimensionless, so
    // adding it to a dimensioned x violates.
    CHECK(violates_dimensions(t, units({"m"}), std::nullopt, false));
}

// Pow with a dimensioned base violates (generic power requires dimensionless base);
// dimensionless base is fine. [x0, c0, ^].
void test_pow_requires_dimensionless_base() {
    Tree t = {rsymbolic::variable_node(0), rsymbolic::constant_node(0, 2.0),
              rsymbolic::binary_node(BinaryOp::Pow)};
    CHECK(violates_dimensions(t, units({"m"}), std::nullopt, true));    // m^c -> violates
    CHECK(!violates_dimensions(t, units({"1"}), std::nullopt, true));   // dimensionless^c ok
}

// The penalty helper: off-path is byte-identical; on-path adds the flat penalty exactly
// when the tree violates.
void test_penalty_helper() {
    Tree violating = {rsymbolic::variable_node(0), rsymbolic::variable_node(1),
                      rsymbolic::binary_node(BinaryOp::Add)};  // kg + m/s^2
    Tree clean = {rsymbolic::variable_node(0), rsymbolic::variable_node(1),
                  rsymbolic::binary_node(BinaryOp::Mul)};      // kg * m/s^2

    // Disabled context: no change whatsoever (units-off byte identity).
    DimAnalysis off;  // enabled = false
    CHECK(add_dim_penalty(3.14, violating, off) == 3.14);
    CHECK(add_dim_penalty(3.14, clean, off) == 3.14);

    // Enabled context.
    DimAnalysis on;
    on.enabled = true;
    on.x_units = units({"kg", "m/s^2"});
    on.y_units = std::nullopt;
    on.allow_wildcards = true;
    on.penalty_sse = 1000.0 * 42.0;  // penalty * W (n=42)

    CHECK(add_dim_penalty(2.0, violating, on) == 2.0 + 1000.0 * 42.0);
    CHECK(add_dim_penalty(2.0, clean, on) == 2.0);
    // Non-finite loss is passed through unchanged (kInf handling upstream).
    const double inf = 1.0 / 0.0;
    CHECK(add_dim_penalty(inf, violating, on) == inf);
}

}  // namespace

int main() {
    test_force_ma_consistent();
    test_force_ma_add_violates();
    test_transcendental_requires_dimensionless();
    test_sqrt_square_roundtrip();
    test_inv_inverts_dimension();
    test_y_units_mismatch();
    test_wildcard_unification();
    test_pow_requires_dimensionless_base();
    test_penalty_helper();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
