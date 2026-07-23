// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Macro operators (docs/57): body parsing, validation, and expansion.
//
// A macro is a single-argument template over the primitive operators, expanded into the
// tree when a growth mutation creates a unary node. These tests cover the two halves that
// must hold before the search can use one: the parser accepts exactly the bodies we intend
// (and rejects the rest with a message), and expansion produces a well-formed postfix tree
// of the expected size.

#include <cstdio>
#include <string>
#include <vector>

#include "rsymbolic/evolution/macro_op.hpp"
#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/random_tree.hpp"

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

using namespace rsymbolic;

MacroOp make(const std::string& body, int max_nodes = 30) {
    MacroOp m;
    std::string error;
    const bool ok = make_macro_op("f", body, "x", max_nodes, m, error);
    if (!ok) std::printf("  unexpected parse failure: %s\n", error.c_str());
    CHECK(ok);
    return m;
}

bool rejects(const std::string& body, std::string& error, int max_nodes = 30) {
    MacroOp m;
    return !make_macro_op("f", body, "x", max_nodes, m, error);
}

// A parsed body is a normal postfix tree; expanding it over a single variable gives back
// the expression the user wrote.
void test_parse_and_expand() {
    const MacroOp gauss = make("exp(neg(square(x)))");
    CHECK(is_valid_postfix(gauss.body));
    CHECK(gauss.body.size() == 4);       // x, square, neg, exp
    CHECK(macro_extra_nodes(gauss) == 3);
    CHECK(gauss.placeholder == 0);

    Tree arg = {variable_node(0)};
    Tree out = expand_macro(gauss, arg);
    CHECK(is_valid_postfix(out));
    CHECK(out.size() == 4);
    CHECK(to_string(out) == "exp(neg(square(x0)))");
}

// Infix operators, precedence and the placeholder in a non-leading position.
void test_infix_grammar() {
    const MacroOp logistic = make("1 / (1 + exp(neg(x)))");
    Tree out = expand_macro(logistic, Tree{variable_node(2)});
    CHECK(is_valid_postfix(out));
    CHECK(to_string(out) == "(1 / (1 + exp(neg(x2))))");
    CHECK(logistic.placeholder != 0);  // the argument is not the first postfix node here

    // Precedence: a + b * c parses as a + (b * c); ^ is right-associative.
    const MacroOp prec = make("1 + 2 * x");
    CHECK(to_string(expand_macro(prec, Tree{variable_node(0)})) == "(1 + (2 * x0))");
    const MacroOp powr = make("x ^ 2 ^ 3");
    CHECK(to_string(expand_macro(powr, Tree{variable_node(0)})) == "(x0 ^ (2 ^ 3))");

    // A literal negation folds into the constant: -2 is one node, not neg(2).
    const MacroOp negc = make("-2 * x");
    CHECK(negc.body.size() == 3);
    CHECK(to_string(expand_macro(negc, Tree{variable_node(0)})) == "(-2 * x0)");
}

// Expanding over a multi-node argument splices the whole subtree in.
void test_expand_multi_node_argument() {
    const MacroOp gauss = make("exp(neg(square(x)))");
    // argument: (x0 + 1.5)
    Tree arg = {variable_node(0), constant_node(0, 1.5), binary_node(BinaryOp::Add)};
    Tree out = expand_macro(gauss, arg);
    CHECK(is_valid_postfix(out));
    CHECK(out.size() == arg.size() + macro_extra_nodes(gauss));
    CHECK(to_string(out) == "exp(neg(square((x0 + 1.5))))");
}

// Numeric literals become ordinary tunable constants once the tree is re-indexed.
void test_constants_are_tunable() {
    const MacroOp damped = make("exp(-0.5 * x)");
    Tree out = expand_macro(damped, Tree{variable_node(0)});
    reindex_constants(out);
    CHECK(count_constants(out) == 1);
    CHECK(initial_constants(out)[0] == -0.5);
}

void test_validation_rejects() {
    std::string error;
    CHECK(rejects("exp(x) + x", error));           // two placeholders
    CHECK(error.find("exactly once") != std::string::npos);
    CHECK(rejects("exp(1)", error));               // no placeholder
    CHECK(rejects("gauss(x)", error));             // unknown function
    CHECK(error.find("unknown function") != std::string::npos);
    CHECK(rejects("exp(x", error));                // syntax error
    CHECK(rejects("", error));                     // empty body
    CHECK(rejects("x0 + x", error));               // data columns are not addressable
    CHECK(rejects("x", error));                    // identity macro: adds nothing
    CHECK(error.find("at least one operator") != std::string::npos);
    CHECK(rejects("add(x, 1)", error));            // binary in call form
    CHECK(error.find("infix") != std::string::npos);
    CHECK(rejects("exp(neg(square(x)))", error, /*max_nodes=*/3));  // cannot fit the cap
    CHECK(error.find("max_nodes") != std::string::npos);

    MacroOp m;
    CHECK(!make_macro_op("square", "exp(x)", "x", 30, m, error));  // shadows a built-in
    CHECK(error.find("shadows") != std::string::npos);
}

// The growth mutations must produce valid trees that still respect max_nodes when the
// unary alphabet contains a macro.
void test_growth_mutations_with_macro() {
    SearchSpace space;
    space.unary_ops = {};  // macro only: any unary node in the result came from the macro
    space.binary_ops = {BinaryOp::Add, BinaryOp::Mul};
    space.num_features = 1;
    space.max_nodes = 20;
    space.macro_ops = {make("exp(neg(square(x)))")};

    std::mt19937_64 rng(12345);
    bool saw_macro = false;
    for (int trial = 0; trial < 200; ++trial) {
        Tree tree = gen_random_tree(3, space, rng);
        CHECK(is_valid_postfix(tree));
        append_random_op(tree, space, rng);
        prepend_random_op(tree, space, rng);
        insert_random_op(tree, space, rng);
        CHECK(is_valid_postfix(tree));
        CHECK(static_cast<int>(tree.size()) <= space.max_nodes + 3);  // soft cap, as ever
        for (const Node& n : tree) {
            if (n.kind == NodeKind::Unary) saw_macro = true;
            CHECK(!is_macro_placeholder(n));  // placeholders never survive expansion
        }
    }
    CHECK(saw_macro);  // the macro is actually reachable through the growth mutations
}

// A macro too large to ever fit under max_nodes must be filtered out of the alphabet, and
// filtering must leave the RNG stream exactly as it was — same seed, same tree.
void test_unfittable_macro_is_inert() {
    SearchSpace base;
    base.unary_ops = {UnaryOp::Exp, UnaryOp::Sin};
    base.binary_ops = {BinaryOp::Add, BinaryOp::Mul};
    base.num_features = 2;
    base.max_nodes = 5;

    SearchSpace with_macro = base;
    // 7 body nodes: macro_extra_nodes = 6 > max_nodes, so it can never be offered.
    with_macro.macro_ops = {make("exp(neg(square(sin(x + 1))))", /*max_nodes=*/30)};
    CHECK(macro_extra_nodes(with_macro.macro_ops[0]) > base.max_nodes);

    std::mt19937_64 rng_a(99), rng_b(99);
    for (int trial = 0; trial < 50; ++trial) {
        Tree a = gen_random_tree(3, base, rng_a);
        Tree b = gen_random_tree(3, with_macro, rng_b);
        CHECK(to_string(a) == to_string(b));
        append_random_op(a, base, rng_a);
        append_random_op(b, with_macro, rng_b);
        insert_random_op(a, base, rng_a);
        insert_random_op(b, with_macro, rng_b);
        CHECK(to_string(a) == to_string(b));
    }
}

}  // namespace

int main() {
    test_parse_and_expand();
    test_infix_grammar();
    test_expand_multi_node_argument();
    test_constants_are_tunable();
    test_validation_rejects();
    test_growth_mutations_with_macro();
    test_unfittable_macro_is_inert();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
}
