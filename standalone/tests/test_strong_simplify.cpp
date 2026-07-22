// Opt-in search-time strong simplification tests (SearchOptions::strong_simplify;
// docs/55). The option offers each population member's weak-simplified tree to the
// display-only two-layer simplifier (display_simplify(), docs/54) during evolution,
// under a small deterministic budget, adopting the rewrite only when it is strictly
// smaller AND every operator it introduces is already enabled in the search's
// operator set. See evolutionary_search.cpp's optimize_and_simplify_population for
// the hook and ops_within_search_space for the gate.
//
// Gates:
//   1. OFF-path bit-identity: default-constructed options vs explicit
//      strong_simplify=false give identical expression/loss/Pareto front/n_evals,
//      and both leave the telemetry counters at 0.
//   2. ON-path determinism: same seed, ON, run twice => byte-identical results.
//   3. Thread-count determinism: ON, n_threads=1 vs >1 (n_populations>1) => identical.
//   4. Recovery sanity on a known-answer problem (Nguyen-1): ON completes with a
//      finite, small loss. NOT a bit-identity gate vs OFF — class-B rewrites
//      (reassociation/redistribution) may shift the loss by a rounding step (docs/54
//      FP policy), so only a sanity threshold is checked.
//   5. Telemetry: OFF => both counters 0; ON (rich operator set incl. neg/square/abs,
//      a square(x)-shaped target, enough generations) => attempts > 0, attempts >=
//      adopted, and adopted > 0 (the option actually does something under this config).
//   6. Operator-set gate: an ON run whose operator set has NO unary operators enabled
//      must never produce a result (or Pareto member) containing any unary node — the
//      only channel for one to appear is a strong_simplify rewrite (neg/square/abs),
//      since the search's own generators/mutations never emit a disabled operator.
//      Also a direct unit test of a replica of the production membership helper
//      (ops_within_search_space is file-local in evolutionary_search.cpp; replicated
//      here for direct testing, following the test_linear_scaling.cpp precedent for
//      file-local production helpers).
//   7. Rejected-candidate-untouched: not separately tested — the hook only assigns to
//      `s` on adoption (see the source), so an untouched `s` on rejection is provable
//      by inspection; gates 1 (OFF path unaffected) and 6 (a rejected rewrite never
//      reaches the final result/Pareto front) exercise it indirectly.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

namespace {

int g_failures = 0;
int g_checks   = 0;

void check(bool cond, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("FAIL: %s  (%s:%d)\n", expr, file, line);
    }
}
#define CHECK(c) check((c), #c, __FILE__, __LINE__)

using namespace rsymbolic;

// Build search options for the linear problem with given n_populations (mirrors the
// linear_opts helper in test_eval_cache.cpp / test_island_model.cpp).
SearchOptions linear_opts(std::size_t n_pops) {
    const auto prob = problem_linear();
    SearchOptions opts;
    opts.space                = prob.space;
    opts.population_size      = 200;
    opts.generations          = 40;
    opts.tournament_size      = 4;
    opts.target_loss          = 1e-10;
    opts.simplify_expressions = true;
    opts.seed                 = 42;
    opts.n_populations        = n_pops;
    return opts;
}

// Bit-identity comparison of two search results (mirrors test_eval_cache.cpp /
// test_linear_scaling.cpp): expression string, bitwise loss, Pareto front (size and
// per-member losses/complexities), and eval accounting.
void check_bit_identical(const SearchResult& a, const SearchResult& b) {
    CHECK(a.expression == b.expression);
    CHECK(std::memcmp(&a.loss, &b.loss, sizeof(double)) == 0);
    CHECK(a.pareto_front.size() == b.pareto_front.size());
    if (a.pareto_front.size() == b.pareto_front.size()) {
        for (std::size_t i = 0; i < a.pareto_front.size(); ++i) {
            CHECK(std::memcmp(&a.pareto_front[i].loss, &b.pareto_front[i].loss,
                              sizeof(double)) == 0);
            CHECK(a.pareto_front[i].complexity == b.pareto_front[i].complexity);
        }
    }
    CHECK(a.n_evals == b.n_evals);
    CHECK(a.n_forward_evals == b.n_forward_evals);
}

// True if `tree` contains any Unary node at all.
bool has_unary_node(const Tree& tree) {
    for (const Node& n : tree) {
        if (n.kind == NodeKind::Unary) return true;
    }
    return false;
}

// Replica of the production ops_within_search_space (file-local in
// evolutionary_search.cpp, deliberately kept off the public surface — same rationale
// as test_linear_scaling.cpp's closed_form_ab replica): every unary/binary operator
// node in `tree` must be enabled in `space`'s operator set.
bool ops_within_search_space_replica(const Tree& tree, const SearchSpace& space) {
    for (const Node& node : tree) {
        if (node.kind == NodeKind::Unary) {
            const bool ok = std::find(space.unary_ops.begin(), space.unary_ops.end(),
                                      node.uop) != space.unary_ops.end();
            if (!ok) return false;
        } else if (node.kind == NodeKind::Binary) {
            const bool ok = std::find(space.binary_ops.begin(), space.binary_ops.end(),
                                      node.bop) != space.binary_ops.end();
            if (!ok) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Gate 1: OFF-path bit-identity — flag absent vs explicitly false.
// ---------------------------------------------------------------------------
void test_off_path_bit_identity() {
    SearchOptions absent = linear_opts(4);  // strong_simplify left at default (false)
    SearchOptions explicit_off = linear_opts(4);
    explicit_off.strong_simplify = false;
    const auto prob = problem_linear();
    const SearchResult r1 = run_evolution(prob.X, prob.y, absent);
    const SearchResult r2 = run_evolution(prob.X, prob.y, explicit_off);
    check_bit_identical(r1, r2);
    CHECK(r1.n_strong_simplify_attempts == 0 && r1.n_strong_simplify_adopted == 0);
    CHECK(r2.n_strong_simplify_attempts == 0 && r2.n_strong_simplify_adopted == 0);
}

// ---------------------------------------------------------------------------
// Gate 2: ON-path determinism — same seed, ON, run twice => byte-identical.
// ---------------------------------------------------------------------------
void test_on_path_determinism() {
    SearchOptions on = linear_opts(4);
    on.strong_simplify = true;
    const auto prob = problem_linear();
    const SearchResult r1 = run_evolution(prob.X, prob.y, on);
    const SearchResult r2 = run_evolution(prob.X, prob.y, on);
    check_bit_identical(r1, r2);
    CHECK(r1.n_strong_simplify_attempts == r2.n_strong_simplify_attempts);
    CHECK(r1.n_strong_simplify_adopted == r2.n_strong_simplify_adopted);
}

// ---------------------------------------------------------------------------
// Gate 3: thread-count determinism — ON, n_threads=1 vs >1, n_populations>1.
// ---------------------------------------------------------------------------
void test_thread_count_determinism() {
    SearchOptions on = linear_opts(4);
    on.strong_simplify = true;
    const auto prob = problem_linear();
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const SearchResult r1 = run_evolution(prob.X, prob.y, on);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const SearchResult r2 = run_evolution(prob.X, prob.y, on);
    std::printf("thread-1 expr: %s\nthread-4 expr: %s\n",
                r1.expression.c_str(), r2.expression.c_str());
    CHECK(r1.expression == r2.expression);
    check_bit_identical(r1, r2);
    CHECK(r1.n_strong_simplify_attempts == r2.n_strong_simplify_attempts);
    CHECK(r1.n_strong_simplify_adopted == r2.n_strong_simplify_adopted);
}

// ---------------------------------------------------------------------------
// Gate 4: recovery sanity on a known-answer problem (Nguyen-1: x^3+x^2+x).
// ---------------------------------------------------------------------------
void test_recovery_sanity() {
    const auto prob = problem_nguyen1();
    SearchOptions on;
    on.space                = prob.space;
    on.population_size      = 200;
    on.generations           = 150;
    on.tournament_size      = 10;
    on.target_loss          = 1e-8;
    on.simplify_expressions = true;
    on.strong_simplify      = true;
    on.seed                 = 7;
    on.n_populations        = 4;
    const SearchResult r = run_evolution(prob.X, prob.y, on);
    std::printf("recovery: loss=%.3e  complexity=%d  expr=%s  attempts=%llu adopted=%llu\n",
                r.loss, r.complexity, r.expression.c_str(),
                static_cast<unsigned long long>(r.n_strong_simplify_attempts),
                static_cast<unsigned long long>(r.n_strong_simplify_adopted));
    CHECK(std::isfinite(r.loss));
    // Sanity recovery threshold on held-out/extrapolation points, looser than the
    // problem's strict recovery_threshold (docs/54's FP-drift caveat), since this
    // gate is about "the option does not break recovery", not exact reproduction.
    double max_err = 0.0;
    for (std::size_t i = 0; i < prob.X_test.size(); ++i) {
        const std::vector<double> c = initial_constants(r.tree);
        const double pred = evaluate<double>(r.tree, prob.X_test[i].data(), c.data());
        if (!std::isfinite(pred)) { max_err = std::numeric_limits<double>::infinity(); break; }
        max_err = std::max(max_err, std::abs(pred - prob.y_test[i]));
    }
    std::printf("recovery: max extrapolation error = %.3e (threshold %.3e)\n",
                max_err, prob.recovery_threshold * 100.0);
    CHECK(max_err < prob.recovery_threshold * 100.0);
}

// ---------------------------------------------------------------------------
// Gate 5: telemetry — OFF is all-zero; ON (rich ops, square(x) target) fires and
// adopts.
// ---------------------------------------------------------------------------
SearchOptions square_target_opts() {
    SearchOptions o;
    o.space.binary_ops   = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    o.space.unary_ops    = {UnaryOp::Neg, UnaryOp::Square, UnaryOp::Abs};
    o.space.num_features = 1;
    o.space.max_depth    = 6;
    o.space.max_nodes    = 30;
    o.population_size    = 150;
    o.generations        = 150;
    o.tournament_size    = 10;
    o.n_populations      = 2;
    o.seed               = 99;
    o.simplify_expressions = true;
    return o;
}

void test_telemetry() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 30; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / 29.0;
        X.push_back({x});
        y.push_back(x * x);  // a natural square(x) collection target
    }

    const SearchOptions off = square_target_opts();
    const SearchResult r_off = run_evolution(X, y, off);
    CHECK(r_off.n_strong_simplify_attempts == 0);
    CHECK(r_off.n_strong_simplify_adopted == 0);

    SearchOptions on = square_target_opts();
    on.strong_simplify = true;
    const SearchResult r_on = run_evolution(X, y, on);
    std::printf("telemetry: attempts=%llu  adopted=%llu  expr=%s\n",
                static_cast<unsigned long long>(r_on.n_strong_simplify_attempts),
                static_cast<unsigned long long>(r_on.n_strong_simplify_adopted),
                r_on.expression.c_str());
    CHECK(r_on.n_strong_simplify_attempts > 0);
    CHECK(r_on.n_strong_simplify_attempts >= r_on.n_strong_simplify_adopted);
    CHECK(r_on.n_strong_simplify_adopted > 0);
}

// ---------------------------------------------------------------------------
// Gate 6: operator-set gate — search-level, plus a direct unit test of the
// (replica) membership helper.
// ---------------------------------------------------------------------------
void test_operator_set_gate() {
    // Direct unit test of the membership helper.
    {
        SearchSpace space;
        space.binary_ops = {BinaryOp::Add};
        space.unary_ops  = {};
        const Tree neg_tree{variable_node(0), unary_node(UnaryOp::Neg)};
        CHECK(!ops_within_search_space_replica(neg_tree, space));
        space.unary_ops = {UnaryOp::Neg};
        CHECK(ops_within_search_space_replica(neg_tree, space));

        // Binary membership too: a tree using Mul must fail when only Add is enabled.
        const Tree mul_tree{variable_node(0), variable_node(0), binary_node(BinaryOp::Mul)};
        SearchSpace add_only;
        add_only.binary_ops = {BinaryOp::Add};
        CHECK(!ops_within_search_space_replica(mul_tree, add_only));
        add_only.binary_ops = {BinaryOp::Add, BinaryOp::Mul};
        CHECK(ops_within_search_space_replica(mul_tree, add_only));
    }

    // Search-level gate: NO unary operators enabled at all, target = x*x (the natural
    // square(x)-collection trigger). Without the gate, display_simplify would happily
    // fold a member's x*x into square(x) and it would leak into the result/Pareto
    // front; the gate must reject that rewrite every time, so no unary node should
    // ever appear (the search's own generators/mutations never emit a disabled op —
    // the ONLY channel for one to appear here is an ungated strong_simplify rewrite).
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 30; ++i) {
        const double x = -3.0 + 6.0 * static_cast<double>(i) / 29.0;
        X.push_back({x});
        y.push_back(x * x);
    }
    SearchOptions on;
    on.space.binary_ops      = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    on.space.unary_ops       = {};  // no unary ops enabled
    on.space.num_features    = 1;
    on.space.max_depth       = 6;
    on.space.max_nodes       = 30;
    on.population_size       = 150;
    on.generations           = 150;
    on.tournament_size       = 10;
    on.n_populations         = 2;
    on.seed                  = 99;
    on.simplify_expressions  = true;
    on.strong_simplify       = true;
    const SearchResult r = run_evolution(X, y, on);
    std::printf("operator-set gate: attempts=%llu  adopted=%llu  expr=%s\n",
                static_cast<unsigned long long>(r.n_strong_simplify_attempts),
                static_cast<unsigned long long>(r.n_strong_simplify_adopted),
                r.expression.c_str());
    CHECK(!has_unary_node(r.tree));
    for (const PopMember& m : r.pareto_front) {
        CHECK(!has_unary_node(m.tree));
    }
}

}  // namespace

int main() {
    test_off_path_bit_identity();
    test_on_path_determinism();
    test_thread_count_determinism();
    test_recovery_sanity();
    test_telemetry();
    test_operator_set_gate();

    if (g_failures == 0) {
        std::printf("All %d strong-simplify checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d strong-simplify checks FAILED\n", g_failures, g_checks);
    return 1;
}
