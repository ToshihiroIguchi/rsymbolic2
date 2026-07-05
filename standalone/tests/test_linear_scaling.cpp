// Opt-in Keijzer-2003 linear scaling tests (SearchOptions::linear_scaling).
//
// Testable-design note: the production scorer (sse_linear_scaled) is file-local in
// evolutionary_search.cpp, deliberately kept off the public surface. It is tested
// on two levels:
//   (a) the moment -> (a, b) closed-form step is replicated here and checked against
//       a brute-force 2x2 normal-equations solve on random data (unweighted AND
//       weighted, relative agreement 1e-12);
//   (b) the PRODUCTION path is exercised end-to-end through run_evolution using the
//       internal seed_trees hook with population_size = 1 and generations = 0: the
//       run then evaluates exactly the seeded tree with the scaled scorer and the
//       finalize pass materialises the fitted (a, b), so result.loss must match the
//       brute-force best-affine SSE computed independently in the test.
//
// Gates:
//   1. Closed form vs brute force (unweighted + weighted), rel 1e-12.
//   2. Degenerate: constant predictor -> a = 0, b = weighted mean(y), loss = weighted
//      SST; a non-finite prediction -> infinite loss.
//   3. Search-level recovery of y = 2.5*x + 1.7 with {add,sub,mul}: loss ~ 0, and for
//      the best member AND every Pareto member, member.loss equals the plain SSE of
//      the (materialised) tree recomputed point-by-point — self-consistency of the
//      finalize wrap.
//   4. Wrap-skip: a seeded exact candidate (y = x*x, f = x0*x0) gives a = 1, b = 0
//      exactly, so the returned tree is NOT wrapped (complexity unchanged).
//   5. OFF-path gate: flag absent vs explicitly false -> identical result (the wider
//      off-path regression coverage is the existing suites).

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

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

constexpr double kInf = std::numeric_limits<double>::infinity();

bool rel_close(double a, double b, double tol) {
    return std::abs(a - b) <= tol * (1.0 + std::max(std::abs(a), std::abs(b)));
}

// Replica of the production moment -> (a, b) closed form (sse_linear_scaled in
// evolutionary_search.cpp), including its degeneracy guard.
void closed_form_ab(const std::vector<double>& f, const std::vector<double>& y,
                    const std::vector<double>& w, double& a, double& b) {
    double Sw = 0.0, Swf = 0.0, Swy = 0.0, Swff = 0.0, Swfy = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double wi = w.empty() ? 1.0 : w[i];
        Sw   += wi;
        Swf  += wi * f[i];
        Swy  += wi * y[i];
        Swff += wi * f[i] * f[i];
        Swfy += wi * f[i] * y[i];
    }
    a = 0.0; b = 0.0;
    if (Sw > 0.0) {
        const double var_f = Swff - (Swf * Swf) / Sw;
        const double cov   = Swfy - (Swf * Swy) / Sw;
        a = cov / var_f;
        b = Swy / Sw - a * (Swf / Sw);
        if (!(var_f > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
            a = 0.0;
            b = Swy / Sw;
        }
    }
}

// Brute-force 2x2 normal-equations solve of min_{a,b} sum w (a*f + b - y)^2 by
// Cramer's rule — an independent derivation of the same optimum.
void brute_force_ab(const std::vector<double>& f, const std::vector<double>& y,
                    const std::vector<double>& w, double& a, double& b) {
    double Sw = 0.0, Swf = 0.0, Swy = 0.0, Swff = 0.0, Swfy = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double wi = w.empty() ? 1.0 : w[i];
        Sw   += wi;
        Swf  += wi * f[i];
        Swy  += wi * y[i];
        Swff += wi * f[i] * f[i];
        Swfy += wi * f[i] * y[i];
    }
    const double det = Swff * Sw - Swf * Swf;
    a = (Swfy * Sw - Swf * Swy) / det;
    b = (Swff * Swy - Swf * Swfy) / det;
}

double affine_sse(double a, double b, const std::vector<double>& f,
                  const std::vector<double>& y, const std::vector<double>& w) {
    double sse = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double wi = w.empty() ? 1.0 : w[i];
        const double r = a * f[i] + b - y[i];
        sse += wi * r * r;
    }
    return sse;
}

// Plain weighted SSE of a tree, computed point-by-point with the production scalar
// evaluate<double> — the same operations in the same order as sse_current, so the
// value is bit-identical to the loss the finalize pass writes (soa_eval.hpp's
// bit-exactness guarantee).
double plain_sse(const Tree& t, const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y, const std::vector<double>& w = {}) {
    const std::vector<double> c = initial_constants(t);
    std::vector<double> stack;
    double sse = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        double r = evaluate<double>(t, X[i].data(), c.data(), stack) - y[i];
        if (!w.empty()) r *= std::sqrt(w[i]);
        if (!std::isfinite(r)) return kInf;
        sse += r * r;
    }
    return sse;
}

// Run the PRODUCTION scaled scorer + finalize on exactly one candidate tree: the
// internal seed_trees hook with population_size = 1 and generations = 0 evaluates
// the seeded tree once (scaled) and materialises the fitted (a, b) into the result.
SearchResult probe(const Tree& seed_tree,
                   const std::vector<std::vector<double>>& X,
                   const std::vector<double>& y,
                   const std::vector<double>& w = {}) {
    SearchOptions opts;
    opts.space.num_features = static_cast<int>(X[0].size());
    opts.population_size    = 1;
    opts.generations        = 0;
    opts.n_populations      = 1;
    opts.seed               = 7;
    opts.linear_scaling     = true;
    opts.weights            = w;
    opts.seed_trees         = {seed_tree};
    return run_evolution(X, y, opts);
}

SearchOptions line_options() {
    SearchOptions o;
    o.space.binary_ops   = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    o.space.unary_ops    = {};
    o.space.num_features = 1;
    o.population_size    = 200;
    o.generations        = 40;
    o.tournament_size    = 4;
    o.seed               = 42;
    o.n_populations      = 1;
    o.target_loss        = 1e-10;
    return o;
}

// ---------------------------------------------------------------------------
// Gate 1: closed form vs brute-force normal equations, rel 1e-12.
// ---------------------------------------------------------------------------
void test_closed_form_vs_brute_force() {
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> uf(-3.0, 3.0);
    std::uniform_real_distribution<double> uw(0.1, 3.0);
    for (int trial = 0; trial < 20; ++trial) {
        const std::size_t n = 40;
        std::vector<double> f(n), y(n), w(n);
        for (std::size_t i = 0; i < n; ++i) {
            f[i] = uf(rng);
            y[i] = 1.9 * f[i] - 0.7 + 0.3 * uf(rng);  // affine + noise
            w[i] = uw(rng);
        }
        for (const bool weighted : {false, true}) {
            const std::vector<double> wv = weighted ? w : std::vector<double>{};
            double a1, b1, a2, b2;
            closed_form_ab(f, y, wv, a1, b1);
            brute_force_ab(f, y, wv, a2, b2);
            CHECK(rel_close(a1, a2, 1e-12));
            CHECK(rel_close(b1, b2, 1e-12));
            CHECK(rel_close(affine_sse(a1, b1, f, y, wv),
                            affine_sse(a2, b2, f, y, wv), 1e-12));
        }
    }
}

// ---------------------------------------------------------------------------
// Gate 1b: the PRODUCTION scorer agrees with the brute-force optimum (probe on a
// nonlinear candidate f = x*x + x under y = 3.7*f + 0.9 + noise).
// ---------------------------------------------------------------------------
void test_production_scorer_matches_brute_force() {
    std::mt19937_64 rng(456);
    std::uniform_real_distribution<double> noise(-0.5, 0.5);
    const std::size_t n = 50;
    std::vector<std::vector<double>> X;
    std::vector<double> y(n), f(n), w(n);
    std::uniform_real_distribution<double> uw(0.2, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = -2.0 + 4.0 * static_cast<double>(i) / (n - 1);
        X.push_back({x});
        f[i] = x * x + x;
        y[i] = 3.7 * f[i] + 0.9 + noise(rng);
        w[i] = uw(rng);
    }
    // f = (x0 * x0) + x0 in postfix.
    const Tree tree{variable_node(0), variable_node(0), binary_node(BinaryOp::Mul),
                    variable_node(0), binary_node(BinaryOp::Add)};
    for (const bool weighted : {false, true}) {
        const std::vector<double> wv = weighted ? w : std::vector<double>{};
        const SearchResult r = probe(tree, X, y, wv);
        double a, b;
        brute_force_ab(f, y, wv, a, b);
        const double best = affine_sse(a, b, f, y, wv);
        std::printf("production %sweighted: loss=%.12e  brute=%.12e  expr=%s\n",
                    weighted ? "" : "un", r.loss, best, r.expression.c_str());
        CHECK(rel_close(r.loss, best, 1e-9));
        // The materialised wrap: constants 3.7-ish and 0.9-ish are in the tree.
        CHECK(r.expression.find("*") != std::string::npos);
        // Self-consistency: the reported loss is the plain SSE of the reported tree.
        CHECK(rel_close(r.loss, plain_sse(r.tree, X, y, wv), 1e-12));
    }
}

// ---------------------------------------------------------------------------
// Gate 2: degenerate cases.
// ---------------------------------------------------------------------------
void test_degenerate_constant_and_nonfinite() {
    const std::size_t n = 24;
    std::vector<std::vector<double>> X;
    std::vector<double> y(n), w(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = 0.5 * static_cast<double>(i) - 4.0;
        X.push_back({x});
        y[i] = std::sin(0.7 * x) + 0.1 * x;   // non-constant target
        w[i] = 0.5 + 0.1 * static_cast<double>(i % 5);
    }
    // Constant predictor f = 1.0: var_f collapses, so the guard yields a = 0,
    // b = (weighted) mean(y), and the loss is the (weighted) SST. f = 1.0 is chosen
    // so the moment cancellations are numerically benign in both weightings.
    const Tree const_tree{constant_node(0, 1.0)};
    for (const bool weighted : {false, true}) {
        const std::vector<double> wv = weighted ? w : std::vector<double>{};
        double Sw = 0.0, Swy = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double wi = wv.empty() ? 1.0 : wv[i];
            Sw += wi; Swy += wi * y[i];
        }
        const double mean = Swy / Sw;
        double sst = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double wi = wv.empty() ? 1.0 : wv[i];
            sst += wi * (y[i] - mean) * (y[i] - mean);
        }
        const SearchResult r = probe(const_tree, X, y, wv);
        std::printf("degenerate %sweighted: loss=%.12e  sst=%.12e  expr=%s\n",
                    weighted ? "" : "un", r.loss, sst, r.expression.c_str());
        CHECK(rel_close(r.loss, sst, 1e-12));
        // The wrap materialises b = mean(y): the reported tree predicts the mean.
        CHECK(rel_close(r.loss, plain_sse(r.tree, X, y, wv), 1e-12));
    }
    // Non-finite prediction: log of a negative input is NaN at every point, so the
    // scaled scorer returns an infinite loss (matching sse_current's rejection).
    std::vector<std::vector<double>> Xneg;
    std::vector<double> yneg;
    for (int i = 0; i < 10; ++i) {
        Xneg.push_back({-1.0 - i});
        yneg.push_back(static_cast<double>(i));
    }
    const Tree log_tree{variable_node(0), unary_node(UnaryOp::Log)};
    const SearchResult r_inf = probe(log_tree, Xneg, yneg);
    CHECK(std::isinf(r_inf.loss));
}

// ---------------------------------------------------------------------------
// Gate 3: search-level recovery + full self-consistency of the materialised wrap.
// ---------------------------------------------------------------------------
void test_search_recovery_and_self_consistency() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 40; ++i) {
        const double x = -5.0 + 10.0 * static_cast<double>(i) / 39.0;
        X.push_back({x});
        y.push_back(2.5 * x + 1.7);
    }
    SearchOptions opts = line_options();
    opts.linear_scaling = true;
    const SearchResult r = run_evolution(X, y, opts);
    std::printf("recovery: loss=%.3e  complexity=%d  expr=%s\n",
                r.loss, r.complexity, r.expression.c_str());
    CHECK(r.loss < 1e-6);
    CHECK(!r.pareto_front.empty());
    // member.loss == plain SSE of the materialised tree, EXACTLY (the finalize pass
    // recomputes the loss with sse_current, which is bit-identical to this scalar
    // recomputation), for the best member and every Pareto member.
    CHECK(r.loss == plain_sse(r.tree, X, y));
    for (const PopMember& m : r.pareto_front) {
        CHECK(m.loss == plain_sse(m.tree, X, y));
        CHECK(m.complexity == static_cast<int>(m.tree.size()));
    }
}

// ---------------------------------------------------------------------------
// Gate 4: wrap-skip when (a, b) is exactly the identity.
// ---------------------------------------------------------------------------
void test_wrap_skip_on_identity() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 25; ++i) {
        const double x = -2.0 + 4.0 * static_cast<double>(i) / 24.0;
        X.push_back({x});
        y.push_back(x * x);  // y = x*x exactly
    }
    // Seed the exact candidate f = x0 * x0: its predictions equal y bitwise, so the
    // closed form gives a = 1, b = 0 exactly and the finalize must NOT wrap it.
    const Tree exact{variable_node(0), variable_node(0), binary_node(BinaryOp::Mul)};
    const SearchResult r = probe(exact, X, y);
    std::printf("wrap-skip: loss=%.3e  complexity=%d  expr=%s\n",
                r.loss, r.complexity, r.expression.c_str());
    CHECK(r.loss == 0.0);
    CHECK(r.complexity == 3);                 // unchanged: no +4-node affine wrap
    CHECK(r.expression == "(x0 * x0)");
}

// ---------------------------------------------------------------------------
// Gate 5: OFF path — flag absent vs explicitly false is identical.
// ---------------------------------------------------------------------------
void test_off_path_identity() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    for (int i = 0; i < 20; ++i) {
        const double x = static_cast<double>(i) - 10.0;
        X.push_back({x});
        y.push_back(2.5 * x + 1.7);
    }
    const SearchOptions absent = line_options();  // linear_scaling left at default
    SearchOptions explicit_off = line_options();
    explicit_off.linear_scaling = false;
    const SearchResult r1 = run_evolution(X, y, absent);
    const SearchResult r2 = run_evolution(X, y, explicit_off);
    CHECK(r1.expression == r2.expression);
    CHECK(r1.loss == r2.loss);
    CHECK(r1.n_evals == r2.n_evals);
}

}  // namespace

int main() {
    test_closed_form_vs_brute_force();
    test_production_scorer_matches_brute_force();
    test_degenerate_constant_and_nonfinite();
    test_search_recovery_and_self_consistency();
    test_wrap_skip_on_identity();
    test_off_path_identity();

    if (g_failures == 0) {
        std::printf("All %d linear-scaling checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d linear-scaling checks FAILED\n", g_failures, g_checks);
    return 1;
}
