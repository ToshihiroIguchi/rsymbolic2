// Unit tests for MultiDual<N> (batched / vector-mode forward-mode AD) and the tiled
// Jacobian it drives in make_least_squares_problem (docs/30).
//
// The central guarantee is that the batched Jacobian is BIT-IDENTICAL to the original
// scalar k-pass Jacobian: same value path, same per-direction derivative formula, just
// computed N directions at a time. Bit-identity means the whole search is unchanged
// (only faster), so we assert it strictly (raw bit patterns, not a tolerance):
//   - the production closure (uses kJacobianBlockWidth) vs the scalar Dual reference;
//   - a local tiled implementation at N = 4, 8, 16 vs the same reference
//     (certifies the result is independent of the block width).
// A finite-difference check on smooth configurations guards the derivative formulas
// themselves, independent of the scalar reference.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/multi_dual.hpp"
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"

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

using rsymbolic::BinaryOp;
using rsymbolic::Dual;
using rsymbolic::MultiDual;
using rsymbolic::Node;
using rsymbolic::Tree;
using rsymbolic::UnaryOp;
using rsymbolic::binary_node;
using rsymbolic::constant_node;
using rsymbolic::evaluate;
using rsymbolic::make_least_squares_problem;
using rsymbolic::unary_node;
using rsymbolic::variable_node;

// Bit-exact equality of two doubles (treats NaN/inf/signed-zero by their bit pattern,
// which is the strict "bit-identical" property we are asserting).
bool same_bits(double a, double b) {
    std::uint64_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

// Reference Jacobian: the original scalar k-pass forward-mode loop (one constant
// direction per pass). jacobian[i*k + kk] = d prediction_i / d c_kk.
std::vector<double> reference_jacobian(const Tree& tree,
                                       const std::vector<std::vector<double>>& X,
                                       const std::vector<double>& params) {
    const int k = static_cast<int>(params.size());
    const std::size_t m = X.size();
    std::vector<double> jac(m * static_cast<std::size_t>(k), 0.0);
    std::vector<Dual> c(static_cast<std::size_t>(k));
    std::vector<Dual> stack;
    for (int kk = 0; kk < k; ++kk) {
        for (int j = 0; j < k; ++j)
            c[static_cast<std::size_t>(j)] =
                Dual(params[static_cast<std::size_t>(j)], j == kk ? 1.0 : 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            const Dual p = evaluate<Dual>(tree, X[i].data(), c.data(), stack);
            jac[i * static_cast<std::size_t>(k) + static_cast<std::size_t>(kk)] =
                p.deriv;
        }
    }
    return jac;
}

// Tiled batched Jacobian at an arbitrary block width N (mirrors the production loop in
// least_squares_problem.hpp). Used to certify the result is independent of N.
template <int N>
std::vector<double> tiled_jacobian(const Tree& tree,
                                   const std::vector<std::vector<double>>& X,
                                   const std::vector<double>& params) {
    const int k = static_cast<int>(params.size());
    const std::size_t m = X.size();
    std::vector<double> jac(m * static_cast<std::size_t>(k), 0.0);
    std::vector<MultiDual<N>> c(static_cast<std::size_t>(k));
    std::vector<MultiDual<N>> stack;
    for (int block = 0; block < k; block += N) {
        const int w = std::min(N, k - block);
        for (int j = 0; j < k; ++j) {
            MultiDual<N>& cj = c[static_cast<std::size_t>(j)];
            cj.value = params[static_cast<std::size_t>(j)];
            cj.grad.fill(0.0);
        }
        for (int cc = 0; cc < w; ++cc)
            c[static_cast<std::size_t>(block + cc)].grad[static_cast<std::size_t>(cc)] =
                1.0;
        for (std::size_t i = 0; i < m; ++i) {
            const MultiDual<N> p =
                evaluate<MultiDual<N>>(tree, X[i].data(), c.data(), stack);
            for (int cc = 0; cc < w; ++cc)
                jac[i * static_cast<std::size_t>(k) +
                    static_cast<std::size_t>(block + cc)] =
                    p.grad[static_cast<std::size_t>(cc)];
        }
    }
    return jac;
}

// Production Jacobian via the actual closure under test (uses kJacobianBlockWidth).
std::vector<double> production_jacobian(const Tree& tree,
                                        const std::vector<std::vector<double>>& X,
                                        const std::vector<double>& params) {
    const std::vector<double> y(X.size(), 0.0);  // derivatives do not depend on y
    auto problem = make_least_squares_problem(tree, X, y, params);
    std::vector<double> jac(X.size() * params.size(), 0.0);
    problem.jacobian(params, jac);
    return jac;
}

// Assert every Jacobian entry is bit-identical between the batched implementations and
// the scalar reference. Optionally also run a finite-difference sanity check (only on
// smooth configurations, where the operators stay on their standard branches).
void run_case(const char* label, const Tree& tree,
              const std::vector<std::vector<double>>& X,
              const std::vector<double>& params, bool finite_diff) {
    const int k = static_cast<int>(params.size());
    const std::size_t m = X.size();

    const std::vector<double> ref = reference_jacobian(tree, X, params);
    const std::vector<double> prod = production_jacobian(tree, X, params);
    const std::vector<double> t4 = tiled_jacobian<4>(tree, X, params);
    const std::vector<double> t8 = tiled_jacobian<8>(tree, X, params);
    const std::vector<double> t16 = tiled_jacobian<16>(tree, X, params);

    bool prod_ok = (prod.size() == ref.size());
    bool n_ok = true;
    for (std::size_t e = 0; e < ref.size() && prod_ok; ++e) {
        if (!same_bits(prod[e], ref[e])) prod_ok = false;
        if (!same_bits(t4[e], ref[e]) || !same_bits(t8[e], ref[e]) ||
            !same_bits(t16[e], ref[e]))
            n_ok = false;
    }
    if (!prod_ok) std::printf("  [%s] production Jacobian differs from scalar\n", label);
    if (!n_ok) std::printf("  [%s] tiled N=4/8/16 differ from scalar\n", label);
    CHECK(prod_ok);
    CHECK(n_ok);

    if (!finite_diff) return;

    // Centered finite differences on the plain-double evaluator.
    const double h = 1e-6;
    bool fd_ok = true;
    std::vector<double> pp = params;
    for (int j = 0; j < k && fd_ok; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            pp = params;
            pp[static_cast<std::size_t>(j)] += h;
            const double fp = evaluate<double>(tree, X[i].data(), pp.data());
            pp[static_cast<std::size_t>(j)] = params[static_cast<std::size_t>(j)] - h;
            const double fm = evaluate<double>(tree, X[i].data(), pp.data());
            const double fd = (fp - fm) / (2.0 * h);
            const double an =
                ref[i * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
            if (std::fabs(fd - an) > 1e-5 * (1.0 + std::fabs(an))) {
                std::printf("  [%s] FD mismatch j=%d i=%zu fd=%.6g an=%.6g\n",
                            label, j, i, fd, an);
                fd_ok = false;
                break;
            }
        }
    }
    CHECK(fd_ok);
}

// ((c0 * x0) + c1)  — k=2, standard arithmetic.
void test_linear() {
    Tree t = {constant_node(0, 0.7), variable_node(0), binary_node(BinaryOp::Mul),
              constant_node(1, -0.3), binary_node(BinaryOp::Add)};
    std::vector<std::vector<double>> X = {{1.5}, {-2.0}, {0.3}, {4.1}};
    run_case("linear", t, X, {0.7, -0.3}, true);
}

// (sin(c0 * x0) * sqrt(c1 + x1))  — k=2, mixes sin/sqrt/mul/add.
void test_sin_sqrt() {
    Tree t = {constant_node(0, 1.1), variable_node(0), binary_node(BinaryOp::Mul),
              unary_node(UnaryOp::Sin), constant_node(1, 0.9), variable_node(1),
              binary_node(BinaryOp::Add), unary_node(UnaryOp::Sqrt),
              binary_node(BinaryOp::Mul)};
    std::vector<std::vector<double>> X = {{0.5, 1.0}, {1.2, 2.0}, {-0.4, 3.0}};
    run_case("sin_sqrt", t, X, {1.1, 0.9}, true);
}

// All remaining unary ops: square(exp(c0*x0)) + log(c1+x1) + cos(c2) + tanh(c3*x0)
//                          + abs(c4 - x1)   — k=5, exercises every unary.
void test_all_unary() {
    Tree t = {
        // square(exp(c0 * x0))
        constant_node(0, 0.4), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Exp), unary_node(UnaryOp::Square),
        // + log(c1 + x1)
        constant_node(1, 1.3), variable_node(1), binary_node(BinaryOp::Add),
        unary_node(UnaryOp::Log), binary_node(BinaryOp::Add),
        // + cos(c2)
        constant_node(2, 0.8), unary_node(UnaryOp::Cos), binary_node(BinaryOp::Add),
        // + tanh(c3 * x0)
        constant_node(3, -0.6), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Tanh), binary_node(BinaryOp::Add),
        // + abs(c4 - x1)
        constant_node(4, 0.2), variable_node(1), binary_node(BinaryOp::Sub),
        unary_node(UnaryOp::Abs), binary_node(BinaryOp::Add)};
    std::vector<std::vector<double>> X = {{0.5, 2.0}, {1.0, 3.5}, {-0.7, 1.1}};
    run_case("all_unary", t, X, {0.4, 1.3, 0.8, -0.6, 0.2}, true);
}

// inv: c0 * inv(x0 + c1)  — 1/x is unguarded, so keep the inputs away from the pole.
void test_inv() {
    Tree t = {constant_node(0, 1.4), variable_node(0), constant_node(1, 0.5),
              binary_node(BinaryOp::Add), unary_node(UnaryOp::Inv),
              binary_node(BinaryOp::Mul)};
    std::vector<std::vector<double>> X = {{1.0}, {2.5}, {-3.0}};
    run_case("inv", t, X, {1.4, 0.5}, true);
}

// pow on its standard branch: (x0 ^ c0) / (c1 ^ x1)  — base > 0, smooth.
void test_pow_standard() {
    Tree t = {variable_node(0), constant_node(0, 1.7), binary_node(BinaryOp::Pow),
              constant_node(1, 1.3), variable_node(1), binary_node(BinaryOp::Pow),
              binary_node(BinaryOp::Div)};
    std::vector<std::vector<double>> X = {{2.0, 1.5}, {3.0, 0.5}, {1.2, 2.2}};
    run_case("pow_standard", t, X, {1.7, 1.3}, true);
}

// pow guard branches (x <= 0): bit-identity must still hold; FD skipped because the
// guarded derivative is 0 by design and does not match a finite difference.
void test_pow_guarded() {
    Tree t = {variable_node(0), constant_node(0, 1.5), binary_node(BinaryOp::Pow),
              constant_node(1, 2.0), binary_node(BinaryOp::Add)};
    std::vector<std::vector<double>> X = {{-2.0}, {0.0}, {-0.5}};
    run_case("pow_guarded", t, X, {1.5, 2.0}, false);
}

// Many constants to exercise tiling past the block width: sum_{i<12} c_i * x_(i%4).
// k = 12 > kJacobianBlockWidth (8) and > all tested N except 16.
void test_tiling_large_k() {
    const int terms = 12;
    const int features = 4;
    Tree t;
    for (int i = 0; i < terms; ++i) {
        t.push_back(constant_node(i, 0.1 * (i + 1)));
        t.push_back(variable_node(i % features));
        t.push_back(binary_node(BinaryOp::Mul));
        if (i > 0) t.push_back(binary_node(BinaryOp::Add));
    }
    std::vector<std::vector<double>> X = {
        {0.5, 1.0, -1.5, 2.0}, {1.2, -0.3, 0.7, 1.1}, {2.0, 0.4, -0.9, 0.6}};
    std::vector<double> params(static_cast<std::size_t>(terms));
    for (int i = 0; i < terms; ++i)
        params[static_cast<std::size_t>(i)] = 0.1 * (i + 1);
    run_case("tiling_large_k", t, X, params, true);
}

// k exactly at and just past the production block width (8), to hit the boundary.
void test_block_boundary() {
    for (int terms : {8, 9}) {
        Tree t;
        for (int i = 0; i < terms; ++i) {
            t.push_back(constant_node(i, 0.2 * (i + 1)));
            t.push_back(variable_node(0));
            t.push_back(binary_node(BinaryOp::Mul));
            if (i > 0) t.push_back(binary_node(BinaryOp::Add));
        }
        std::vector<std::vector<double>> X = {{1.3}, {-0.7}, {2.1}};
        std::vector<double> params(static_cast<std::size_t>(terms));
        for (int i = 0; i < terms; ++i)
            params[static_cast<std::size_t>(i)] = 0.2 * (i + 1);
        run_case(terms == 8 ? "boundary_k8" : "boundary_k9", t, X, params, true);
    }
}

}  // namespace

int main() {
    test_linear();
    test_sin_sqrt();
    test_all_unary();
    test_inv();
    test_pow_standard();
    test_pow_guarded();
    test_tiling_large_k();
    test_block_boundary();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
