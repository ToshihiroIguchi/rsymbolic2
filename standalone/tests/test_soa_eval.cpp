// Unit tests for the SoA (struct-of-arrays) point-batched evaluator (soa_eval.hpp).
//
// The load-bearing guarantee is BIT-IDENTITY with the scalar path: the SoA residual must
// equal the production scalar evaluate<double>() per point, and the SoA Jacobian must
// equal the canonical scalar k-pass forward-mode Dual reference per entry — down to the
// last bit. Bit-identity is what proves the whole search is numerically unchanged (only
// faster) after switching the fit() closures to SoA. We assert it strictly (raw bit
// patterns, no tolerance). A finite-difference check on smooth configurations guards the
// derivative formulas independently of the scalar reference.
//
// Coverage mirrors test_multi_dual.cpp (every unary/binary op, pow std + guarded
// branches, k beyond the block width to force tiling) and additionally exercises:
//   - column-major (Xcol) input layout the SoA evaluators consume;
//   - point counts that cross the kStride tile boundary (the closures tile points).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "rsymbolic/expression/dual.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"  // kStride
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/soa_eval.hpp"
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
using rsymbolic::evaluate;
using rsymbolic::evaluate_soa_jacobian;
using rsymbolic::evaluate_soa_residual;
using rsymbolic::kStride;
using rsymbolic::Node;
using rsymbolic::Tree;
using rsymbolic::UnaryOp;
using rsymbolic::binary_node;
using rsymbolic::constant_node;
using rsymbolic::unary_node;
using rsymbolic::variable_node;

bool same_bits(double a, double b) {
    std::uint64_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

// Column-major view (Xcol[j][i]) of a row-major X, the layout the SoA evaluators take.
std::vector<std::vector<double>> to_columns(const std::vector<std::vector<double>>& X) {
    const std::size_t m = X.size();
    const std::size_t nf = m ? X[0].size() : 0;
    std::vector<std::vector<double>> cols(nf, std::vector<double>(m));
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < nf; ++j) cols[j][i] = X[i][j];
    return cols;
}

// Reference residual: production scalar evaluator, one point at a time.
std::vector<double> reference_residual(const Tree& tree,
                                       const std::vector<std::vector<double>>& X,
                                       const std::vector<double>& params) {
    std::vector<double> r(X.size());
    std::vector<double> stack;
    for (std::size_t i = 0; i < X.size(); ++i)
        r[i] = evaluate<double>(tree, X[i].data(), params.data(), stack);
    return r;
}

// SoA residual driven exactly as the fit() closure does (tiled at kStride).
std::vector<double> soa_residual(const Tree& tree,
                                 const std::vector<std::vector<double>>& cols,
                                 std::size_t m, const std::vector<double>& params) {
    std::vector<double> r(m);
    std::vector<double> pool;
    std::vector<double*> stk;
    for (std::size_t lo = 0; lo < m; lo += kStride) {
        const std::size_t P = std::min(kStride, m - lo);
        const double* pred =
            evaluate_soa_residual(tree, cols, params.data(), lo, P, pool, stk);
        for (std::size_t p = 0; p < P; ++p) r[lo + p] = pred[p];
    }
    return r;
}

// Reference Jacobian: scalar k-pass forward-mode Dual (one direction per pass).
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
            jac[i * static_cast<std::size_t>(k) + static_cast<std::size_t>(kk)] = p.deriv;
        }
    }
    return jac;
}

// SoA Jacobian at block width N, driven exactly as the fit() closure does.
template <int N>
std::vector<double> soa_jacobian(const Tree& tree,
                                 const std::vector<std::vector<double>>& cols,
                                 std::size_t m, const std::vector<double>& params) {
    const int k = static_cast<int>(params.size());
    std::vector<double> jac(m * static_cast<std::size_t>(k), 0.0);
    std::vector<double> pool;
    std::vector<double*> stk;
    std::vector<double> coeff;
    for (int block = 0; block < k; block += N) {
        const int w = std::min(N, k - block);
        for (std::size_t lo = 0; lo < m; lo += kStride) {
            const std::size_t P = std::min(kStride, m - lo);
            const double* seg = evaluate_soa_jacobian<N>(
                tree, cols, params.data(), k, block, lo, P, pool, stk, coeff);
            for (int c = 0; c < w; ++c) {
                const double* gc = seg + static_cast<std::size_t>(1 + c) * P;
                for (std::size_t p = 0; p < P; ++p)
                    jac[(lo + p) * static_cast<std::size_t>(k) +
                        static_cast<std::size_t>(block + c)] = gc[p];
            }
        }
    }
    return jac;
}

void run_case(const char* label, const Tree& tree,
              const std::vector<std::vector<double>>& X,
              const std::vector<double>& params, bool finite_diff) {
    const int k = static_cast<int>(params.size());
    const std::size_t m = X.size();
    const std::vector<std::vector<double>> cols = to_columns(X);

    // --- residual bit-identity ---
    const std::vector<double> r_ref = reference_residual(tree, X, params);
    const std::vector<double> r_soa = soa_residual(tree, cols, m, params);
    bool res_ok = (r_ref.size() == r_soa.size());
    for (std::size_t i = 0; i < r_ref.size() && res_ok; ++i)
        if (!same_bits(r_ref[i], r_soa[i])) res_ok = false;
    if (!res_ok) std::printf("  [%s] SoA residual differs from scalar\n", label);
    CHECK(res_ok);

    // --- Jacobian bit-identity (production N=8, plus N=4/16 to certify N-independence) ---
    const std::vector<double> j_ref = reference_jacobian(tree, X, params);
    const std::vector<double> j8 = soa_jacobian<8>(tree, cols, m, params);
    const std::vector<double> j4 = soa_jacobian<4>(tree, cols, m, params);
    const std::vector<double> j16 = soa_jacobian<16>(tree, cols, m, params);
    bool jac_ok = (j_ref.size() == j8.size());
    for (std::size_t e = 0; e < j_ref.size() && jac_ok; ++e)
        if (!same_bits(j8[e], j_ref[e]) || !same_bits(j4[e], j_ref[e]) ||
            !same_bits(j16[e], j_ref[e]))
            jac_ok = false;
    if (!jac_ok) std::printf("  [%s] SoA Jacobian differs from scalar\n", label);
    CHECK(jac_ok);

    if (!finite_diff) return;

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
                j_ref[i * static_cast<std::size_t>(k) + static_cast<std::size_t>(j)];
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

// ((c0 * x0) + c1)
void test_linear() {
    Tree t = {constant_node(0, 0.7), variable_node(0), binary_node(BinaryOp::Mul),
              constant_node(1, -0.3), binary_node(BinaryOp::Add)};
    std::vector<std::vector<double>> X = {{1.5}, {-2.0}, {0.3}, {4.1}};
    run_case("linear", t, X, {0.7, -0.3}, true);
}

// sin(c0*x0) * sqrt(c1 + x1)
void test_sin_sqrt() {
    Tree t = {constant_node(0, 1.1), variable_node(0), binary_node(BinaryOp::Mul),
              unary_node(UnaryOp::Sin), constant_node(1, 0.9), variable_node(1),
              binary_node(BinaryOp::Add), unary_node(UnaryOp::Sqrt),
              binary_node(BinaryOp::Mul)};
    std::vector<std::vector<double>> X = {{0.5, 1.0}, {1.2, 2.0}, {-0.4, 3.0}};
    run_case("sin_sqrt", t, X, {1.1, 0.9}, true);
}

// Every unary op: square(exp(c0*x0)) + log(c1+x1) + cos(c2) + tanh(c3*x0) + abs(c4-x1)
void test_all_unary() {
    Tree t = {
        constant_node(0, 0.4), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Exp), unary_node(UnaryOp::Square),
        constant_node(1, 1.3), variable_node(1), binary_node(BinaryOp::Add),
        unary_node(UnaryOp::Log), binary_node(BinaryOp::Add),
        constant_node(2, 0.8), unary_node(UnaryOp::Cos), binary_node(BinaryOp::Add),
        constant_node(3, -0.6), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Tanh), binary_node(BinaryOp::Add),
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

// Neg + Div: -(c0 / (x0 + c1))
void test_neg_div() {
    Tree t = {constant_node(0, 1.4), variable_node(0), constant_node(1, 0.5),
              binary_node(BinaryOp::Add), binary_node(BinaryOp::Div),
              unary_node(UnaryOp::Neg)};
    std::vector<std::vector<double>> X = {{1.0}, {2.5}, {0.3}, {-3.0}};
    run_case("neg_div", t, X, {1.4, 0.5}, true);
}

// pow standard branch: (x0 ^ c0) / (c1 ^ x1)
void test_pow_standard() {
    Tree t = {variable_node(0), constant_node(0, 1.7), binary_node(BinaryOp::Pow),
              constant_node(1, 1.3), variable_node(1), binary_node(BinaryOp::Pow),
              binary_node(BinaryOp::Div)};
    std::vector<std::vector<double>> X = {{2.0, 1.5}, {3.0, 0.5}, {1.2, 2.2}};
    run_case("pow_standard", t, X, {1.7, 1.3}, true);
}

// pow guard branches (x <= 0): bit-identity still required; FD skipped (guarded deriv = 0).
void test_pow_guarded() {
    Tree t = {variable_node(0), constant_node(0, 1.5), binary_node(BinaryOp::Pow),
              constant_node(1, 2.0), binary_node(BinaryOp::Add)};
    std::vector<std::vector<double>> X = {{-2.0}, {0.0}, {-0.5}};
    run_case("pow_guarded", t, X, {1.5, 2.0}, false);
}

// k = 12 > block width 8: forces tiling over constant blocks.
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
    for (int i = 0; i < terms; ++i) params[static_cast<std::size_t>(i)] = 0.1 * (i + 1);
    run_case("tiling_large_k", t, X, params, true);
}

// Many points (> kStride) to cross the point-tile boundary and exercise a partial tile.
void test_point_tiling() {
    Tree t = {constant_node(0, 0.6), variable_node(0), binary_node(BinaryOp::Mul),
              unary_node(UnaryOp::Sin), constant_node(1, 0.2),
              binary_node(BinaryOp::Add)};
    const std::size_t m = 2 * kStride + 37;  // two full tiles + a partial last tile
    std::vector<std::vector<double>> X(m, std::vector<double>(1));
    for (std::size_t i = 0; i < m; ++i)
        X[i][0] = -3.0 + 6.0 * (static_cast<double>(i) / static_cast<double>(m));
    run_case("point_tiling", t, X, {0.6, 0.2}, true);
}

}  // namespace

int main() {
    test_linear();
    test_sin_sqrt();
    test_all_unary();
    test_inv();
    test_neg_div();
    test_pow_standard();
    test_pow_guarded();
    test_tiling_large_k();
    test_point_tiling();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
