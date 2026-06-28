// Evidence micro-benchmark for the proposed SoA (struct-of-arrays) point-batched
// evaluator (docs/30 "Real prerequisite for a Float32 win").
//
// The hypothesis under test: rewriting the residual evaluator to process a *batch of
// P points per node* (instead of one point at a time through the postfix stack) lets
// the compiler auto-vectorise the inner P-loop, speeding up the residual closure in
// Float64 already. The known risk (docs/30, Float32 section): the Feynman search is
// rate-limited by transcendentals (exp/log/sin/cos/sqrt/pow), and libm's scalar
// transcendentals do NOT auto-vectorise without a vector math library (SLEEF / libmvec /
// SVML), which we do not ship. If the gain only appears on pure +-*/ trees and vanishes
// on transcendental-heavy ones, the rewrite buys little on the real workload — exactly
// the trap the Float32 attempt fell into, so we measure BEFORE committing.
//
// This driver compares, for several representative trees:
//   scalar : the production evaluate<double>() called once per point (current code path)
//   batch  : a hand-written SoA evaluator, switch hoisted out of a tight per-point loop
// Both compute bit-identical residuals (same ops, same order, no -ffast-math), which the
// driver verifies. It reports ns per full m-point pass and the batch/scalar speedup.
//
// Build (compile at BOTH -O2 (R ships -O2) and -O3 (standalone Release); add
// -fopt-info-vec-optimized to see which loops actually vectorised):
//   g++ -std=c++17 -O2 -march=native -fopt-info-vec-optimized \
//       -I r-package/rsymbolic2/src standalone/benchmarks/bench_soa_eval.cpp -o bench_soa_O2
//
// NOT part of the shipped package; pure evidence-gathering.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"

using namespace rsymbolic;

namespace {

// ---------------------------------------------------------------------------------
// SoA batched residual evaluator.
//
// Stack of length-P segments carved from one reused pool. Each Variable/Constant node
// copies into a fresh owned segment so later in-place unary/binary writes never corrupt
// the shared input columns. The switch on node kind/op is OUTSIDE the per-point loop, so
// each loop body is a single scalar op over P contiguous doubles — the form the compiler
// can vectorise (for +-*/) and the form it CANNOT vectorise for libm transcendentals
// without a vector-math library. Every op matches tree.hpp/dual.hpp exactly (bit-for-bit).
// ---------------------------------------------------------------------------------

void batch_unary(UnaryOp op, double* a, std::size_t P) {
    switch (op) {
        case UnaryOp::Neg:    for (std::size_t p = 0; p < P; ++p) a[p] = -a[p]; break;
        case UnaryOp::Exp:    for (std::size_t p = 0; p < P; ++p) a[p] = std::exp(a[p]); break;
        case UnaryOp::Log:    for (std::size_t p = 0; p < P; ++p) a[p] = std::log(a[p]); break;
        case UnaryOp::Sin:    for (std::size_t p = 0; p < P; ++p) a[p] = std::sin(a[p]); break;
        case UnaryOp::Cos:    for (std::size_t p = 0; p < P; ++p) a[p] = std::cos(a[p]); break;
        case UnaryOp::Sqrt:   for (std::size_t p = 0; p < P; ++p) a[p] = std::sqrt(a[p]); break;
        case UnaryOp::Tanh:   for (std::size_t p = 0; p < P; ++p) a[p] = std::tanh(a[p]); break;
        case UnaryOp::Abs:    for (std::size_t p = 0; p < P; ++p) a[p] = std::abs(a[p]); break;
        case UnaryOp::Square: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] * a[p]; break;
    }
}

void batch_binary(BinaryOp op, double* a, const double* b, std::size_t P) {
    switch (op) {
        case BinaryOp::Add: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] + b[p]; break;
        case BinaryOp::Sub: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] - b[p]; break;
        case BinaryOp::Mul: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] * b[p]; break;
        case BinaryOp::Div: for (std::size_t p = 0; p < P; ++p) a[p] = a[p] / b[p]; break;
        // safe pow (rsymbolic::pow): per-element branches; cannot vectorise (representative).
        case BinaryOp::Pow: for (std::size_t p = 0; p < P; ++p) a[p] = rsymbolic::pow(a[p], b[p]); break;
    }
}

// cols[j] points to the length-P column of variable j (struct-of-arrays input layout).
// Returns a pointer to the length-P result column. pool/stk are reused scratch (no
// per-call allocation).
double* evaluate_batch(const Tree& tree, const std::vector<const double*>& cols,
                       const double* constants, std::size_t P,
                       std::vector<double>& pool, std::vector<double*>& stk) {
    pool.resize(tree.size() * P);
    stk.clear();
    std::size_t next_seg = 0;
    auto seg = [&]() -> double* { return pool.data() + (next_seg++) * P; };
    for (const Node& node : tree) {
        switch (node.kind) {
            case NodeKind::Constant: {
                double* s = seg();
                const double v = constants[node.index];
                for (std::size_t p = 0; p < P; ++p) s[p] = v;
                stk.push_back(s);
                break;
            }
            case NodeKind::Variable: {
                double* s = seg();
                const double* src = cols[static_cast<std::size_t>(node.index)];
                std::memcpy(s, src, P * sizeof(double));
                stk.push_back(s);
                break;
            }
            case NodeKind::Unary: {
                batch_unary(node.uop, stk.back(), P);
                break;
            }
            case NodeKind::Binary: {
                double* b = stk.back(); stk.pop_back();
                double* a = stk.back();
                batch_binary(node.bop, a, b, P);
                break;
            }
        }
    }
    return stk.back();
}

// ---------------------------------------------------------------------------------
// Representative trees, ordered by transcendental density.
// ---------------------------------------------------------------------------------
struct Case { std::string name; Tree tree; int n_vars; std::vector<double> consts; };

// poly: c0*x0^2 + c1*x1 + c2   — pure arithmetic (+,-,*,square). Best case for SoA.
Case make_poly() {
    Tree t = {
        constant_node(0, 0.5), variable_node(0), unary_node(UnaryOp::Square),
        binary_node(BinaryOp::Mul),
        constant_node(1, 1.3), variable_node(1), binary_node(BinaryOp::Mul),
        binary_node(BinaryOp::Add),
        constant_node(2, 0.7), binary_node(BinaryOp::Add)};
    return {"poly (pure arith)", t, 2, {0.5, 1.3, 0.7}};
}

// rel_mass: c0 / sqrt(1 - (x0/x1)^2)   — one sqrt + div (the I.10.7 target shape).
Case make_rel_mass() {
    Tree t = {
        constant_node(0, 2.0),
        // 1 - (x0/x1)^2
        constant_node(1, 1.0), variable_node(0), variable_node(1),
        binary_node(BinaryOp::Div), unary_node(UnaryOp::Square),
        binary_node(BinaryOp::Sub),
        unary_node(UnaryOp::Sqrt),
        binary_node(BinaryOp::Div)};
    return {"rel_mass (1 sqrt+div)", t, 2, {2.0, 1.0}};
}

// trig: sin(c0*x0) * cos(c1*x1) + c2   — two transcendentals.
Case make_trig() {
    Tree t = {
        constant_node(0, 1.1), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Sin),
        constant_node(1, 0.9), variable_node(1), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Cos),
        binary_node(BinaryOp::Mul),
        constant_node(2, 0.3), binary_node(BinaryOp::Add)};
    return {"trig (sin*cos)", t, 2, {1.1, 0.9, 0.3}};
}

// transc: exp(c0*x0) + log(c1 + x1^2) + sin(c2*x0)  — transcendental-heavy (exp/log/sin),
// the character of the timed-out Feynman trees (docs/30).
Case make_transc() {
    Tree t = {
        constant_node(0, 0.4), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Exp),
        constant_node(1, 1.3), variable_node(1), unary_node(UnaryOp::Square),
        binary_node(BinaryOp::Add), unary_node(UnaryOp::Log),
        binary_node(BinaryOp::Add),
        constant_node(2, 0.8), variable_node(0), binary_node(BinaryOp::Mul),
        unary_node(UnaryOp::Sin), binary_node(BinaryOp::Add)};
    return {"transc (exp+log+sin)", t, 2, {0.4, 1.3, 0.8}};
}

bool same_bits(double a, double b) {
    std::uint64_t ua, ub;
    std::memcpy(&ua, &a, sizeof(ua));
    std::memcpy(&ub, &b, sizeof(ub));
    return ua == ub;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::size_t P = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000;
    const int reps       = argc > 2 ? std::atoi(argv[2]) : 20000;

    // Sample P points, 3 feature columns, in both layouts: row-major for the scalar
    // evaluator (its native input) and column-major for the SoA evaluator (its native).
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(1.0, 5.0);
    const int nf = 3;
    std::vector<std::vector<double>> rows(P, std::vector<double>(nf));
    std::vector<std::vector<double>> colsv(nf, std::vector<double>(P));
    for (std::size_t p = 0; p < P; ++p)
        for (int j = 0; j < nf; ++j) { double v = u(rng); rows[p][j] = v; colsv[j][p] = v; }
    std::vector<const double*> cols(nf);
    for (int j = 0; j < nf; ++j) cols[j] = colsv[j].data();

    std::vector<Case> cases = {make_poly(), make_rel_mass(), make_trig(), make_transc()};

    std::printf("bench_soa_eval  P=%zu  reps=%d\n", P, reps);
    std::printf("%-22s %12s %12s %9s %s\n",
                "tree", "scalar ns", "batch ns", "speedup", "bit-exact");

    std::vector<double> scalar_stack;
    std::vector<double> pool;
    std::vector<double*> stk;
    std::vector<double> out_scalar(P), out_batch(P);

    for (Case& c : cases) {
        // Correctness: residuals must be bit-identical between the two evaluators.
        for (std::size_t p = 0; p < P; ++p)
            out_scalar[p] = evaluate<double>(c.tree, rows[p].data(), c.consts.data(),
                                             scalar_stack);
        {
            const double* res = evaluate_batch(c.tree, cols, c.consts.data(), P, pool, stk);
            for (std::size_t p = 0; p < P; ++p) out_batch[p] = res[p];
        }
        bool exact = true;
        for (std::size_t p = 0; p < P && exact; ++p)
            if (!same_bits(out_scalar[p], out_batch[p])) exact = false;

        // Time scalar: one evaluate<double> per point, reps full passes.
        volatile double sink = 0.0;
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            double s = 0.0;
            for (std::size_t p = 0; p < P; ++p)
                s += evaluate<double>(c.tree, rows[p].data(), c.consts.data(), scalar_stack);
            sink = sink + s;
        }
        auto t1 = std::chrono::steady_clock::now();
        const double scalar_ns =
            std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;

        // Time batch: one SoA pass over all P points, reps full passes.
        t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            const double* res = evaluate_batch(c.tree, cols, c.consts.data(), P, pool, stk);
            double s = 0.0;
            for (std::size_t p = 0; p < P; ++p) s += res[p];
            sink = sink + s;
        }
        t1 = std::chrono::steady_clock::now();
        const double batch_ns =
            std::chrono::duration<double, std::nano>(t1 - t0).count() / reps;
        (void)sink;

        std::printf("%-22s %12.0f %12.0f %8.2fx %s\n", c.name.c_str(), scalar_ns,
                    batch_ns, scalar_ns / batch_ns, exact ? "yes" : "NO");
    }
    return 0;
}
