// De-risking probe (docs/23 §4): does reusing one Eigen LevenbergMarquardt object
// across fits eliminate the per-fit heap allocations that serialize island workers?
//
// bench_heap proved the serialization is inside Eigen's optimize() (our residual-only
// path scales; EigenLM blocks), and bench_alloc showed optimize does ~35 raw malloc/fit
// invisible to operator new (Eigen's dynamic matrices). If those allocations happen
// only in minimizeInit/resizeit (per object), a reused object would allocate on the
// first minimize() and ~0 thereafter -> a contained fix (reuse the LM per island) works.
// If minimizeOneStep itself allocates per step, reuse won't help and a different
// optimizer (allocation-free) is required.
//
// This replicates the real AnalyticFunctor (eigen_lm_optimizer.cpp) closely enough to
// measure allocation behaviour. Raw malloc is counted via --wrap (see CMake).

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/tree.hpp"

using namespace rsymbolic;

namespace {
std::atomic<long long> g_mallocs{0};
constexpr double kLargeResidual = 1.0e10;
}  // namespace

extern "C" {
void* __real_malloc(std::size_t);
void  __real_free(void*);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);
void* __wrap_malloc(std::size_t n) { g_mallocs.fetch_add(1, std::memory_order_relaxed); return __real_malloc(n); }
void  __wrap_free(void* p) { __real_free(p); }
void* __wrap_calloc(std::size_t a, std::size_t b) { g_mallocs.fetch_add(1, std::memory_order_relaxed); return __real_calloc(a, b); }
void* __wrap_realloc(void* p, std::size_t n) { g_mallocs.fetch_add(1, std::memory_order_relaxed); return __real_realloc(p, n); }
}

namespace {

struct AnalyticFunctor {
    using Scalar = double;
    using InputType = Eigen::VectorXd;
    using ValueType = Eigen::VectorXd;
    using JacobianType = Eigen::MatrixXd;
    enum { InputsAtCompileTime = Eigen::Dynamic, ValuesAtCompileTime = Eigen::Dynamic };
    const ResidualFunction& residuals;
    const JacobianFunction& jacobian;
    int n_inputs, n_values;
    std::vector<double>& params;
    std::vector<double>& rbuf;
    std::vector<double>& jbuf;
    int inputs() const { return n_inputs; }
    int values() const { return n_values; }
    int operator()(const InputType& x, ValueType& fvec) const {
        for (int i = 0; i < n_inputs; ++i) params[(std::size_t)i] = x[i];
        residuals(params, rbuf);
        for (int i = 0; i < n_values; ++i)
            fvec[i] = std::isfinite(rbuf[(std::size_t)i]) ? rbuf[(std::size_t)i] : kLargeResidual;
        return 0;
    }
    int df(const InputType& x, JacobianType& fjac) const {
        for (int i = 0; i < n_inputs; ++i) params[(std::size_t)i] = x[i];
        jacobian(params, jbuf);
        fjac.resize(n_values, n_inputs);
        for (int i = 0; i < n_values; ++i)
            for (int j = 0; j < n_inputs; ++j)
                fjac(i, j) = jbuf[(std::size_t)i * n_inputs + j];
        return 0;
    }
};

}  // namespace

int main() {
    const std::size_t m = 1000;
    const int features = 4;
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<std::vector<double>> X(m, std::vector<double>(features));
    std::vector<double> y(m);
    for (std::size_t i = 0; i < m; ++i) {
        for (int j = 0; j < features; ++j) X[i][j] = u(rng);
        y[i] = X[i][0] * X[i][1] + 0.5 * X[i][2];
    }
    const auto dataset = std::make_shared<const Dataset>(Dataset{std::move(X), std::move(y)});

    SearchSpace space;
    space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    space.unary_ops = {UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp};
    space.num_features = features;
    space.max_depth = 6; space.max_nodes = 50;

    Tree tree;
    int k = 0;
    for (int t = 0; t < 1000 && k == 0; ++t) { tree = generate_random_tree(space, rng); k = count_constants(tree); }
    const std::vector<double> init = initial_constants(tree);

    OptimizationProblem p = make_least_squares_problem(tree, dataset, init);
    const int kk = (int)init.size();
    const int mm = (int)m;
    std::vector<double> params(kk), rbuf(mm), jbuf((std::size_t)mm * kk);

    // Persistent LM object + functor referencing stable buffers and the problem's
    // (stable) closures. Run minimize() repeatedly on the SAME object.
    AnalyticFunctor functor{p.residuals, p.jacobian, kk, mm, params, rbuf, jbuf};
    Eigen::LevenbergMarquardt<AnalyticFunctor> lm(functor);
    lm.parameters.maxfev = 100 * (kk + 1);

    std::printf("k=%d  m=%d\n", kk, mm);
    for (int call = 0; call < 5; ++call) {
        Eigen::VectorXd x(kk);
        for (int i = 0; i < kk; ++i) x[i] = init[(std::size_t)i];
        g_mallocs.store(0, std::memory_order_relaxed);
        lm.minimize(x);  // reuse the same lm object across calls
        std::printf("minimize() call %d on REUSED lm object: %lld mallocs\n",
                    call, g_mallocs.load(std::memory_order_relaxed));
    }

    // For contrast: a fresh lm object each call (what the library does today).
    std::printf("--- fresh lm object each call (current behaviour) ---\n");
    for (int call = 0; call < 3; ++call) {
        Eigen::VectorXd x(kk);
        for (int i = 0; i < kk; ++i) x[i] = init[(std::size_t)i];
        g_mallocs.store(0, std::memory_order_relaxed);
        AnalyticFunctor f2{p.residuals, p.jacobian, kk, mm, params, rbuf, jbuf};
        Eigen::LevenbergMarquardt<AnalyticFunctor> lm2(f2);
        lm2.parameters.maxfev = 100 * (kk + 1);
        lm2.minimize(x);
        std::printf("minimize() call %d on FRESH lm object:  %lld mallocs\n",
                    call, g_mallocs.load(std::memory_order_relaxed));
    }
    return 0;
}
