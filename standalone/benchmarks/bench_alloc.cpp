// Allocation-attribution probe for the per-evaluation hot path (docs/23 §4).
//
// The island-parallel health probe (benchmarks/diag_omp_check.R) shows cpu/wall
// ~1.5 at n_populations=4 (threads blocked, not spinning) — the signature of
// lock/heap contention, not frequency throttling. This probe localizes the heap
// pressure: it counts global operator new calls separately for
//   (a) make_least_squares_problem(tree, X, y, ...)  — builds the problem, and
//   (b) optimizer.optimize(problem)                  — runs Levenberg-Marquardt.
// fit() (evolutionary_search.cpp) calls both once per candidate evaluation.
//
// Hypothesis under test: make_least_squares_problem copies X (a
// vector<vector<double>> of m rows) BY VALUE on every call, costing ~m heap
// allocations per evaluation. With m=1000 that is ~1000 allocs/fit of pure waste
// (the dataset is invariant across the whole search), the dominant malloc traffic
// that serializes under multi-island parallelism.
//
// Not a gate; single-threaded; reports allocation counts, not time.

#include <atomic>
#include <cstdio>
#include <random>
#include <vector>

#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/optimization/optimizer_factory.hpp"

namespace {
std::atomic<long long> g_allocs{0};   // global operator new calls (our C++ code)
std::atomic<long long> g_mallocs{0};  // raw malloc/calloc/realloc calls (incl. Eigen)
}  // namespace

void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

// Count raw allocations too. Eigen's dynamic matrices use std::malloc directly (not
// operator new), so they are invisible to the operator-new counter above. The linker
// --wrap option routes every malloc/free call through these. operator new's own
// std::malloc is also wrapped, so g_mallocs >= g_allocs; the Eigen share is the gap.
extern "C" {
void* __real_malloc(std::size_t);
void  __real_free(void*);
void* __real_calloc(std::size_t, std::size_t);
void* __real_realloc(void*, std::size_t);

void* __wrap_malloc(std::size_t n) {
    g_mallocs.fetch_add(1, std::memory_order_relaxed);
    return __real_malloc(n);
}
void __wrap_free(void* p) { __real_free(p); }
void* __wrap_calloc(std::size_t a, std::size_t b) {
    g_mallocs.fetch_add(1, std::memory_order_relaxed);
    return __real_calloc(a, b);
}
void* __wrap_realloc(void* p, std::size_t n) {
    g_mallocs.fetch_add(1, std::memory_order_relaxed);
    return __real_realloc(p, n);
}
}  // extern "C"

int main() {
    using namespace rsymbolic;

    const std::size_t m = 1000;   // matches diag_omp_check.R dataset size
    const int features  = 4;

    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<std::vector<double>> X(m, std::vector<double>(features));
    std::vector<double> y(m);
    for (std::size_t i = 0; i < m; ++i) {
        for (int j = 0; j < features; ++j) X[i][j] = u(rng);
        y[i] = X[i][0] * X[i][1] + 0.5 * X[i][2];
    }
    // Share the dataset once, like run_evolution's hot path (docs/23 §4).
    const auto dataset =
        std::make_shared<const Dataset>(Dataset{std::move(X), std::move(y)});

    SearchSpace space;
    space.binary_ops   = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div};
    space.unary_ops    = {UnaryOp::Sin, UnaryOp::Cos, UnaryOp::Exp};
    space.num_features = features;
    space.max_depth    = 6;
    space.max_nodes    = 50;

    // Find a representative tree that actually has tunable constants, so optimize()
    // does real LM work (k == 0 would short-circuit it).
    Tree tree;
    int k = 0;
    for (int tries = 0; tries < 1000 && k == 0; ++tries) {
        tree = generate_random_tree(space, rng);
        k = count_constants(tree);
    }
    const std::vector<double> init = initial_constants(tree);

    const auto eigen_opt = OptimizerFactory::create(OptimizerType::EigenLM, {});
    const auto self_opt  = OptimizerFactory::create(OptimizerType::SelfLM, {});

    // Warm up both (let any one-time allocations / scratch sizing settle).
    {
        OptimizationProblem p = make_least_squares_problem(tree, dataset, init);
        eigen_opt->optimize(p);
        self_opt->optimize(p);
    }

    const int iters = 200;
    long long new_make = 0, mal_make = 0;        // problem construction
    long long new_eig = 0, mal_eig = 0;          // EigenLM optimize
    long long new_self = 0, mal_self = 0;        // SelfLM optimize
    for (int it = 0; it < iters; ++it) {
        g_allocs.store(0); g_mallocs.store(0);
        OptimizationProblem p = make_least_squares_problem(tree, dataset, init);
        new_make += g_allocs.load(); mal_make += g_mallocs.load();

        g_allocs.store(0); g_mallocs.store(0);
        eigen_opt->optimize(p);
        new_eig += g_allocs.load(); mal_eig += g_mallocs.load();

        // Re-build the problem so SelfLM sees the same fresh-problem state as EigenLM
        // (the problem is otherwise const across the two optimize calls; rebuilding
        // keeps the comparison apples-to-apples and resets the abort flag).
        g_allocs.store(0); g_mallocs.store(0);
        OptimizationProblem p2 = make_least_squares_problem(tree, dataset, init);
        (void)g_allocs.load(); (void)g_mallocs.load();

        g_allocs.store(0); g_mallocs.store(0);
        self_opt->optimize(p2);
        new_self += g_allocs.load(); mal_self += g_mallocs.load();
    }

    std::printf("dataset: m=%zu rows  features=%d  tree constants k=%d\n", m, features, k);
    std::printf("iterations: %d\n\n", iters);
    std::printf("%-34s %10s %10s\n", "", "op-new", "malloc");
    std::printf("%-34s %10.1f %10.1f\n", "make_least_squares / fit",
                (double)new_make / iters, (double)mal_make / iters);
    std::printf("%-34s %10.1f %10.1f\n", "EigenLM optimize / fit",
                (double)new_eig / iters, (double)mal_eig / iters);
    std::printf("%-34s %10.1f %10.1f\n", "SelfLM optimize / fit",
                (double)new_self / iters, (double)mal_self / iters);
    std::printf("\nEigen raw allocs in optimize (malloc not seen by op-new): %.1f/fit\n",
                (double)(mal_eig - new_eig) / iters);
    std::printf("SelfLM raw allocs in optimize (malloc not seen by op-new): %.1f/fit\n",
                (double)(mal_self - new_self) / iters);
    return 0;
}
