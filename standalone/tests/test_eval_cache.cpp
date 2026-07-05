// Duplicate-evaluation cache tests (opt-in SearchOptions::eval_cache; eval_cache.hpp).
//
// Gates:
//   1. tree_hash distinguishes structure / constants / variable indices and is
//      equal for identical trees.
//   2. tree_eval_equal compares the evaluation-relevant fields, with bitwise
//      (NaN-stable) constant comparison.
//   3. A key collision with a different tree is a miss, never a wrong hit.
//   4. A 2-slot cache overwrites unconditionally (last write wins) and stays correct.
//   5. eval_cache on vs off is bit-identical: expression, loss, Pareto front,
//      n_evals — including under a max_evals budget (hit charging).
//   6. Thread-count determinism holds with the cache on.
//   7. Counters: a real run with the cache on records hits and misses; batching
//      keeps the cache inactive and the result unchanged.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/search/eval_cache.hpp"
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

// Build search options for the linear problem with given n_populations.
rsymbolic::SearchOptions linear_opts(std::size_t n_pops) {
    const auto prob = rsymbolic::problem_linear();
    rsymbolic::SearchOptions opts;
    opts.space            = prob.space;
    opts.population_size  = 200;
    opts.generations      = 40;
    opts.tournament_size  = 4;
    opts.target_loss      = 1e-10;
    opts.simplify_expressions = true;
    opts.seed             = 42;
    opts.n_populations    = n_pops;
    return opts;
}

// Bit-identity comparison of two search results: expression string, bitwise loss,
// Pareto front (size and per-member losses/complexities), and eval accounting.
void check_bit_identical(const rsymbolic::SearchResult& a,
                         const rsymbolic::SearchResult& b) {
    CHECK(a.expression == b.expression);
    CHECK(std::memcmp(&a.loss, &b.loss, sizeof(double)) == 0);  // bitwise-equal loss
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

}  // namespace

int main() {
    using namespace rsymbolic;

    // -----------------------------------------------------------------
    // Gate 1: tree_hash sanity.
    // -----------------------------------------------------------------
    {
        // x0 + 2.0  (postfix: x0, c0=2.0, +)
        Tree t1{variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Add)};
        Tree t1_copy = t1;
        CHECK(tree_hash(t1) == tree_hash(t1_copy));           // identical => equal

        Tree t2{variable_node(0), constant_node(0, 3.0), binary_node(BinaryOp::Add)};
        CHECK(tree_hash(t1) != tree_hash(t2));                // different constant

        Tree t3{variable_node(1), constant_node(0, 2.0), binary_node(BinaryOp::Add)};
        CHECK(tree_hash(t1) != tree_hash(t3));                // different variable index

        Tree t4{variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Mul)};
        CHECK(tree_hash(t1) != tree_hash(t4));                // different operator

        Tree t5{variable_node(0), unary_node(UnaryOp::Sin)};
        Tree t6{variable_node(0), unary_node(UnaryOp::Cos)};
        CHECK(tree_hash(t5) != tree_hash(t6));                // different unary op
        CHECK(tree_hash(t1) != tree_hash(t5));                // different structure
    }

    // -----------------------------------------------------------------
    // Gate 2: tree_eval_equal — eval-relevant fields, bitwise constants.
    // -----------------------------------------------------------------
    {
        Tree a{variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Add)};
        Tree b{variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Add)};
        CHECK(tree_eval_equal(a, b));

        Tree c{variable_node(0), constant_node(0, 2.5), binary_node(BinaryOp::Add)};
        CHECK(!tree_eval_equal(a, c));                        // constant value differs
        Tree d{variable_node(1), constant_node(0, 2.0), binary_node(BinaryOp::Add)};
        CHECK(!tree_eval_equal(a, d));                        // variable index differs
        Tree e{variable_node(0), constant_node(0, 2.0), binary_node(BinaryOp::Sub)};
        CHECK(!tree_eval_equal(a, e));                        // operator differs
        Tree f{variable_node(0), constant_node(0, 2.0)};
        CHECK(!tree_eval_equal(a, f));                        // size differs

        // Bitwise NaN handling: a NaN constant compares equal to the same NaN bit
        // pattern (IEEE == would say unequal), and hashes identically too.
        const double nan = std::numeric_limits<double>::quiet_NaN();
        Tree n1{constant_node(0, nan)};
        Tree n2{constant_node(0, nan)};
        CHECK(tree_eval_equal(n1, n2));
        CHECK(tree_hash(n1) == tree_hash(n2));

        // Constant slot index is evaluation-relevant (initial_constants round-trip):
        // same value, different slot => not eval-equal, different hash.
        Tree s1{constant_node(0, 2.0), constant_node(1, 2.0), binary_node(BinaryOp::Add)};
        Tree s2{constant_node(0, 2.0), constant_node(0, 2.0), binary_node(BinaryOp::Add)};
        CHECK(!tree_eval_equal(s1, s2));
        CHECK(tree_hash(s1) != tree_hash(s2));
    }

    // -----------------------------------------------------------------
    // Gate 3: collision guard — same key, different tree => miss.
    // -----------------------------------------------------------------
    {
        EvalCache cache(4);
        Tree a{variable_node(0), unary_node(UnaryOp::Sin)};
        Tree b{variable_node(0), unary_node(UnaryOp::Cos)};
        const std::uint64_t key = tree_hash(a);
        cache.store(key, a, 1.25);
        CHECK(cache.lookup(key, a) != nullptr);
        CHECK(*cache.lookup(key, a) == 1.25);
        // Forged collision: b stored nowhere, looked up under a's key => must miss.
        CHECK(cache.lookup(key, b) == nullptr);
        // An unused slot is a miss too.
        Tree c{variable_node(1)};
        const std::uint64_t key_c = tree_hash(c);
        if ((key_c & 3u) != (key & 3u)) CHECK(cache.lookup(key_c, c) == nullptr);
    }

    // -----------------------------------------------------------------
    // Gate 4: 2-slot cache — unconditional overwrite, last write wins.
    // -----------------------------------------------------------------
    {
        EvalCache cache(2);
        Tree a{variable_node(0)};
        Tree b{variable_node(0), unary_node(UnaryOp::Sin)};
        // Force both into the same slot by using keys with equal low bit.
        const std::uint64_t key_a = 0x10;  // slot 0
        const std::uint64_t key_b = 0x20;  // slot 0
        cache.store(key_a, a, 1.0);
        CHECK(cache.lookup(key_a, a) != nullptr && *cache.lookup(key_a, a) == 1.0);
        cache.store(key_b, b, 2.0);        // evicts a (direct-mapped overwrite)
        CHECK(cache.lookup(key_a, a) == nullptr);
        CHECK(cache.lookup(key_b, b) != nullptr && *cache.lookup(key_b, b) == 2.0);
        // Re-store a: b evicted in turn; values stay correct throughout.
        cache.store(key_a, a, 3.0);
        CHECK(cache.lookup(key_b, b) == nullptr);
        CHECK(cache.lookup(key_a, a) != nullptr && *cache.lookup(key_a, a) == 3.0);
    }

    // -----------------------------------------------------------------
    // Gate 5: eval_cache ON vs OFF is bit-identical (same options/seed),
    // with and without a max_evals budget (exercises hit charging).
    // -----------------------------------------------------------------
    const auto prob = problem_linear();
    {
        SearchOptions off = linear_opts(4);
        SearchOptions on  = linear_opts(4);
        on.eval_cache = true;
        const SearchResult r_off = run_evolution(prob.X, prob.y, off);
        const SearchResult r_on  = run_evolution(prob.X, prob.y, on);
        std::printf("cache off: loss=%.3e  expr=%s\n", r_off.loss, r_off.expression.c_str());
        std::printf("cache on : loss=%.3e  expr=%s\n", r_on.loss, r_on.expression.c_str());
        check_bit_identical(r_off, r_on);
        CHECK(r_off.cache_hits == 0 && r_off.cache_misses == 0);  // off => no stats
    }
    {
        // max_evals budget: a hit must be charged like a real eval, so the budgeted
        // runs stop at the same point and stay bit-identical.
        SearchOptions off = linear_opts(4);
        SearchOptions on  = linear_opts(4);
        off.target_loss = 0.0;      // don't early-stop: let the budget bind
        on.target_loss  = 0.0;
        off.max_evals = 30000;      // small enough to bind mid-run
        on.max_evals  = 30000;
        on.eval_cache = true;
        const SearchResult r_off = run_evolution(prob.X, prob.y, off);
        const SearchResult r_on  = run_evolution(prob.X, prob.y, on);
        std::printf("budget off: n_evals=%llu  expr=%s\n",
                    static_cast<unsigned long long>(r_off.n_evals),
                    r_off.expression.c_str());
        std::printf("budget on : n_evals=%llu  hits=%llu  expr=%s\n",
                    static_cast<unsigned long long>(r_on.n_evals),
                    static_cast<unsigned long long>(r_on.cache_hits),
                    r_on.expression.c_str());
        check_bit_identical(r_off, r_on);
    }

    // -----------------------------------------------------------------
    // Gate 6: thread-count determinism with the cache ON.
    // -----------------------------------------------------------------
    {
        SearchOptions on = linear_opts(4);
        on.eval_cache = true;
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        const SearchResult r_t1 = run_evolution(prob.X, prob.y, on);
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        const SearchResult r_t4 = run_evolution(prob.X, prob.y, on);
        std::printf("cache on thread-1 expr: %s\n", r_t1.expression.c_str());
        std::printf("cache on thread-4 expr: %s\n", r_t4.expression.c_str());
        CHECK(r_t1.expression == r_t4.expression);
        CHECK(r_t1.cache_hits == r_t4.cache_hits);
        CHECK(r_t1.cache_misses == r_t4.cache_misses);
    }

    // -----------------------------------------------------------------
    // Gate 7: counters. The linear problem's small operator set makes duplicate
    // candidates certain, so a cache-on run records both hits and misses; under
    // batching the cache stays inactive and the result is unchanged.
    // -----------------------------------------------------------------
    {
        SearchOptions on = linear_opts(1);
        on.eval_cache = true;
        const SearchResult r = run_evolution(prob.X, prob.y, on);
        std::printf("counters: hits=%llu  misses=%llu\n",
                    static_cast<unsigned long long>(r.cache_hits),
                    static_cast<unsigned long long>(r.cache_misses));
        CHECK(r.cache_hits + r.cache_misses > 0);
        CHECK(r.cache_hits > 0);
        CHECK(r.cache_misses > 0);
        // Hits and misses partition exactly the forward passes routed through the
        // cache (all of them on the full-data path), never exceeding the total.
        CHECK(r.cache_hits + r.cache_misses <= r.n_forward_evals);
    }
    {
        SearchOptions b_off = linear_opts(1);
        b_off.batching   = true;
        b_off.batch_size = 10;
        SearchOptions b_on = b_off;
        b_on.eval_cache = true;
        const SearchResult r_off = run_evolution(prob.X, prob.y, b_off);
        const SearchResult r_on  = run_evolution(prob.X, prob.y, b_on);
        check_bit_identical(r_off, r_on);
        CHECK(r_on.cache_hits == 0 && r_on.cache_misses == 0);  // inactive under batching
    }

    if (g_failures == 0) {
        std::printf("All %d eval-cache checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d eval-cache checks FAILED\n", g_failures, g_checks);
    return 1;
}
