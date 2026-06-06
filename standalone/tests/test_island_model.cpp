// Island-model tests.
//
// Gates:
//   1. n_populations=1 produces the same result as the pre-island baseline.
//   2. n_populations=4 recovers the linear target.
//   3. Results are independent of the number of OpenMP threads
//      (n_populations=4, omp_set_num_threads(1) vs (4) => identical expression).
//   4. HallOfFame::merge is correct.

#include <cstdio>
#include <string>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/evolution/hall_of_fame.hpp"
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

}  // namespace

int main() {
    using namespace rsymbolic;
    const auto prob = problem_linear();

    // -----------------------------------------------------------------
    // Gate 1: n_populations=1 recovers the linear target (loss < 1e-6).
    // -----------------------------------------------------------------
    {
        const SearchResult r = run_evolution(prob.X, prob.y, linear_opts(1));
        std::printf("n_pops=1: loss=%.3e  expr=%s\n", r.loss, r.expression.c_str());
        CHECK(r.loss < 1e-6);
    }

    // -----------------------------------------------------------------
    // Gate 2: n_populations=4 also recovers the linear target.
    // -----------------------------------------------------------------
    {
        const SearchResult r = run_evolution(prob.X, prob.y, linear_opts(4));
        std::printf("n_pops=4: loss=%.3e  expr=%s\n", r.loss, r.expression.c_str());
        CHECK(r.loss < 1e-6);
    }

    // -----------------------------------------------------------------
    // Gate 3: thread count must not change results (determinism).
    // Run n_populations=4 with 1 thread and then with 4 threads; the
    // returned expressions must be identical because each island's RNG
    // is independent and migration is serial / RNG-free.
    // -----------------------------------------------------------------
    {
        std::string expr_t1, expr_t4;

#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        expr_t1 = run_evolution(prob.X, prob.y, linear_opts(4)).expression;

#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        expr_t4 = run_evolution(prob.X, prob.y, linear_opts(4)).expression;

        std::printf("thread-1 expr: %s\n", expr_t1.c_str());
        std::printf("thread-4 expr: %s\n", expr_t4.c_str());
        CHECK(expr_t1 == expr_t4);
    }

    // -----------------------------------------------------------------
    // Gate 4: HallOfFame::merge is correct.
    // -----------------------------------------------------------------
    {
        HallOfFame a, b;
        Tree dummy;
        a.update(PopMember{dummy, 1.0, 3});
        a.update(PopMember{dummy, 2.0, 5});
        b.update(PopMember{dummy, 0.5, 3});  // better at complexity 3
        b.update(PopMember{dummy, 3.0, 7});  // new complexity level

        a.merge(b);
        const PopMember best = a.best();
        CHECK(best.loss == 0.5);
        const auto front = a.pareto_front();
        // After merge: complexity 3 -> 0.5, complexity 7 -> 3.0
        // Pareto: [3, 0.5] and [7, 3.0] — but 3.0 > 0.5, so non-dominated
        // only has [3, 0.5]. Complexity 5 -> 2.0 but 2.0 > 0.5, so excluded.
        // Complexity 7 -> 3.0 but 3.0 > 0.5, so excluded.
        CHECK(front.size() == 1);
        CHECK(front[0].loss == 0.5 && front[0].complexity == 3);
    }

    if (g_failures == 0) {
        std::printf("All %d island-model checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d island-model checks FAILED\n", g_failures, g_checks);
    return 1;
}
