// warmup_maxsize_by tests (PySR warmup_maxsize_by; SR.jl get_cur_maxsize).
//
// Gates:
//   1. get_cur_maxsize matches the SR.jl formula across the warmup ramp, the post-warmup
//      plateau, the warmup-off default, and the small-maxsize clamp.
//   2. warmup_maxsize_by = 0 (default) reproduces the fixed-maxsize search byte-for-byte
//      (same expression as leaving the field unset).
//   3. A warmup-on run completes, returns a finite loss with complexity <= max_nodes, and is
//      deterministic for a fixed seed.

#include <cstdio>
#include <string>

#include "rsymbolic/benchmark/problems.hpp"
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

rsymbolic::SearchOptions linear_opts() {
    const auto prob = rsymbolic::problem_linear();
    rsymbolic::SearchOptions opts;
    opts.space            = prob.space;
    opts.population_size  = 50;
    opts.generations      = 40;
    opts.tournament_size  = 4;
    opts.target_loss      = 1e-10;
    opts.simplify_expressions = true;
    opts.seed             = 7;
    opts.n_populations    = 2;
    return opts;
}

}  // namespace

int main() {
    using namespace rsymbolic;

    // --- Gate 1: get_cur_maxsize formula ----------------------------------------------
    // Warmup off: always the full cap regardless of progress.
    CHECK(get_cur_maxsize(30, 0.0, 0.0)  == 30);
    CHECK(get_cur_maxsize(30, 0.5, 0.0)  == 30);
    CHECK(get_cur_maxsize(30, 1.0, 0.0)  == 30);

    // Warmup on (fraction f, warmup_by w): cur = 3 + floor((maxsize-3)*f/w) for f <= w.
    // maxsize=30, w=0.5:
    //   f=0        -> 3
    //   f=0.25     -> 3 + floor(27*0.5)   = 3 + 13 = 16
    //   f=0.5      -> 3 + floor(27*1.0)   = 30      (ramp reaches maxsize at f == w)
    //   f>0.5      -> plateau at maxsize
    CHECK(get_cur_maxsize(30, 0.0,  0.5) == 3);
    CHECK(get_cur_maxsize(30, 0.25, 0.5) == 16);
    CHECK(get_cur_maxsize(30, 0.5,  0.5) == 30);
    CHECK(get_cur_maxsize(30, 0.75, 0.5) == 30);
    CHECK(get_cur_maxsize(30, 1.0,  0.5) == 30);

    // Clamp: a maxsize below the SR.jl start value (3) must never exceed the absolute cap,
    // and the result is always >= 1.
    CHECK(get_cur_maxsize(2, 0.0, 0.5)  == 2);
    CHECK(get_cur_maxsize(1, 0.0, 0.5)  == 1);
    CHECK(get_cur_maxsize(2, 0.5, 0.5)  == 2);

    // --- Gate 2: warmup_maxsize_by = 0 reproduces the fixed-maxsize search ----------------
    SearchOptions base = linear_opts();          // field defaults to 0.0
    SearchOptions off  = linear_opts();
    off.warmup_maxsize_by = 0.0;                 // explicitly off
    const SearchResult r_base = run_evolution(problem_linear().X, problem_linear().y, base);
    const SearchResult r_off  = run_evolution(problem_linear().X, problem_linear().y, off);
    CHECK(r_base.expression == r_off.expression);
    CHECK(r_base.loss == r_off.loss);

    // --- Gate 3: warmup-on run is valid and deterministic --------------------------------
    SearchOptions warm = linear_opts();
    warm.warmup_maxsize_by = 0.5;
    const SearchResult r_warm1 = run_evolution(problem_linear().X, problem_linear().y, warm);
    const SearchResult r_warm2 = run_evolution(problem_linear().X, problem_linear().y, warm);
    CHECK(std::isfinite(r_warm1.loss));
    CHECK(r_warm1.complexity <= warm.space.max_nodes);
    CHECK(r_warm1.expression == r_warm2.expression);   // deterministic for a fixed seed
    CHECK(r_warm1.loss == r_warm2.loss);

    std::printf("%s: %d checks, %d failures\n",
                g_failures == 0 ? "PASS" : "FAIL", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
