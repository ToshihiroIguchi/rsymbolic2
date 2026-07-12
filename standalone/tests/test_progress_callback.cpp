// Progress callback (docs/53): behaviour-neutral observation of the evolving Pareto
// front. Verifies three things:
//   1. With the callback unset vs. a counting no-op callback attached, the final
//      result (expression/loss/pareto_front) is byte-identical — attaching an
//      observer cannot change the search (PySR Default Parity, CLAUDE.md).
//   2. The callback fires at least once, and never more than the number of outer
//      epochs the run could possibly take (ceil(generations / migration_interval)).
//   3. Each snapshot is internally well-formed: complexity/loss vectors have equal
//      length and complexity is strictly increasing (the Pareto-front invariant).

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "rsymbolic/search/evolutionary_search.hpp"

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

using namespace rsymbolic;

// Small clean-line problem with a multi-island config, so the run actually exercises
// the seam after the OpenMP island region has joined and ring/HOF migration has run.
void make_line(std::vector<std::vector<double>>& X, std::vector<double>& y) {
    X.clear();
    y.clear();
    for (int i = 0; i < 20; ++i) {
        const double x = static_cast<double>(i) - 10.0;  // -10 .. 9
        X.push_back({x});
        y.push_back(2.5 * x + 1.7);
    }
}

SearchOptions small_options() {
    SearchOptions o;
    o.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul};
    o.space.unary_ops = {};
    o.space.num_features = 1;
    o.space.max_depth = 3;
    o.population_size = 40;
    o.generations = 200;
    o.migration_interval = 20;  // -> at most ceil(200/20) = 10 epochs
    o.tournament_size = 4;
    o.n_populations = 4;
    o.seed = 20240603;
    o.target_loss = 1e-10;
    return o;
}

// (1) Attaching a counting no-op callback must not change the result at all.
void test_callback_is_bit_identical() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_line(X, y);

    const SearchOptions baseline = small_options();  // progress_callback left null
    const SearchResult r_base = run_evolution(X, y, baseline);

    SearchOptions observed = small_options();
    std::size_t fire_count = 0;
    observed.progress_callback = [&fire_count](const ProgressSnapshot&) { ++fire_count; };
    const SearchResult r_obs = run_evolution(X, y, observed);

    CHECK(fire_count > 0);  // sanity: the callback actually ran

    CHECK(r_base.expression == r_obs.expression);
    CHECK(r_base.loss == r_obs.loss);
    CHECK(r_base.complexity == r_obs.complexity);
    CHECK(r_base.best_index == r_obs.best_index);
    CHECK(r_base.n_evals == r_obs.n_evals);
    CHECK(r_base.n_forward_evals == r_obs.n_forward_evals);
    CHECK(r_base.n_lm_resid_evals == r_obs.n_lm_resid_evals);
    CHECK(r_base.n_lm_jac_evals == r_obs.n_lm_jac_evals);

    CHECK(r_base.pareto_front.size() == r_obs.pareto_front.size());
    const std::size_t n = std::min(r_base.pareto_front.size(), r_obs.pareto_front.size());
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(r_base.pareto_front[i].complexity == r_obs.pareto_front[i].complexity);
        CHECK(r_base.pareto_front[i].loss == r_obs.pareto_front[i].loss);
    }
}

// (2) Fire count is bounded: >= 1 and <= the maximum possible number of epochs.
void test_callback_fire_count_bounded() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_line(X, y);

    SearchOptions options = small_options();
    const std::size_t max_epochs =
        (options.generations + options.migration_interval - 1) / options.migration_interval;

    std::size_t fire_count = 0;
    options.progress_callback = [&fire_count](const ProgressSnapshot&) { ++fire_count; };
    run_evolution(X, y, options);

    std::printf("progress_callback fired %zu times (max possible %zu)\n", fire_count,
                max_epochs);
    CHECK(fire_count >= 1);
    CHECK(fire_count <= max_epochs);
}

// (3) Snapshot well-formedness: equal-length vectors, strictly increasing complexity.
void test_snapshot_shape_and_pareto_invariant() {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    make_line(X, y);

    SearchOptions options = small_options();
    bool all_lengths_equal = true;
    bool all_strictly_increasing = true;
    std::size_t snapshots_seen = 0;

    options.progress_callback = [&](const ProgressSnapshot& snap) {
        ++snapshots_seen;
        if (snap.complexity.size() != snap.loss.size()) all_lengths_equal = false;
        for (std::size_t i = 1; i < snap.complexity.size(); ++i) {
            if (snap.complexity[i] <= snap.complexity[i - 1]) all_strictly_increasing = false;
        }
    };
    run_evolution(X, y, options);

    CHECK(snapshots_seen >= 1);
    CHECK(all_lengths_equal);
    CHECK(all_strictly_increasing);
}

}  // namespace

int main() {
    test_callback_is_bit_identical();
    test_callback_fire_count_bounded();
    test_snapshot_shape_and_pareto_invariant();

    if (g_failures == 0) {
        std::printf("All %d checks passed\n", g_checks);
        return 0;
    }
    std::printf("%d/%d checks FAILED\n", g_failures, g_checks);
    return 1;
}
