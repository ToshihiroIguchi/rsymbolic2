// Hot-path phase profiler driver (docs/30).
//
// Runs run_evolution() once on a representative Feynman problem at the faithful
// PySR-default gate config (benchmarks/02_feynman_gate.R BENCH_PARAMS) and reports
// where the time goes. The per-phase breakdown is emitted by run_evolution itself
// when the core is built with -DRSYMBOLIC2_PROFILE (init_fit / evolve_fit /
// evolve_sse / reopt_fit / simplify / mutate_xover / tournament / hof / migration).
// This driver adds the parallel-health numbers the per-island accumulators cannot
// give: wall time and cpu/wall (≈ thread count when healthy).
//
// The Feynman data is generated analytically (uniform sampling over the published
// domains) so the binary is self-contained. The exact RNG stream differs from R's,
// but the PHASE RATIO depends on the workload character (n_vars, n=1000, op set,
// tree sizes reached, HOF size, optimize_probability) — not on the precise points —
// so it faithfully characterises the timed-out gate workload.
//
// Usage: bench_profile [problem] [timeout_s] [seed]
//   problem   : spring_pe (recovers) | rel_mass (times out)   [default spring_pe]
//   timeout_s : wall-clock budget; phase ratio stabilises well before 300s  [default 60]
//   seed      : search seed                                                  [default 1]
//
// Build with the core compiled for profiling, e.g.:
//   cmake -S standalone -B build-prof -DCMAKE_CXX_FLAGS=-DRSYMBOLIC2_PROFILE ...
// then OMP_NUM_THREADS=4 ./build-prof/.../bench_profile rel_mass 60

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
static double process_cpu_seconds() {
    FILETIME c, e, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0.0;
    ULARGE_INTEGER ku, uu;
    ku.LowPart = k.dwLowDateTime; ku.HighPart = k.dwHighDateTime;
    uu.LowPart = u.dwLowDateTime; uu.HighPart = u.dwHighDateTime;
    return (static_cast<double>(ku.QuadPart) + static_cast<double>(uu.QuadPart)) * 1e-7;
}
#else
#  include <sys/resource.h>
static double process_cpu_seconds() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_utime.tv_usec * 1e-6 +
           r.ru_stime.tv_sec + r.ru_stime.tv_usec * 1e-6;
}
#endif

#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

using namespace rsymbolic;

namespace {

// One Feynman problem: analytic target plus per-variable sampling domains.
// Mirrors benchmarks/feynman_datasets.R (same formula and ranges).
struct FeynmanProblem {
    std::string name;
    std::string formula;
    std::vector<std::pair<double, double>> domains;          // [lo, hi] per variable
    std::function<double(const std::vector<double>&)> fn;
};

FeynmanProblem make_spring_pe() {
    // I.14.4  0.5 * k * x^2   (recovers under the gate)
    return {"spring_pe", "0.5 * k * x^2",
            {{1, 5}, {1, 5}},
            [](const std::vector<double>& v) { return 0.5 * v[0] * v[1] * v[1]; }};
}

FeynmanProblem make_rel_mass() {
    // I.10.7  m0 / sqrt(1 - (v/c)^2)   (times out under the gate)
    return {"rel_mass", "m0 / sqrt(1 - (v/c)^2)",
            {{1, 5}, {1, 2}, {3, 10}},
            [](const std::vector<double>& v) {
                return v[0] / std::sqrt(1.0 - (v[1] / v[2]) * (v[1] / v[2]));
            }};
}

// Sample n rows uniformly over the domains (fixed seed: reproducible data).
void sample(const FeynmanProblem& p, std::size_t n, std::uint64_t data_seed,
            std::vector<std::vector<double>>& X, std::vector<double>& y) {
    std::mt19937_64 rng(data_seed);
    const std::size_t d = p.domains.size();
    X.assign(n, std::vector<double>(d));
    y.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double>& row = X[i];
        for (std::size_t j = 0; j < d; ++j) {
            std::uniform_real_distribution<double> u(p.domains[j].first,
                                                     p.domains[j].second);
            row[j] = u(rng);
        }
        y[i] = p.fn(row);
    }
}

// Faithful gate configuration (benchmarks/02_feynman_gate.R BENCH_PARAMS / docs/28 §A).
SearchOptions gate_options(int num_features) {
    SearchOptions o;
    o.space.binary_ops = {BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul,
                          BinaryOp::Div, BinaryOp::Pow};
    o.space.unary_ops  = {UnaryOp::Neg, UnaryOp::Exp, UnaryOp::Log, UnaryOp::Sin,
                          UnaryOp::Cos, UnaryOp::Sqrt, UnaryOp::Tanh, UnaryOp::Abs,
                          UnaryOp::Square};
    o.space.num_features = num_features;
    o.space.max_depth = 30;
    o.space.max_nodes = 30;

    o.population_size            = 27;
    o.n_populations              = 31;
    o.generations                = 2800;
    o.tournament_size            = 15;
    o.target_loss                = 1e-10;
    o.simplify_expressions       = true;
    o.crossover_probability      = 0.0259;
    o.parsimony                  = 0.0;
    o.adaptive_parsimony_scaling = 1040.0;
    o.optimize_probability       = 0.14;
    o.tournament_selection_p     = 0.982;
    o.fraction_replaced_hof      = 0.0614;
    o.should_optimize_constants  = true;
    return o;
}

}  // namespace

int main(int argc, char* argv[]) {
    const std::string which = argc > 1 ? argv[1] : "spring_pe";
    const double timeout_s   = argc > 2 ? std::atof(argv[2]) : 60.0;
    const std::uint64_t seed = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1;

    const FeynmanProblem prob =
        (which == "rel_mass") ? make_rel_mass() : make_spring_pe();

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    sample(prob, 1000, /*data_seed=*/42, X, y);

    SearchOptions opts = gate_options(static_cast<int>(prob.domains.size()));
    opts.timeout_seconds = timeout_s;
    opts.seed = seed;

    std::printf("bench_profile  problem=%s  formula=%s\n",
                prob.name.c_str(), prob.formula.c_str());
    std::printf("config: pop=%zu islands=%zu gens=%zu tournament=%zu max_nodes=%d\n",
                opts.population_size, opts.n_populations, opts.generations,
                opts.tournament_size, opts.space.max_nodes);
    std::printf("        opt_prob=%.3f scaling=%.0f timeout=%.0fs seed=%llu  n=%zu vars=%zu\n",
                opts.optimize_probability, opts.adaptive_parsimony_scaling, timeout_s,
                static_cast<unsigned long long>(seed), y.size(), prob.domains.size());
#ifndef RSYMBOLIC2_PROFILE
    std::printf("WARNING: built without -DRSYMBOLIC2_PROFILE; no phase breakdown will print.\n");
#endif
    std::fflush(stdout);

    const double cpu0 = process_cpu_seconds();
    const auto   t0   = std::chrono::steady_clock::now();
    const SearchResult sr = run_evolution(X, y, opts);
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    const double cpu  = process_cpu_seconds() - cpu0;

    std::printf("\nresult: loss=%.4e complexity=%d\n  expr: %s\n",
                sr.loss, sr.complexity, sr.expression.c_str());
    std::printf("wall=%.2fs  cpu=%.2fs  cpu/wall=%.2f  (healthy ~= OMP_NUM_THREADS)\n",
                wall, cpu, wall > 0.0 ? cpu / wall : 0.0);
    return 0;
}
