// Phase 0-C "seed-expression oracle" diagnostic driver (docs/43 follow-up).
//
// Purpose: isolate whether a hard-to-recover Feynman target is EXPLORATION-bound (the
// search never finds the right backbone) or POLISH-bound (it finds the backbone but
// can't fit/simplify it to a low-NMSE expression within the generation budget). This
// driver injects a hand-built "seed" expression as member 0 of every island's initial
// population (via SearchOptions::seed_trees, evolutionary_search.hpp) and otherwise runs
// the exact Stage-1 parity search — if a seed close to (or equal to) the true structure
// still fails to converge quickly, the bottleneck is polish/optimisation, not discovery.
//
// This is a throwaway diagnostic, not a shipped feature: seed_trees is an internal hook
// not exposed via the R/Python bindings (empty = default byte-identical search path).
//
// Usage:
//   diag_seed_oracle <key> <arm> <seeds> <gens> <csv_path>
//     key   : interference | planck | bose_einstein
//     arm   : baseline | (problem-specific partial/full seed arms; see seed_trees_for)
//     seeds : number of seeds to run (seed = 1..N)
//     gens  : generations (rsymbolic2 SearchOptions::generations)
//     csv_path : output CSV, append mode; header written once if the file is new
//
// Search configuration: a verbatim copy of the Stage-1 parity BENCH_PARAMS in
// benchmarks/diag_interference.R, mapped onto SearchOptions the same way the R bridge
// (r-package/rsymbolic2/src/rsymbolic2_r.cpp) maps them. Fields not set here (n_threads,
// verbosity, weights, batching, warmup_maxsize_by, model_selection, mutation_weights,
// max_evals, early_stop_condition, should_optimize_constants) are left at SearchOptions'
// own defaults, which are exactly the R wrapper's own defaults too
// (r-package/rsymbolic2/R/symbolic_regression.R:247-278) — so this driver reproduces the
// gate's search behaviour with no silent divergence.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/node.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

using namespace rsymbolic;

namespace {

// ---- CSV data loading ---------------------------------------------------------------

struct RawDataset {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

// Minimal reader for the exported gate training files (benchmarks/data/*.csv): a header
// row (x1..xn,y, optionally quoted, discarded here since column order is fixed by the
// export) followed by plain comma-separated numeric rows, last column = y.
RawDataset read_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "diag_seed_oracle: cannot open data file: " << path << "\n";
        std::exit(1);
    }
    std::string line;
    std::getline(f, line);  // header, discarded

    RawDataset ds;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
        if (row.empty()) continue;
        ds.y.push_back(row.back());
        row.pop_back();
        ds.X.push_back(std::move(row));
    }
    return ds;
}

// ---- Seed-tree construction -----------------------------------------------------------
// Postfix (reverse-Polish) node sequences, built with the same Node-builder idiom as
// standalone/tests/test_simplify.cpp. Constant parameter indices are placeholder (0) and
// fixed up by finish() -> reindex_constants() after assembly. Variable indices are
// 0-indexed features matching the CSV columns x1..xn (x1 -> V(0), x2 -> V(1), ...).

Node C(double v)    { return constant_node(0, v); }
Node V(int i)       { return variable_node(i); }
Node U(UnaryOp op)  { return unary_node(op); }
Node B(BinaryOp op) { return binary_node(op); }

Tree finish(Tree nodes) {
    reindex_constants(nodes);
    return nodes;
}

void append(Tree& dst, const Tree& src) { dst.insert(dst.end(), src.begin(), src.end()); }

// exp((a*b)/(c*d)) - 1.0 — the shared "exponential denominator" subtree used by both the
// planck and bose_einstein arms.
Tree exp_ratio_minus_one(int a, int b, int c, int d) {
    return { V(a), V(b), B(BinaryOp::Mul), V(c), V(d), B(BinaryOp::Mul),
             B(BinaryOp::Div), U(UnaryOp::Exp), C(1.0), B(BinaryOp::Sub) };
}

// interference: x1=I1 (V0), x2=I2 (V1), x3=delta (V2). Target: I1 + I2 + 2*sqrt(I1*I2)*cos(delta).
Tree seed_interference_sqrt_prod() {
    return finish({ V(0), V(1), B(BinaryOp::Mul), U(UnaryOp::Sqrt) });
}
Tree seed_interference_sqrt_cos() {
    return finish({ C(1.0), V(0), V(1), B(BinaryOp::Mul), U(UnaryOp::Sqrt), B(BinaryOp::Mul),
                    V(2), U(UnaryOp::Cos), B(BinaryOp::Mul) });
}
Tree seed_interference_full() {
    return finish({ V(0), V(1), B(BinaryOp::Add),
                    C(2.0), V(0), V(1), B(BinaryOp::Mul), U(UnaryOp::Sqrt), B(BinaryOp::Mul),
                    V(2), U(UnaryOp::Cos), B(BinaryOp::Mul),
                    B(BinaryOp::Add) });
}

// planck: x1=hbar (V0), x2=omega (V1), x3=c (V2), x4=k (V3), x5=T (V4).
// Target: hbar*omega^3 / (pi^2 c^2 (exp(hbar*omega/(k*T)) - 1)).
Tree seed_planck_expden() { return finish(exp_ratio_minus_one(0, 1, 3, 4)); }
Tree seed_planck_recip() {
    Tree t{ C(1.0) };
    append(t, exp_ratio_minus_one(0, 1, 3, 4));
    t.push_back(B(BinaryOp::Div));
    return finish(t);
}
Tree seed_planck_full() {
    // numerator = x1 * (1/pi^2) * x2 * x2 * x2  (plain mul chain: order is immaterial to value)
    Tree t{ V(0), C(0.1013211836), B(BinaryOp::Mul),
            V(1), B(BinaryOp::Mul), V(1), B(BinaryOp::Mul), V(1), B(BinaryOp::Mul) };
    // denominator = x3*x3 * (exp((x1*x2)/(x4*x5)) - 1.0)
    t.push_back(V(2));
    t.push_back(V(2));
    t.push_back(B(BinaryOp::Mul));
    append(t, exp_ratio_minus_one(0, 1, 3, 4));
    t.push_back(B(BinaryOp::Mul));
    t.push_back(B(BinaryOp::Div));
    return finish(t);
}

// bose_einstein: x1=hbar (V0), x2=omega (V1), x3=kB (V2), x4=T (V3).
// Target: hbar*omega / (exp(hbar*omega/(kB*T)) - 1).
Tree seed_bose_expden() { return finish(exp_ratio_minus_one(0, 1, 2, 3)); }
Tree seed_bose_recip() {
    Tree t{ C(1.0) };
    append(t, exp_ratio_minus_one(0, 1, 2, 3));
    t.push_back(B(BinaryOp::Div));
    return finish(t);
}
Tree seed_bose_full() {
    Tree t{ V(0), V(1), B(BinaryOp::Mul) };
    append(t, exp_ratio_minus_one(0, 1, 2, 3));
    t.push_back(B(BinaryOp::Div));
    return finish(t);
}

// Empty vector = baseline (no seed injected; every island's initial population is fully
// random, exactly the default search path). Non-empty = one seed tree, injected as member
// 0 of every island (SearchOptions::seed_trees, evolutionary_search.hpp).
std::vector<Tree> seed_trees_for(const std::string& key, const std::string& arm) {
    if (arm == "baseline") return {};
    if (key == "interference") {
        if (arm == "sqrt_prod") return { seed_interference_sqrt_prod() };
        if (arm == "sqrt_cos")  return { seed_interference_sqrt_cos() };
        if (arm == "full")      return { seed_interference_full() };
    } else if (key == "planck") {
        if (arm == "expden") return { seed_planck_expden() };
        if (arm == "recip")  return { seed_planck_recip() };
        if (arm == "full")   return { seed_planck_full() };
    } else if (key == "bose_einstein") {
        if (arm == "expden") return { seed_bose_expden() };
        if (arm == "recip")  return { seed_bose_recip() };
        if (arm == "full")   return { seed_bose_full() };
    }
    std::cerr << "diag_seed_oracle: unknown key/arm combination: " << key << "/" << arm << "\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: diag_seed_oracle <key> <arm> <seeds> <gens> <csv_path>\n"
                     "  key in {interference, planck, bose_einstein}\n";
        return 1;
    }
    const std::string key      = argv[1];
    const std::string arm      = argv[2];
    const int n_seeds          = std::atoi(argv[3]);
    const int gens             = std::atoi(argv[4]);
    const std::string out_csv  = argv[5];
    if (n_seeds < 1 || gens < 1) {
        std::cerr << "diag_seed_oracle: seeds and gens must be >= 1\n";
        return 1;
    }

    int num_features = 0;
    std::string data_file;
    if (key == "interference") {
        num_features = 3;
        data_file = "feynman_I_interference_train.csv";
    } else if (key == "planck") {
        num_features = 5;
        data_file = "feynman_I_planck_train.csv";
    } else if (key == "bose_einstein") {
        num_features = 4;
        data_file = "feynman_III_bose_einstein_train.csv";
    } else {
        std::cerr << "diag_seed_oracle: unknown key: " << key << "\n";
        return 1;
    }

    const RawDataset ds = read_csv("benchmarks/data/" + data_file);
    if (ds.X.empty() || static_cast<int>(ds.X.front().size()) != num_features) {
        std::cerr << "diag_seed_oracle: unexpected column count in " << data_file << "\n";
        return 1;
    }

    const std::vector<Tree> seeds = seed_trees_for(key, arm);

    // NMSE denominator = benchmarks/utils.R compute_nmse: loss / ((n-1)*var(y)), and
    // (n-1)*var(y) == sum((y-mean(y))^2) exactly (var(y) = ss/(n-1)).
    const std::size_t n = ds.y.size();
    double mean_y = 0.0;
    for (double v : ds.y) mean_y += v;
    mean_y /= static_cast<double>(n);
    double nmse_denom = 0.0;
    for (double v : ds.y) nmse_denom += (v - mean_y) * (v - mean_y);

    // Stage-1 parity search configuration: verbatim BENCH_PARAMS from
    // benchmarks/diag_interference.R, mapped onto SearchOptions field-for-field the way
    // rsymbolic2_r.cpp / symbolic_regression.R does. Every field not listed here keeps
    // SearchOptions' own default, which is exactly the R wrapper's default too.
    SearchSpace space;
    space.num_features = num_features;
    space.max_depth    = 30;
    space.max_nodes     = 30;
    space.unary_ops  = { UnaryOp::Exp, UnaryOp::Log, UnaryOp::Sin, UnaryOp::Cos,
                         UnaryOp::Sqrt, UnaryOp::Tanh, UnaryOp::Abs, UnaryOp::Square };
    space.binary_ops = { BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div,
                         BinaryOp::Pow };

    SearchOptions opts;
    opts.space                      = space;
    opts.population_size            = 27;
    opts.n_populations               = 31;
    opts.generations                = static_cast<std::size_t>(gens);
    opts.tournament_size            = 15;
    opts.target_loss                = 1e-10;
    opts.simplify_expressions       = true;
    opts.crossover_probability      = 0.0259;
    opts.parsimony                  = 0.0;
    opts.adaptive_parsimony_scaling = 1040.0;
    opts.optimize_probability       = 0.14;
    opts.tournament_selection_p     = 0.982;
    opts.fraction_replaced_hof      = 0.0614;
    opts.timeout_seconds            = (gens > 2800) ? 900.0 : 300.0;
    opts.seed_trees                 = seeds;
    // n_threads = 0 (auto team size) and verbosity = 0 (silent) are SearchOptions'
    // defaults already; not overridden, so the island team uses all available threads
    // (bridge default) and this driver does its own progress printing below instead.

    std::ifstream probe(out_csv);
    const bool need_header = !probe.good();
    probe.close();
    std::ofstream out(out_csv, std::ios::app);
    if (!out) {
        std::cerr << "diag_seed_oracle: cannot open output csv: " << out_csv << "\n";
        return 1;
    }
    if (need_header) out << "key,arm,seed,gens,elapsed_sec,loss,nmse,recovered,expression\n";

    for (int seed = 1; seed <= n_seeds; ++seed) {
        opts.seed = static_cast<std::uint64_t>(seed);

        const auto t0 = std::chrono::steady_clock::now();
        const SearchResult res = run_evolution(ds.X, ds.y, opts);
        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        const double nmse = (nmse_denom > 0.0)
            ? (res.loss / nmse_denom)
            : std::numeric_limits<double>::infinity();
        const bool recovered = std::isfinite(nmse) && nmse < 1e-4;

        out << key << ',' << arm << ',' << seed << ',' << gens << ','
            << std::fixed << std::setprecision(3) << elapsed << ','
            << std::defaultfloat << std::setprecision(17) << res.loss << ','
            << nmse << ','
            << (recovered ? "TRUE" : "FALSE") << ",\"" << res.expression << "\"\n";
        out.flush();

        std::printf(
            "key=%s arm=%s seed=%d gens=%d elapsed=%.1fs loss=%.3e nmse=%.3e %s\n",
            key.c_str(), arm.c_str(), seed, gens, elapsed, res.loss, nmse,
            recovered ? "RECOVERED" : "not recovered");
    }

    return 0;
}
