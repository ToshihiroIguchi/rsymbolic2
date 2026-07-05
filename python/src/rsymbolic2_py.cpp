// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

// pybind11 bridge: exposes run_evolution() to Python as symbolic_regression_cpp().
//
// This file is the Python counterpart of r-package/rsymbolic2/src/rsymbolic2_r.cpp.
// Both bindings are thin translation layers over the SAME C++ core
// (r-package/rsymbolic2/src/rsymbolic/...); the search engine itself is shared and
// unchanged. Keeping the two bridges symmetric guarantees that R and Python see
// identical defaults and behaviour — only the host-language marshalling differs.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rsymbolic/evolution/mutation_weights.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"
#include "rsymbolic/expression/latex.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/units/unit_parser.hpp"

namespace py = pybind11;
using namespace rsymbolic;

namespace {

// Overlay a {name: weight} dict onto a MutationWeights, leaving absent fields at their
// PySR defaults. Unknown names are rejected so typos surface immediately. Mirrors
// parse_mutation_weights() in the R bridge.
MutationWeights parse_mutation_weights(const py::dict& w) {
    MutationWeights mw;
    for (auto item : w) {
        const std::string key = py::cast<std::string>(item.first);
        const double v = py::cast<double>(item.second);
        if      (key == "mutate_constant") mw.mutate_constant = v;
        else if (key == "mutate_operator") mw.mutate_operator = v;
        else if (key == "swap_operands")   mw.swap_operands   = v;
        else if (key == "rotate_tree")     mw.rotate_tree     = v;
        else if (key == "add_node")        mw.add_node        = v;
        else if (key == "insert_node")     mw.insert_node     = v;
        else if (key == "delete_node")     mw.delete_node     = v;
        else if (key == "do_nothing")      mw.do_nothing      = v;
        else if (key == "simplify")        mw.simplify        = v;
        else if (key == "randomize")       mw.randomize       = v;
        else throw std::invalid_argument("Unknown mutation weight name: '" + key + "'.");
    }
    return mw;
}

UnaryOp parse_unary(const std::string& s) {
    if (s == "neg")    return UnaryOp::Neg;
    if (s == "exp")    return UnaryOp::Exp;
    if (s == "log")    return UnaryOp::Log;
    if (s == "sin")    return UnaryOp::Sin;
    if (s == "cos")    return UnaryOp::Cos;
    if (s == "sqrt")   return UnaryOp::Sqrt;
    if (s == "tanh")   return UnaryOp::Tanh;
    if (s == "abs")    return UnaryOp::Abs;
    if (s == "square") return UnaryOp::Square;
    throw std::invalid_argument(
        "Unknown unary operator: '" + s +
        "'. Use neg/exp/log/sin/cos/sqrt/tanh/abs/square.");
}

BinaryOp parse_binary(const std::string& s) {
    if (s == "add") return BinaryOp::Add;
    if (s == "sub") return BinaryOp::Sub;
    if (s == "mul") return BinaryOp::Mul;
    if (s == "div") return BinaryOp::Div;
    if (s == "pow") return BinaryOp::Pow;
    throw std::invalid_argument("Unknown binary operator: '" + s +
                                "'. Use add/sub/mul/div/pow.");
}

ModelSelection parse_model_selection(const std::string& s) {
    if (s == "best")     return ModelSelection::Best;
    if (s == "accuracy") return ModelSelection::Accuracy;
    if (s == "score")    return ModelSelection::Score;
    throw std::invalid_argument("Unknown model_selection: '" + s +
                                "'. Use best/accuracy/score.");
}

py::dict symbolic_regression_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> X,
    py::array_t<double, py::array::c_style | py::array::forcecast> y,
    int                      population_size,
    int                      generations,
    int                      tournament_size,
    std::vector<std::string> unary_ops,
    std::vector<std::string> binary_ops,
    int                      max_depth,
    int                      max_nodes,
    double                   target_loss,
    bool                     simplify,
    double                   crossover_probability,
    double                   seed,
    int                      n_populations,
    double                   timeout_seconds,
    int                      verbosity,
    double                   optimize_probability,
    double                   parsimony,
    double                   adaptive_parsimony_scaling,
    double                   tournament_selection_p,
    bool                     should_optimize_constants,
    double                   fraction_replaced_hof,
    py::dict                 mutation_weights,
    std::string              model_selection,
    double                   max_evals,
    double                   early_stop_condition,
    py::array_t<double, py::array::c_style | py::array::forcecast> weights,
    bool                     batching,
    int                      batch_size,
    double                   warmup_maxsize_by,
    int                      n_threads,
    std::vector<std::string> X_units,
    std::string              y_units,
    double                   dimensional_constraint_penalty,
    bool                     dimensionless_constants_only,
    bool                     eval_cache,
    bool                     linear_scaling) {
    // --- Marshal X (2-D) and y (1-D) ------------------------------------------------
    if (X.ndim() != 2)
        throw std::invalid_argument("X must be a 2-D array (rows = observations, "
                                    "columns = features).");
    if (y.ndim() != 1)
        throw std::invalid_argument("y must be a 1-D array.");

    const std::size_t n = static_cast<std::size_t>(X.shape(0));
    const std::size_t p = static_cast<std::size_t>(X.shape(1));
    if (static_cast<std::size_t>(y.shape(0)) != n)
        throw std::invalid_argument("nrow(X) must equal length(y).");
    if (n == 0)
        throw std::invalid_argument("X must have at least one row.");

    auto Xv = X.unchecked<2>();
    std::vector<std::vector<double>> X_cpp(n);
    for (std::size_t i = 0; i < n; ++i) {
        X_cpp[i].resize(p);
        for (std::size_t j = 0; j < p; ++j)
            X_cpp[i][j] = Xv(i, j);
    }
    auto yv = y.unchecked<1>();
    std::vector<double> y_cpp(n);
    for (std::size_t i = 0; i < n; ++i) y_cpp[i] = yv(i);

    // --- Build SearchSpace ----------------------------------------------------------
    SearchSpace space;
    space.num_features = static_cast<int>(p);
    space.max_depth    = max_depth;
    space.max_nodes    = max_nodes;
    for (const auto& s : unary_ops) space.unary_ops.push_back(parse_unary(s));
    space.binary_ops.clear();
    for (const auto& s : binary_ops) space.binary_ops.push_back(parse_binary(s));
    if (space.binary_ops.empty())
        throw std::invalid_argument("binary_ops must contain at least one operator.");

    // Opt-in dimensional analysis (PySR X_units / y_units / dimensionless_constants_only;
    // docs/46). Empty X_units leaves the feature off (default). parse_unit throws
    // std::invalid_argument (-> Python ValueError) naming a bad unit token. Mirrors the R
    // bridge exactly so R and Python parse units identically.
    if (!X_units.empty()) {
        if (X_units.size() != p)
            throw std::invalid_argument(
                "X_units must have length ncol(X) (= " + std::to_string(p) + ").");
        for (const auto& s : X_units) space.x_units.push_back(parse_unit(s));
    }
    if (!y_units.empty()) space.y_units = parse_unit(y_units);
    space.dimensionless_constants_only = dimensionless_constants_only;

    // --- Build SearchOptions (defaults already match PySR; we only override what the
    //     caller passed). Field-for-field identical to the R bridge. -----------------
    SearchOptions opts;
    opts.space                      = space;
    opts.population_size            = static_cast<std::size_t>(population_size);
    opts.generations                = static_cast<std::size_t>(generations);
    opts.tournament_size            = static_cast<std::size_t>(tournament_size);
    opts.target_loss                = target_loss;
    opts.simplify_expressions       = simplify;
    opts.crossover_probability      = crossover_probability;
    opts.seed                       = static_cast<std::uint64_t>(seed);
    opts.n_populations              = static_cast<std::size_t>(std::max(1, n_populations));
    opts.timeout_seconds            = timeout_seconds;
    opts.verbosity                  = verbosity;
    opts.optimize_probability       = optimize_probability;
    opts.parsimony                  = parsimony;
    opts.adaptive_parsimony_scaling = adaptive_parsimony_scaling;
    opts.tournament_selection_p     = tournament_selection_p;
    opts.should_optimize_constants  = should_optimize_constants;
    opts.fraction_replaced_hof      = fraction_replaced_hof;
    opts.mutation_weights           = parse_mutation_weights(mutation_weights);
    opts.model_selection            = parse_model_selection(model_selection);
    opts.early_stop_condition       = early_stop_condition;
    // PySR batching: subsample batch_size rows for the evolution/optimisation passes; the
    // hall of fame and final result stay full-data (see SearchOptions).
    opts.batching                   = batching;
    opts.batch_size                 = static_cast<std::size_t>(std::max(1, batch_size));
    // PySR warmup_maxsize_by: 0 = off (fixed maxsize); the Python wrapper rejects negatives.
    opts.warmup_maxsize_by          = warmup_maxsize_by;
    // OpenMP team size. 0 (the Python wrapper's None default) = auto (all cores, honouring
    // OMP_NUM_THREADS); positive = that many island workers, capped at n_populations. The
    // Python wrapper rejects non-positive non-None values, so any value here is 0 or positive.
    opts.n_threads                  = n_threads;
    // Opt-in dimensional-constraint penalty (PySR default 1000, resolved in the Python
    // wrapper). Inert unless X_units/y_units are set; the units live on `space` above.
    opts.dimensional_constraint_penalty = dimensional_constraint_penalty;
    // Opt-in duplicate-evaluation cache (implementation-only memoisation; results are
    // bit-identical on/off, and it is ignored when batching is on). Default False.
    opts.eval_cache                 = eval_cache;
    // Opt-in Keijzer-2003 linear scaling (behaviour-changing high-accuracy option;
    // default False keeps exact PySR parity). The Python wrapper rejects combining it
    // with X_units/y_units, so the core never sees that (undefined) combination.
    opts.linear_scaling             = linear_scaling;
    // max_evals arrives as a double (mirrors the R bridge, where R has no 64-bit int);
    // negative/zero => off.
    opts.max_evals = max_evals > 0.0 ? static_cast<std::size_t>(max_evals) : 0;

    // Optional per-point weights for weighted least squares. Length 0 => unweighted.
    if (weights.size() > 0) {
        if (static_cast<std::size_t>(weights.shape(0)) != n)
            throw std::invalid_argument(
                "weights must have length nrow(X) (= length(y)).");
        auto wv = weights.unchecked<1>();
        opts.weights.resize(n);
        for (std::size_t i = 0; i < n; ++i) opts.weights[i] = wv(i);
    }

    // --- Run the search (release the GIL: this is a long pure-C++ computation that
    //     touches no Python objects). ------------------------------------------------
    SearchResult res;
    {
        py::gil_scoped_release release;
        res = run_evolution(X_cpp, y_cpp, opts);
    }

    // Weighted total sum of squares about the weighted mean (unit weights when
    // unweighted), for downstream fit statistics: R^2 = 1 - loss / sst is consistent
    // with the (weighted) SSE loss reported per Pareto member.
    double wsum = 0.0, wysum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = opts.weights.empty() ? 1.0 : opts.weights[i];
        wsum  += w;
        wysum += w * y_cpp[i];
    }
    const double ybar = (wsum > 0.0) ? (wysum / wsum) : 0.0;
    double sst = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = opts.weights.empty() ? 1.0 : opts.weights[i];
        const double d = y_cpp[i] - ybar;
        sst += w * d * d;
    }

    // --- Pareto front -> parallel lists ---------------------------------------------
    const std::vector<double> scores = pareto_scores(res.pareto_front);
    py::list pf_complexity, pf_loss, pf_score, pf_expr, pf_latex;
    for (std::size_t i = 0; i < res.pareto_front.size(); ++i) {
        const auto& m = res.pareto_front[i];
        pf_complexity.append(m.complexity);
        pf_loss.append(m.loss);
        pf_score.append(scores[i]);
        pf_expr.append(to_string(m.tree));
        pf_latex.append(to_latex(m.tree));
    }

    // Recommended ("best") trade-off; best_index is 0-based in C++ and is kept 0-based
    // in Python (unlike the R bridge, which exposes it 1-based to match R indexing).
    const int n_front = static_cast<int>(res.pareto_front.size());
    std::string recommended = res.expression;
    py::object best_index = py::none();
    if (n_front > 0) {
        best_index = py::int_(res.best_index);
        if (res.best_index >= 0 && res.best_index < n_front)
            recommended =
                to_string(res.pareto_front[static_cast<std::size_t>(res.best_index)].tree);
    }

    py::dict result;
    result["expression"]   = res.expression;
    result["loss"]         = res.loss;
    result["complexity"]   = res.complexity;
    result["recommended"]  = recommended;
    result["best_index"]   = best_index;
    result["n_obs"]        = static_cast<int>(n);
    result["sst"]          = sst;
    // Evaluation accounting (reporting only): n_evals = forward + lm_resid (max_evals
    // units); Jacobian builds are reported but never charged to n_evals. Mirrors the R
    // bridge's n_evals / eval_counts fields (Python ints carry 64-bit counts exactly).
    result["n_evals"]      = py::int_(res.n_evals);
    py::dict eval_counts;
    eval_counts["forward"]      = py::int_(res.n_forward_evals);
    eval_counts["lm_resid"]     = py::int_(res.n_lm_resid_evals);
    eval_counts["lm_jac"]       = py::int_(res.n_lm_jac_evals);
    // Duplicate-evaluation cache statistics (both 0 unless eval_cache is on).
    eval_counts["cache_hits"]   = py::int_(res.cache_hits);
    eval_counts["cache_misses"] = py::int_(res.cache_misses);
    result["eval_counts"]  = eval_counts;
    result["pareto_front"] = py::dict(
        py::arg("complexity") = pf_complexity,
        py::arg("loss")       = pf_loss,
        py::arg("score")      = pf_score,
        py::arg("expression") = pf_expr,
        py::arg("latex")      = pf_latex);
    return result;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "rsymbolic2 C++ core (pybind11 bridge). Use rsymbolic2.symbolic_regression.";
    m.def("symbolic_regression_cpp", &symbolic_regression_cpp,
          "Low-level entry point; prefer rsymbolic2.symbolic_regression().");
}
