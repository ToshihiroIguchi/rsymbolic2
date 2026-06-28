// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

// cpp11 bridge: exposes run_evolution() to R as symbolic_regression_cpp().
// All type conversions between R and C++ happen here; the core library is
// unchanged from the standalone C++ build. cpp11 (MIT) is used instead of Rcpp
// so the shipped R package carries no GPL build dependency, keeping it
// consistent with the project's Apache-2.0 licensing.

#include <cpp11.hpp>

#include "rsymbolic/evolution/mutation_weights.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"
#include "rsymbolic/expression/tree.hpp"

using namespace cpp11;
using namespace cpp11::literals;
using namespace rsymbolic;

namespace {

// Overlay a named numeric vector (e.g. c(insert_node = 6, rotate_tree = 5)) onto a
// MutationWeights, leaving unnamed/absent fields at their PySR defaults. Unknown names
// are rejected so typos surface immediately rather than silently doing nothing.
MutationWeights parse_mutation_weights(const cpp11::doubles& w) {
    MutationWeights mw;
    if (w.size() == 0) return mw;
    const cpp11::strings names = w.names();
    if (names.size() != w.size())
        cpp11::stop("mutation_weights must be a *named* numeric vector");
    for (R_xlen_t i = 0; i < w.size(); ++i) {
        const std::string key = static_cast<std::string>(names[i]);
        const double v = w[i];
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
        else cpp11::stop("Unknown mutation weight name: '%s'.", key.c_str());
    }
    return mw;
}

UnaryOp parse_unary(const std::string& s) {
    if (s == "neg")  return UnaryOp::Neg;
    if (s == "exp")  return UnaryOp::Exp;
    if (s == "log")  return UnaryOp::Log;
    if (s == "sin")  return UnaryOp::Sin;
    if (s == "cos")  return UnaryOp::Cos;
    if (s == "sqrt")   return UnaryOp::Sqrt;
    if (s == "tanh")   return UnaryOp::Tanh;
    if (s == "abs")    return UnaryOp::Abs;
    if (s == "square") return UnaryOp::Square;
    cpp11::stop(
        "Unknown unary operator: '%s'. Use neg/exp/log/sin/cos/sqrt/tanh/abs/square.",
        s.c_str());
}

BinaryOp parse_binary(const std::string& s) {
    if (s == "add") return BinaryOp::Add;
    if (s == "sub") return BinaryOp::Sub;
    if (s == "mul") return BinaryOp::Mul;
    if (s == "div") return BinaryOp::Div;
    if (s == "pow") return BinaryOp::Pow;
    cpp11::stop("Unknown binary operator: '%s'. Use add/sub/mul/div/pow.", s.c_str());
}

ModelSelection parse_model_selection(const std::string& s) {
    if (s == "best")     return ModelSelection::Best;
    if (s == "accuracy") return ModelSelection::Accuracy;
    if (s == "score")    return ModelSelection::Score;
    cpp11::stop("Unknown model_selection: '%s'. Use best/accuracy/score.", s.c_str());
}

}  // namespace

[[cpp11::register]]
cpp11::writable::list symbolic_regression_cpp(
    cpp11::doubles_matrix<> X,
    cpp11::doubles  y,
    int             population_size,
    int             generations,
    int             tournament_size,
    cpp11::strings  unary_ops,
    cpp11::strings  binary_ops,
    int             max_depth,
    int             max_nodes,
    double          target_loss,
    bool            simplify,
    double          crossover_probability,
    double          seed,
    int             n_populations,
    double          timeout_seconds,
    int             verbosity,
    double          optimize_probability,
    double          parsimony,
    double          adaptive_parsimony_scaling,
    double          tournament_selection_p,
    bool            should_optimize_constants,
    double          fraction_replaced_hof,
    cpp11::doubles  mutation_weights,
    std::string     model_selection,
    double          max_evals,
    double          early_stop_condition,
    cpp11::doubles  weights,
    bool            batching,
    int             batch_size,
    double          warmup_maxsize_by
) {
    // Convert R matrix → vector<vector<double>> (row-major)
    const int n = X.nrow();
    const int p = X.ncol();
    std::vector<std::vector<double>> X_cpp(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        X_cpp[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
        for (int j = 0; j < p; ++j)
            X_cpp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = X(i, j);
    }
    const std::vector<double> y_cpp(y.begin(), y.end());

    // Build SearchSpace from R arguments
    SearchSpace space;
    space.num_features = p;
    space.max_depth    = max_depth;
    space.max_nodes    = max_nodes;
    for (const auto& s : unary_ops)
        space.unary_ops.push_back(parse_unary(static_cast<std::string>(s)));
    space.binary_ops.clear();
    for (const auto& s : binary_ops)
        space.binary_ops.push_back(parse_binary(static_cast<std::string>(s)));
    if (space.binary_ops.empty())
        cpp11::stop("binary_ops must contain at least one operator");

    SearchOptions opts;
    opts.space                 = space;
    opts.population_size       = static_cast<std::size_t>(population_size);
    opts.generations           = static_cast<std::size_t>(generations);
    opts.tournament_size       = static_cast<std::size_t>(tournament_size);
    opts.target_loss           = target_loss;
    opts.simplify_expressions  = simplify;
    opts.crossover_probability = crossover_probability;
    opts.seed                  = static_cast<std::uint64_t>(seed);
    opts.n_populations         = static_cast<std::size_t>(std::max(1, n_populations));
    opts.timeout_seconds       = timeout_seconds;
    opts.verbosity             = verbosity;
    opts.optimize_probability  = optimize_probability;
    opts.parsimony             = parsimony;
    opts.adaptive_parsimony_scaling = adaptive_parsimony_scaling;
    opts.tournament_selection_p = tournament_selection_p;
    opts.should_optimize_constants = should_optimize_constants;
    opts.fraction_replaced_hof = fraction_replaced_hof;
    opts.mutation_weights      = parse_mutation_weights(mutation_weights);
    opts.model_selection       = parse_model_selection(model_selection);
    opts.early_stop_condition  = early_stop_condition;
    // PySR batching: subsample batch_size rows for the evolution/optimisation passes. The
    // hall of fame and final result are still computed on the full data (see SearchOptions).
    opts.batching              = batching;
    opts.batch_size            = static_cast<std::size_t>(std::max(1, batch_size));
    // PySR warmup_maxsize_by: 0 = off (fixed maxsize). Negative is rejected on the R side.
    opts.warmup_maxsize_by     = warmup_maxsize_by;
    // max_evals arrives as a double (R has no native 64-bit int); negative/zero => off.
    opts.max_evals = max_evals > 0.0
        ? static_cast<std::size_t>(max_evals)
        : 0;
    // Optional per-point weights for weighted least squares. Length 0 => unweighted; any
    // other length must equal the number of observations.
    if (weights.size() > 0) {
        if (weights.size() != y.size())
            cpp11::stop("weights must have length nrow(X) (= length(y)); got %d, expected %d.",
                        static_cast<int>(weights.size()), static_cast<int>(y.size()));
        opts.weights.assign(weights.begin(), weights.end());
    }

    const SearchResult res = run_evolution(X_cpp, y_cpp, opts);

    // Pareto front → data.frame
    cpp11::writable::integers pf_complexity;
    cpp11::writable::doubles  pf_loss;
    cpp11::writable::strings  pf_expr;
    for (const auto& m : res.pareto_front) {
        pf_complexity.push_back(m.complexity);
        pf_loss.push_back(m.loss);
        pf_expr.push_back(to_string(m.tree));
    }

    // Recommended ("best") accuracy/complexity trade-off from the Pareto front
    // (PySR model_selection="best"). best_index is 0-based in C++; expose it 1-based
    // for R. Fall back to the overall-best expression when the front is empty.
    const int n_front = static_cast<int>(res.pareto_front.size());
    std::string recommended = res.expression;
    int best_index_r = (n_front > 0) ? (res.best_index + 1) : NA_INTEGER;
    if (res.best_index >= 0 && res.best_index < n_front)
        recommended = to_string(res.pareto_front[static_cast<std::size_t>(res.best_index)].tree);

    cpp11::writable::data_frame pareto_front({
        "complexity"_nm = pf_complexity,
        "loss"_nm       = pf_loss,
        "expression"_nm = pf_expr
    });

    cpp11::writable::list result({
        "expression"_nm   = res.expression,
        "loss"_nm         = res.loss,
        "complexity"_nm   = res.complexity,
        "recommended"_nm  = recommended,
        "best_index"_nm   = best_index_r,
        "pareto_front"_nm = pareto_front
    });
    result.attr("class") = "rsymbolic2";
    return result;
}
