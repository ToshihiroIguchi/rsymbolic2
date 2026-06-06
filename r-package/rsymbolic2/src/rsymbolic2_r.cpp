// Rcpp bridge: exposes run_evolution() to R as symbolic_regression_cpp().
// All type conversions between R and C++ happen here; the core library is
// unchanged from the standalone C++ build.

// RcppEigen must be included before Rcpp to avoid preprocessor conflicts.
#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(cpp17)]]

#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"
#include "rsymbolic/expression/tree.hpp"

using namespace Rcpp;
using namespace rsymbolic;

namespace {

UnaryOp parse_unary(const std::string& s) {
    if (s == "neg") return UnaryOp::Neg;
    if (s == "exp") return UnaryOp::Exp;
    if (s == "log") return UnaryOp::Log;
    if (s == "sin") return UnaryOp::Sin;
    if (s == "cos") return UnaryOp::Cos;
    Rcpp::stop("Unknown unary operator: '%s'. Use neg/exp/log/sin/cos.", s.c_str());
}

BinaryOp parse_binary(const std::string& s) {
    if (s == "add") return BinaryOp::Add;
    if (s == "sub") return BinaryOp::Sub;
    if (s == "mul") return BinaryOp::Mul;
    if (s == "div") return BinaryOp::Div;
    Rcpp::stop("Unknown binary operator: '%s'. Use add/sub/mul/div.", s.c_str());
}

}  // namespace

// [[Rcpp::export]]
List symbolic_regression_cpp(
    NumericMatrix X,
    NumericVector y,
    int           population_size,
    int           generations,
    int           tournament_size,
    CharacterVector unary_ops,
    CharacterVector binary_ops,
    int           max_depth,
    int           max_nodes,
    double        target_loss,
    bool          simplify,
    double        crossover_probability,
    double        seed,
    int           n_populations,
    double        timeout_seconds,
    int           verbosity,
    double        optimize_probability
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
        space.unary_ops.push_back(parse_unary(Rcpp::as<std::string>(s)));
    space.binary_ops.clear();
    for (const auto& s : binary_ops)
        space.binary_ops.push_back(parse_binary(Rcpp::as<std::string>(s)));
    if (space.binary_ops.empty())
        Rcpp::stop("binary_ops must contain at least one operator");

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

    const SearchResult res = run_evolution(X_cpp, y_cpp, opts);

    // Pareto front → data.frame
    IntegerVector    pf_complexity;
    NumericVector    pf_loss;
    CharacterVector  pf_expr;
    for (const auto& m : res.pareto_front) {
        pf_complexity.push_back(m.complexity);
        pf_loss.push_back(m.loss);
        pf_expr.push_back(to_string(m.tree));
    }

    return List::create(
        Named("expression")   = res.expression,
        Named("loss")         = res.loss,
        Named("complexity")   = res.complexity,
        Named("pareto_front") = DataFrame::create(
            Named("complexity")  = pf_complexity,
            Named("loss")        = pf_loss,
            Named("expression")  = pf_expr,
            Named("stringsAsFactors") = false
        )
    );
}
