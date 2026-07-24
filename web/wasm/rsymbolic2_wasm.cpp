// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

// Emscripten/embind bridge: exposes run_evolution() to JavaScript as
// Module.run(options) for the static-site web GUI.
//
// This file is the WebAssembly counterpart of python/src/rsymbolic2_py.cpp and
// r-package/rsymbolic2/src/rsymbolic2_r.cpp. All three bridges are thin translation
// layers over the SAME C++ core (r-package/rsymbolic2/src/rsymbolic/...); the search
// engine itself is shared and unchanged. Keeping the bridges symmetric guarantees that
// R, Python and the browser see identical defaults and behaviour — only the host
// marshalling differs. Built single-threaded (no OpenMP); deterministic and reproducible
// for a fixed seed within this build. The returned expression is NOT guaranteed
// bit-identical to a native (R/Python) build: the evolutionary trajectory is sensitive to
// last-bit libm differences between Emscripten and the native toolchain, so the builds can
// converge to different but equally valid expressions (same engine, same defaults, same
// recovery quality). See docs/51 and docs/37.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rsymbolic/evolution/macro_op.hpp"
#include "rsymbolic/evolution/mutation_weights.hpp"
#include "rsymbolic/evolution/search_space.hpp"
#include "rsymbolic/expression/latex.hpp"
#include "rsymbolic/expression/op_names.hpp"
#include "rsymbolic/expression/tree.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"
#include "rsymbolic/simplification/display_simplify.hpp"
#include "rsymbolic/units/unit_parser.hpp"

using emscripten::val;
using namespace rsymbolic;

namespace {

// --- Small helpers to read optional fields off the JS options object, falling back to
//     the same public defaults the Python wrapper (rsymbolic2/__init__.py) uses, so a
//     minimal JS call {X, y, nrow, ncol} runs at PySR parity. Explicit values override.
bool has(const val& o, const char* key) {
    val v = o[key];
    return !v.isUndefined() && !v.isNull();
}
double get_num(const val& o, const char* key, double dflt) {
    return has(o, key) ? o[key].as<double>() : dflt;
}
int get_int(const val& o, const char* key, int dflt) {
    return has(o, key) ? o[key].as<int>() : dflt;
}
bool get_bool(const val& o, const char* key, bool dflt) {
    return has(o, key) ? o[key].as<bool>() : dflt;
}
std::string get_str(const val& o, const char* key, const std::string& dflt) {
    return has(o, key) ? o[key].as<std::string>() : dflt;
}
std::vector<std::string> get_str_vec(const val& o, const char* key) {
    if (!has(o, key)) return {};
    return emscripten::vecFromJSArray<std::string>(o[key]);
}
std::vector<double> get_num_vec(const val& o, const char* key) {
    if (!has(o, key)) return {};
    return emscripten::convertJSArrayToNumberVector<double>(o[key]);
}

// Build a JS array from a std::vector without registering the vector type.
template <typename T>
val to_js_array(const std::vector<T>& v) {
    val arr = val::array();
    for (std::size_t i = 0; i < v.size(); ++i) arr.set(i, val(v[i]));
    return arr;
}

UnaryOp parse_unary(const std::string& s) {
    UnaryOp op;
    if (unary_from_name(s, op)) return op;
    throw std::invalid_argument("Unknown unary operator: '" + s + "'. Use " +
                                unary_op_name_list() + ".");
}

BinaryOp parse_binary(const std::string& s) {
    BinaryOp op;
    if (binary_from_name(s, op)) return op;
    throw std::invalid_argument("Unknown binary operator: '" + s + "'. Use " +
                                binary_op_name_list() + ".");
}

ModelSelection parse_model_selection(const std::string& s) {
    if (s == "best")     return ModelSelection::Best;
    if (s == "accuracy") return ModelSelection::Accuracy;
    if (s == "score")    return ModelSelection::Score;
    throw std::invalid_argument("Unknown model_selection: '" + s +
                                "'. Use best/accuracy/score.");
}

// Overlay a {name: weight} JS object onto a MutationWeights (absent keys keep PySR
// defaults). Mirrors parse_mutation_weights() in the R/Python bridges but reads the
// known keys directly to avoid iterating arbitrary object keys from embind.
MutationWeights parse_mutation_weights(const val& o) {
    MutationWeights mw;
    if (!has(o, "mutation_weights")) return mw;
    const val w = o["mutation_weights"];
    mw.mutate_constant = get_num(w, "mutate_constant", mw.mutate_constant);
    mw.mutate_operator = get_num(w, "mutate_operator", mw.mutate_operator);
    mw.swap_operands   = get_num(w, "swap_operands",   mw.swap_operands);
    mw.rotate_tree     = get_num(w, "rotate_tree",     mw.rotate_tree);
    mw.add_node        = get_num(w, "add_node",        mw.add_node);
    mw.insert_node     = get_num(w, "insert_node",     mw.insert_node);
    mw.delete_node     = get_num(w, "delete_node",     mw.delete_node);
    mw.do_nothing      = get_num(w, "do_nothing",      mw.do_nothing);
    mw.simplify        = get_num(w, "simplify",        mw.simplify);
    mw.randomize       = get_num(w, "randomize",       mw.randomize);
    return mw;
}

// Run the search. `opts` is a JS object; X is a flat row-major Float64Array with
// `nrow`/`ncol`, y a Float64Array. Returns a JS object with the SAME shape as the
// Python bridge's dict, or {error: message} on failure.
val run(val opts) {
    try {
        const int nrow = get_int(opts, "nrow", 0);
        const int ncol = get_int(opts, "ncol", 0);
        if (nrow <= 0 || ncol <= 0)
            throw std::invalid_argument("nrow and ncol must be positive.");
        const std::size_t n = static_cast<std::size_t>(nrow);
        const std::size_t p = static_cast<std::size_t>(ncol);

        std::vector<double> Xflat = get_num_vec(opts, "X");
        std::vector<double> y     = get_num_vec(opts, "y");
        if (Xflat.size() != n * p)
            throw std::invalid_argument("X length must equal nrow*ncol.");
        if (y.size() != n)
            throw std::invalid_argument("y length must equal nrow.");

        // Flat row-major -> row vectors (the core takes X[row][col]).
        std::vector<std::vector<double>> X(n, std::vector<double>(p));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < p; ++j)
                X[i][j] = Xflat[i * p + j];
        // Release the flat copy immediately: run_evolution() copies X again into its shared
        // Dataset, so all three would otherwise be live at once on a heap that is fixed at
        // 128 MB and aborts (never returns null) when it runs out. See docs/59.
        std::vector<double>().swap(Xflat);

        // --- SearchSpace ------------------------------------------------------------
        SearchSpace space;
        space.num_features = static_cast<int>(p);
        space.max_depth    = get_int(opts, "max_depth", 30);
        space.max_nodes    = get_int(opts, "max_nodes", 30);

        std::vector<std::string> unary_ops  = get_str_vec(opts, "unary_ops");
        std::vector<std::string> binary_ops = get_str_vec(opts, "binary_ops");
        if (binary_ops.empty())  // Python default binary set
            binary_ops = {"add", "sub", "mul"};
        if (!has(opts, "unary_ops"))  // Python default unary set
            unary_ops = {"neg", "exp", "log", "sin", "cos"};
        for (const auto& s : unary_ops)  space.unary_ops.push_back(parse_unary(s));
        space.binary_ops.clear();
        for (const auto& s : binary_ops) space.binary_ops.push_back(parse_binary(s));
        if (space.binary_ops.empty())
            throw std::invalid_argument("binary_ops must contain at least one operator.");

        // Opt-in macro operators (docs/57): single-argument templates over the primitives,
        // expanded at tree construction. Passed as two parallel arrays (names, bodies) like
        // the R and Python bridges — embind cannot enumerate the keys of an arbitrary JS
        // object, and a macro's name is user-chosen. Empty (the default) leaves the search
        // bit-identical to the PySR-parity run.
        std::vector<std::string> macro_names  = get_str_vec(opts, "macro_names");
        std::vector<std::string> macro_bodies = get_str_vec(opts, "macro_bodies");
        if (macro_names.size() != macro_bodies.size())
            throw std::invalid_argument("macro_ops names and bodies must have the same "
                                        "length.");
        for (std::size_t i = 0; i < macro_names.size(); ++i) {
            MacroOp macro;
            std::string error;
            if (!make_macro_op(macro_names[i], macro_bodies[i], kMacroArgName,
                               space.max_nodes, macro, error)) {
                throw std::invalid_argument(error);
            }
            space.macro_ops.push_back(std::move(macro));
        }

        // Opt-in dimensional analysis (X_units / y_units / dimensionless_constants_only).
        std::vector<std::string> X_units = get_str_vec(opts, "X_units");
        std::string y_units              = get_str(opts, "y_units", "");
        const bool linear_scaling        = get_bool(opts, "linear_scaling", false);
        if (linear_scaling && (!X_units.empty() || !y_units.empty()))
            throw std::invalid_argument(
                "linear scaling is not supported with dimensional analysis "
                "(X_units/y_units).");
        if (!X_units.empty()) {
            if (X_units.size() != p)
                throw std::invalid_argument(
                    "X_units must have length ncol (= " + std::to_string(p) + ").");
            for (const auto& s : X_units) space.x_units.push_back(parse_unit(s));
        }
        if (!y_units.empty()) {
            if (X_units.empty())
                throw std::invalid_argument("y_units requires X_units.");
            space.y_units = parse_unit(y_units);
        }
        space.dimensionless_constants_only =
            get_bool(opts, "dimensionless_constants_only", false);

        // --- SearchOptions (defaults already match PySR; override what the caller set) -
        SearchOptions o;
        o.space                      = space;
        o.population_size            = static_cast<std::size_t>(get_int(opts, "population_size", 27));
        o.generations                = static_cast<std::size_t>(get_int(opts, "generations", 2800));
        o.tournament_size            = static_cast<std::size_t>(get_int(opts, "tournament_size", 15));
        o.target_loss                = get_num(opts, "target_loss", 1e-10);
        o.simplify_expressions       = get_bool(opts, "simplify", true);
        o.crossover_probability      = get_num(opts, "crossover_probability", 0.0259);
        o.seed                       = static_cast<std::uint64_t>(get_num(opts, "seed", 0));
        o.n_populations              = static_cast<std::size_t>(std::max(1, get_int(opts, "n_populations", 31)));
        o.timeout_seconds            = get_num(opts, "timeout_seconds", 0.0);
        o.verbosity                  = get_int(opts, "verbosity", 0);  // silent in-browser
        o.optimize_probability       = get_num(opts, "optimize_probability", 0.14);
        o.parsimony                  = get_num(opts, "parsimony", 0.0);
        o.adaptive_parsimony_scaling = get_num(opts, "adaptive_parsimony_scaling", 1040.0);
        o.tournament_selection_p     = get_num(opts, "tournament_selection_p", 0.982);
        o.should_optimize_constants  = get_bool(opts, "should_optimize_constants", true);
        o.fraction_replaced_hof      = get_num(opts, "fraction_replaced_hof", 0.0614);
        o.mutation_weights           = parse_mutation_weights(opts);
        o.model_selection            = parse_model_selection(get_str(opts, "model_selection", "best"));
        o.early_stop_condition       = get_num(opts, "early_stop_condition", 0.0);
        o.batching                   = get_bool(opts, "batching", false);
        o.batch_size                 = static_cast<std::size_t>(std::max(1, get_int(opts, "batch_size", 50)));
        o.warmup_maxsize_by          = get_num(opts, "warmup_maxsize_by", 0.0);
        o.n_threads                  = 0;  // single-threaded WASM build; ignored anyway
        o.dimensional_constraint_penalty =
            get_num(opts, "dimensional_constraint_penalty", 1000.0);
        o.eval_cache                 = get_bool(opts, "eval_cache", false);
        o.linear_scaling             = linear_scaling;
        o.strong_simplify            = get_bool(opts, "strong_simplify", false);
        const double max_evals       = get_num(opts, "max_evals", 0.0);
        o.max_evals = max_evals > 0.0 ? static_cast<std::size_t>(max_evals) : 0;

        std::vector<double> weights = get_num_vec(opts, "weights");
        if (!weights.empty()) {
            if (weights.size() != n)
                throw std::invalid_argument("weights must have length nrow.");
            o.weights = weights;
        }

        // Optional progress observer (docs/53): forwarded to JS as
        // {epoch, complexity, loss} once per epoch. Safe to capture `cb` (and `opts`
        // indirectly through it) by value in the closure below because run_evolution()
        // completes synchronously within this call — the captured emscripten::val never
        // outlives the `run()` invocation that created it. The WASM build is
        // single-threaded, so the callback fires on the calling (worker) thread, i.e.
        // synchronously inside this same run() call, never concurrently with it.
        if (has(opts, "on_progress")) {
            val cb = opts["on_progress"];
            // How many epochs the run would take if nothing stopped it early. run_evolution()
            // advances `migration_interval` generations per epoch, so this is what turns the
            // snapshot's `epoch` into a fraction of the budget — the only thing a caller needs
            // to show real progress instead of an indeterminate spinner. Derived here from the
            // options rather than added to ProgressSnapshot so the shared core, and with it the
            // R and Python bridges, stay untouched. It is an UPPER bound: target_loss, the
            // timeout and max_evals can all end the loop sooner.
            const std::size_t interval = std::max<std::size_t>(1, o.migration_interval);
            const double total_epochs =
                static_cast<double>((o.generations + interval - 1) / interval);
            o.progress_callback = [cb, total_epochs](const ProgressSnapshot& s) {
                val obj = val::object();
                obj.set("epoch",        static_cast<double>(s.epoch));
                obj.set("total_epochs", total_epochs);
                obj.set("complexity",   to_js_array(s.complexity));
                obj.set("loss",         to_js_array(s.loss));
                cb(obj);
            };
        }

        // --- Run --------------------------------------------------------------------
        SearchResult res = run_evolution(X, y, o);

        // Weighted total sum of squares about the (weighted) mean, for R^2 = 1 - loss/sst.
        double wsum = 0.0, wysum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double w = o.weights.empty() ? 1.0 : o.weights[i];
            wsum  += w;
            wysum += w * y[i];
        }
        const double ybar = (wsum > 0.0) ? (wysum / wsum) : 0.0;
        double sst = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double w = o.weights.empty() ? 1.0 : o.weights[i];
            const double d = y[i] - ybar;
            sst += w * d * d;
        }

        // --- Pareto front -----------------------------------------------------------
        const std::vector<double> scores = pareto_scores(res.pareto_front);
        std::vector<int>         pf_complexity;
        std::vector<double>      pf_loss, pf_score;
        std::vector<std::string> pf_expr, pf_latex;
        std::vector<std::string> pf_expr_simplified, pf_latex_simplified;
        std::vector<int>         pf_complexity_simplified;
        for (std::size_t i = 0; i < res.pareto_front.size(); ++i) {
            const auto& m = res.pareto_front[i];
            pf_complexity.push_back(m.complexity);
            pf_loss.push_back(m.loss);
            pf_score.push_back(scores[i]);
            pf_expr.push_back(to_string(m.tree));
            pf_latex.push_back(to_latex(m.tree));
            // Display-only companions (docs/52): computed on a COPY of m.tree.
            const Tree simplified = display_simplify(m.tree);
            pf_expr_simplified.push_back(to_string(simplified));
            pf_latex_simplified.push_back(to_latex(simplified));
            // Node count of the SIMPLIFIED tree. `complexity` above is the node count of
            // the raw tree the search actually archived (one member per complexity), so
            // two front members can differ in `complexity` yet print the same simplified
            // expression. Reporting both lets the UI show "10 -> 7" instead of an
            // apparent contradiction. Complexity is tree.size() throughout the core
            // (evolutionary_search.cpp), so the same definition applies here.
            pf_complexity_simplified.push_back(static_cast<int>(simplified.size()));
        }

        const int n_front = static_cast<int>(res.pareto_front.size());
        std::string recommended = res.expression;
        if (n_front > 0 && res.best_index >= 0 && res.best_index < n_front)
            recommended = to_string(res.pareto_front[static_cast<std::size_t>(res.best_index)].tree);

        val pareto = val::object();
        pareto.set("complexity", to_js_array(pf_complexity));
        pareto.set("loss",       to_js_array(pf_loss));
        pareto.set("score",      to_js_array(pf_score));
        pareto.set("expression", to_js_array(pf_expr));
        pareto.set("latex",      to_js_array(pf_latex));
        pareto.set("expression_simplified", to_js_array(pf_expr_simplified));
        pareto.set("latex_simplified",      to_js_array(pf_latex_simplified));
        pareto.set("complexity_simplified", to_js_array(pf_complexity_simplified));

        val eval_counts = val::object();
        eval_counts.set("forward",      static_cast<double>(res.n_forward_evals));
        eval_counts.set("lm_resid",     static_cast<double>(res.n_lm_resid_evals));
        eval_counts.set("lm_jac",       static_cast<double>(res.n_lm_jac_evals));
        eval_counts.set("cache_hits",   static_cast<double>(res.cache_hits));
        eval_counts.set("cache_misses", static_cast<double>(res.cache_misses));
        eval_counts.set("strong_simplify_attempts",
                         static_cast<double>(res.n_strong_simplify_attempts));
        eval_counts.set("strong_simplify_adopted",
                         static_cast<double>(res.n_strong_simplify_adopted));

        val result = val::object();
        result.set("expression",            res.expression);
        result.set("expression_simplified", res.expression_simplified);
        result.set("loss",         res.loss);
        result.set("complexity",   res.complexity);
        result.set("recommended",  recommended);
        result.set("best_index",   n_front > 0 ? val(res.best_index) : val::null());
        result.set("n_obs",        static_cast<int>(n));
        result.set("n_features",   static_cast<int>(p));
        result.set("sst",          sst);
        result.set("n_evals",      static_cast<double>(res.n_evals));
        result.set("eval_counts",  eval_counts);
        result.set("pareto_front", pareto);
        return result;
    } catch (const std::exception& e) {
        val err = val::object();
        err.set("error", std::string(e.what()));
        return err;
    } catch (...) {
        val err = val::object();
        err.set("error", std::string("unknown error in rsymbolic2 WASM run()"));
        return err;
    }
}

}  // namespace

EMSCRIPTEN_BINDINGS(rsymbolic2) {
    emscripten::function("run", &run);
}
