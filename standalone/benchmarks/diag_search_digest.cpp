// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
//
// Search-result digest — a diff-based bit-identity harness for refactors that are
// required to leave the search unchanged (docs/65).
//
// The permanent test suite asserts INTERNAL consistency (thread-count determinism,
// option-on-vs-off identity, ...). It cannot catch a refactor that changes every path
// consistently. This driver produces an external golden instead: run it on the parent
// commit, save the output, run it again after the change, and `diff`. Any difference at
// all is a failure.
//
// Every floating-point value is printed with %a (hexadecimal float), which is exact — a
// %g comparison would hide a low-bit change, which is precisely the failure mode being
// guarded against.
//
// Usage:
//   ./standalone/diag_search_digest > before.txt     # on the parent commit
//   ./standalone/diag_search_digest > after.txt      # after the change
//   diff before.txt after.txt
//
// Cases are chosen to cover the code paths the memory work touches: the default
// full-data path, several island counts and thread counts (the per-worker scratch),
// batching (which is the only reader of the row-major dataset copy), and the opt-in
// scorers that route through different evaluation entry points.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/benchmark/problems.hpp"
#include "rsymbolic/search/evolutionary_search.hpp"

namespace {

using rsymbolic::SearchOptions;
using rsymbolic::SearchResult;

void report(const std::string& label, const SearchResult& r) {
    std::printf("=== %s\n", label.c_str());
    std::printf("  expression   : %s\n", r.expression.c_str());
    // expression_simplified is deliberately NOT digested. It is display-only (docs/52),
    // and display_simplify()'s e-graph runs under a wall-clock budget, so the SAME binary
    // produces different strings on different runs. Including it would make this golden
    // report false differences. Everything below is a search output and is exact.
    std::printf("  loss         : %a\n", r.loss);
    std::printf("  complexity   : %d\n", r.complexity);
    std::printf("  best_index   : %d\n", r.best_index);
    std::printf("  n_evals      : %llu\n", static_cast<unsigned long long>(r.n_evals));
    std::printf("  n_forward    : %llu\n",
                static_cast<unsigned long long>(r.n_forward_evals));
    std::printf("  n_lm_resid   : %llu\n",
                static_cast<unsigned long long>(r.n_lm_resid_evals));
    std::printf("  n_lm_jac     : %llu\n",
                static_cast<unsigned long long>(r.n_lm_jac_evals));
    std::printf("  cache        : %llu / %llu\n",
                static_cast<unsigned long long>(r.cache_hits),
                static_cast<unsigned long long>(r.cache_misses));
    for (std::size_t i = 0; i < r.pareto_front.size(); ++i) {
        const auto& m = r.pareto_front[i];
        std::printf("  front[%02zu]    : c=%d loss=%a expr=%s\n",
                    i, m.complexity, m.loss, rsymbolic::to_string(m.tree).c_str());
    }
    std::fflush(stdout);
}

// A small but non-trivial budget: enough generations for several epochs (so migration,
// the constant-optimisation pass and the hall of fame all run) while staying quick.
SearchOptions base_opts(const rsymbolic::BenchmarkProblem& prob, std::uint64_t seed) {
    SearchOptions o;
    o.space           = prob.space;
    o.generations     = 120;
    o.seed            = seed;
    o.n_populations   = 4;
    o.verbosity       = 0;
    // No timeout: a wall-clock stop would make this driver non-reproducible and
    // useless as a golden (docs/22).
    o.timeout_seconds = 0.0;
    return o;
}

void run_case(const std::string& label, const rsymbolic::BenchmarkProblem& prob,
              const SearchOptions& o) {
    report(label, rsymbolic::run_evolution(prob.X, prob.y, o));
}

}  // namespace

int main() {
    const auto nguyen1  = rsymbolic::problem_nguyen1();
    const auto keijzer  = rsymbolic::problem_nguyen5();
    const auto multivar = rsymbolic::problem_multivar();

    for (std::uint64_t seed : {1u, 7u, 42u}) {
        {
            SearchOptions o = base_opts(nguyen1, seed);
            run_case("nguyen1 default seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.n_populations = 1;
            run_case("nguyen1 npop=1 seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.n_populations = 8;
            o.n_threads     = 3;  // team smaller than the island count
            run_case("nguyen1 npop=8 nthreads=3 seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.n_populations = 8;
            o.n_threads     = 1;  // fully serial team
            run_case("nguyen1 npop=8 nthreads=1 seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.batching   = true;   // the only reader of the row-major dataset copy
            o.batch_size = 8;
            run_case("nguyen1 batching seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.eval_cache = true;
            run_case("nguyen1 eval_cache seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.linear_scaling = true;
            run_case("nguyen1 linear_scaling seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.weights.assign(nguyen1.y.size(), 1.0);
            for (std::size_t i = 0; i < o.weights.size(); ++i)
                o.weights[i] = 0.5 + 0.1 * static_cast<double>(i % 7);
            run_case("nguyen1 weights seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(nguyen1, seed);
            o.warmup_maxsize_by = 0.5;
            run_case("nguyen1 warmup seed=" + std::to_string(seed), nguyen1, o);
        }
        {
            SearchOptions o = base_opts(keijzer, seed);
            run_case("nguyen5 default seed=" + std::to_string(seed), keijzer, o);
        }
        {
            // Multi-feature: the case the column-major input change actually reshapes.
            SearchOptions o = base_opts(multivar, seed);
            run_case("multivar default seed=" + std::to_string(seed), multivar, o);
        }
    }
    return 0;
}
