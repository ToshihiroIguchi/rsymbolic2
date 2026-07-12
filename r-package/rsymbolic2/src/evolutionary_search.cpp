// SPDX-License-Identifier: Apache-2.0
// Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
// Defaults and search/mutation mechanisms are matched to SymbolicRegression.jl /
// PySR (Apache-2.0, (C) Miles Cranmer); see the NOTICE file for attribution.

#include "rsymbolic/search/evolutionary_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
// Use R's error channel when building inside the R package; fall back to
// stderr for the standalone cmake build (no R headers available there).
#ifdef __has_include
#  if __has_include(<R_ext/Print.h>)
#    include <R_ext/Print.h>
#    define rsym_eprintf(...) REprintf(__VA_ARGS__)
#  else
#    include <cstdio>
#    define rsym_eprintf(...) std::fprintf(stderr, __VA_ARGS__)
#  endif
#else
#  include <cstdio>
#  define rsym_eprintf(...) std::fprintf(stderr, __VA_ARGS__)
#endif
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#   include <omp.h>
#endif

#include "rsymbolic/evolution/crossover.hpp"
#include "rsymbolic/evolution/hall_of_fame.hpp"
#include "rsymbolic/evolution/mutation.hpp"
#include "rsymbolic/evolution/random_tree.hpp"
#include "rsymbolic/expression/least_squares_problem.hpp"
#include "rsymbolic/search/eval_cache.hpp"
#include "rsymbolic/simplification/display_simplify.hpp"
#include "rsymbolic/simplification/simplify.hpp"
#include "rsymbolic/units/dimensional_analysis.hpp"

namespace rsymbolic {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Slot count of the per-island duplicate-evaluation cache (opt-in options.eval_cache;
// eval_cache.hpp). Power of two (the direct-mapped slot index is key & (slots-1)).
// 1024 entries x one tree of <= max_nodes nodes bounds the memory to a few hundred KB
// per island. Purely an implementation constant — never a search setting.
constexpr std::size_t kEvalCacheSlots = 1024;

// ---------------------------------------------------------------------------
// Optional hot-path phase profiler — DIAGNOSTIC ONLY.
//
// Compiled out entirely unless RSYMBOLIC2_PROFILE is defined, so the shipped R
// package contains none of this (zero footprint, zero overhead by default). It
// attributes wall time to the search's hot-path phases by timing them at their
// call-site boundaries. This granularity is exactly what a sampling profiler
// cannot give here: fit() (now only in the population pass), sse_current() and the
// per-iteration optimise pass all funnel into the same evaluate<double>() kernel, so a
// sampler reports "N% in evaluate()" without saying whether that time is forward-pass
// child eval or population constant optimisation — the distinction this profiler is built
// to make. Each island runs single-threaded
// within the OpenMP parallel-for, so per-island accumulators need no
// synchronisation; they are summed across islands at the end of the run.
#ifdef RSYMBOLIC2_PROFILE
struct IslandProfile {
    double init_fit_sec = 0.0, evolve_fit_sec = 0.0, evolve_sse_sec = 0.0,
           simplify_sec = 0.0, mutate_sec = 0.0,     tournament_sec = 0.0,
           hof_sec = 0.0,      reopt_sec = 0.0;
    std::uint64_t init_fit_n = 0, evolve_fit_n = 0, evolve_sse_n = 0,
           simplify_n = 0, mutate_n = 0, tournament_n = 0, hof_n = 0, reopt_n = 0;
};

// Time the wrapped call and accumulate elapsed seconds / call count. A
// destructor-based guard plus decltype(auto) preserves the wrapped call's value,
// including void, so the same macro times value-returning and void calls alike.
template <class F>
decltype(auto) rsym_timed(double& acc, std::uint64_t& cnt, F&& f) {
    struct Guard {
        double& a;
        std::uint64_t& c;
        std::chrono::steady_clock::time_point t0;
        ~Guard() {
            a += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - t0).count();
            ++c;
        }
    } guard{acc, cnt, std::chrono::steady_clock::now()};
    return f();
}
#  define RSYM_TIMED(acc, cnt, expr) \
       rsym_timed((acc), (cnt), [&] { return (expr); })
#  define RSYM_PROF_BEGIN(acc, cnt) \
       { const auto rsym_t0 = std::chrono::steady_clock::now();
#  define RSYM_PROF_END(acc, cnt) \
       (acc) += std::chrono::duration<double>( \
                    std::chrono::steady_clock::now() - rsym_t0).count(); \
       ++(cnt); }
#else
#  define RSYM_TIMED(acc, cnt, expr) (expr)
#  define RSYM_PROF_BEGIN(acc, cnt)
#  define RSYM_PROF_END(acc, cnt)
#endif

// Per-island running histogram of complexity (node count), for PySR-style
// frequency-based adaptive parsimony. frequencies[c] counts how often a candidate
// of complexity c has been evaluated (every bin starts at 1.0 so none is ever
// zero); normalized[c] = frequencies[c] / sum. Tournament selection multiplies a
// member's base cost by exp(scaling * normalized[complexity]), penalising
// over-represented sizes. Kept per island (share-nothing) so n_populations=1 stays
// fully deterministic. See docs/24.
struct RunningSearchStatistics {
    std::vector<double> frequencies;   // index 0 unused; bins [1, maxsize] are tracked
    std::vector<double> normalized;
    double              sum         = 0.0;
    int                 maxsize     = 1;
    int                 window_size = 100000;

    // SR.jl sizes the histogram exactly by `maxsize` (= max_nodes): one bin per
    // complexity 1..maxsize, each initialised to 1.0, so the uniform baseline is
    // 1/maxsize and the sum-to-1 normalisation matches AdaptiveParsimony.jl. This is
    // what makes PySR's adaptive_parsimony_scaling = 1040 transfer exactly (docs/28
    // §B1) — the earlier max_nodes+2 sizing diluted the normalisation. Bin 0 is kept
    // only for 1-based indexing and stays 0.0. Complexities outside [1, maxsize] are
    // not tracked and incur no frequency penalty, exactly as SR.jl guards them.
    void init(int max_nodes, int window) {
        maxsize = std::max(1, max_nodes);
        frequencies.assign(static_cast<std::size_t>(maxsize) + 1, 1.0);
        frequencies[0] = 0.0;
        normalized.assign(frequencies.size(), 0.0);
        sum = static_cast<double>(maxsize);
        const double inv = 1.0 / sum;
        for (int c = 1; c <= maxsize; ++c)
            normalized[static_cast<std::size_t>(c)] = inv;
        window_size = std::max(1, window);
    }

    bool in_range(int complexity) const {
        return complexity >= 1 && complexity <= maxsize;
    }

    void update(int complexity) {
        if (!in_range(complexity)) return;  // SR.jl tracks only complexities in [1, maxsize]
        frequencies[static_cast<std::size_t>(complexity)] += 1.0;
        sum += 1.0;
        if (sum > static_cast<double>(window_size)) move_window();
    }

    // Sliding window: scale every bin down proportionally so the histogram tracks
    // recent search behaviour rather than the whole run. SR.jl instead subtracts
    // from the largest bins; the proportional form is simpler and keeps every bin
    // strictly positive (docs/24). Bin 0 stays 0.0 under scaling.
    void move_window() {
        const double factor = static_cast<double>(window_size) / sum;
        double s = 0.0;
        for (double& f : frequencies) { f *= factor; s += f; }
        sum = s;
    }

    void normalize() {
        if (sum <= 0.0) return;
        const double inv = 1.0 / sum;
        for (std::size_t i = 0; i < frequencies.size(); ++i)
            normalized[i] = frequencies[i] * inv;
    }

    // Normalised frequency of `complexity`; 0 outside [1, maxsize] so no penalty applies
    // there, exactly as SR.jl's _best_of_sample guards the lookup (docs/28 §B1).
    double freq_at(int complexity) const {
        if (!in_range(complexity)) return 0.0;
        return normalized[static_cast<std::size_t>(complexity)];
    }

    // Frequency used by the use_frequency mutation-acceptance test (SR.jl next_generation):
    // the normalized frequency in range, else 1e-6 (NOT 0) so an out-of-range complexity
    // never makes the old/new ratio divide by zero. Bins are initialised to 1.0, so the
    // in-range value is always strictly positive after normalize().
    double freq_for_accept(int complexity) const {
        if (!in_range(complexity)) return 1e-6;
        const double f = normalized[static_cast<std::size_t>(complexity)];
        return f > 0.0 ? f : 1e-6;
    }
};

// Per-island state. Each island is fully self-contained: its own population,
// hall of fame, RNG, optimizer instance, and complexity statistics. No state is
// shared across islands during parallel evolution, so no synchronisation is
// needed inside the loop.
struct Island {
    std::vector<PopMember>              population;
    HallOfFame                          hof;
    std::mt19937_64                     rng;
    // Dedicated RNG for PySR batching's row subsampling (options.batching). Kept separate
    // from the evolution `rng` so the batch index stream is reproducible and does not
    // perturb the mutation/selection RNG sequence; seeded per island in run_evolution with
    // its own salt so each island's batches are deterministic and thread-count independent.
    std::mt19937_64                     batch_rng;
    std::unique_ptr<ConstantOptimizer>  optimizer;
    RunningSearchStatistics             stats;
    // Monotonic age counter for regularized-evolution replacement. Initial members take
    // 0..population_size-1; every accepted child takes the next value, so the smallest
    // birth is always the oldest surviving member. Per island (share-nothing).
    std::uint64_t                       next_birth = 0;
    // Per-island candidate-evaluation counter (PySR `max_evals` units): forward-pass loss
    // evals (1 each) plus the residual evaluations consumed by constant-optimisation fits.
    // Plain (non-atomic) because it is touched only by this island's single worker thread —
    // no cross-island sharing on the hot path, preserving the share-nothing scaling
    // property. Maintained unconditionally (evaluation accounting, reported in
    // SearchResult); the max_evals budget enforcement reads it only when max_evals > 0.
    // Summed across islands at epoch boundaries. See run_evolution.
    std::uint64_t                       eval_count = 0;
    // Evaluation-accounting breakdown of eval_count (reporting only, never enforced):
    // eval_count == n_forward_evals + n_lm_resid_evals holds by construction. Jacobian
    // builds are reported separately and are NOT part of eval_count (a finite-difference
    // Jacobian's residual calls are already inside n_lm_resid_evals).
    std::uint64_t                       n_forward_evals  = 0;  // forward-pass loss evals
    std::uint64_t                       n_lm_resid_evals = 0;  // LM residual evals (fit)
    std::uint64_t                       n_lm_jac_evals   = 0;  // LM Jacobian builds (fit)
    // Reused scratch for sse_current's SoA residual evaluator. Owned by the island (not a
    // function-local `static thread_local`): each island runs single-threaded on its
    // OpenMP worker, so island-owned buffers are inherently per-worker and reused across
    // the millions of per-child calls without per-call heap churn — and without relying on
    // thread_local, whose per-thread semantics are unreliable for libgomp worker threads
    // inside a loaded DLL on Windows/MinGW (two islands then shared one buffer and raced,
    // corrupting the heap; CLAUDE.md Correctness/Portability).
    std::vector<double>                 sse_pool;
    std::vector<double*>                sse_stk;
    // Reused scratch for sse_linear_scaled's per-point prediction copy (opt-in
    // options.linear_scaling). Island-owned for exactly the same reason as
    // sse_pool/sse_stk above (per-worker reuse without thread_local, whose per-thread
    // semantics are unreliable for libgomp workers inside a loaded DLL). Never touched
    // when linear scaling is off, so the default path allocates nothing.
    std::vector<double>                 pred_buf;
    // Opt-in Keijzer-2003 linear scaling (options.linear_scaling): score_sse dispatches
    // to the best-affine-fit scorer when set. Copied from the options once at island
    // setup (like the cache below) so the hot path reads a plain island-local flag.
    bool                                linear_scaling = false;
    // Opt-in duplicate-evaluation cache (options.eval_cache; eval_cache.hpp). Default-
    // constructed = inactive with zero slots allocated, so the off path costs nothing;
    // run_evolution activates it (kEvalCacheSlots) iff eval_cache is on AND batching is
    // off (a batched SSE is never reusable across passes). Island-local, share-nothing:
    // no cross-island cache state, so the parallel region stays synchronisation-free.
    EvalCache                           cache;
#ifdef RSYMBOLIC2_PROFILE
    IslandProfile                       prof;
#endif
};

// SSE with the tree's current constants — no LM, no Jacobian.
// Used for children that skip the optimizer this step (optimize_probability < 1).
// Strictly cheaper than fit(): one forward pass over the data vs. many LM iterations.
//
// Uses the SoA point-batched residual evaluator (soa_eval.hpp), tiled at kStride: the
// per-point predictions and the in-order summation are bit-identical to the old scalar
// evaluate<double> loop, so the selection cost is unchanged. `pool`/`stk` are caller-owned
// scratch (the calling Island's), reused across the millions of per-child calls without
// per-call heap churn. They must NOT be a function-local `static thread_local`: libgomp
// worker threads inside a loaded DLL (the R package) do not get reliable per-thread storage
// on Windows/MinGW, so two islands shared one buffer and raced — corrupting the heap.
double sse_current(const Tree& tree, const Dataset& data,
                   std::vector<double>& pool, std::vector<double*>& stk) {
    const std::vector<double> c = initial_constants(tree);
    const std::size_t m = data.y.size();
    // Per-point weighting (PySR `weights`): the same sqrt(w_i) scaling as the LM residual
    // closure, so sse = sum_i w_i (pred_i - y_i)^2. nullptr => unweighted (skip multiply).
    const double* sw = data.sqrt_weights.empty() ? nullptr : data.sqrt_weights.data();
    double sse = 0.0;
    for (std::size_t lo = 0; lo < m; lo += kStride) {
        const std::size_t P = std::min(kStride, m - lo);
        const double* pred =
            evaluate_soa_residual(tree, data.Xcol, c.data(), lo, P, pool, stk);
        for (std::size_t p = 0; p < P; ++p) {
            double r = pred[p] - data.y[lo + p];
            if (sw) r *= sw[lo + p];
            if (!std::isfinite(r)) return kInf;
            sse += r * r;
        }
    }
    return sse;
}

// Keijzer-2003 linear-scaling scorer (opt-in options.linear_scaling): the loss of a
// candidate f is the SSE of the best affine transform of its predictions,
//   min_{a,b} sum_i w_i (a*f_i + b - y_i)^2,
// with (a, b) solved in closed form as the weighted least squares of y on f. Pass 1
// tile-evaluates exactly like sse_current (same SoA evaluator, kStride tiles, same
// point order) while accumulating the weighted moments and copying the predictions
// into pred_buf (caller-owned scratch — the calling Island's — with the same ownership
// rules as pool/stk); any non-finite prediction returns kInf immediately, matching
// sse_current's rejection. Pass 2 sums the affine residuals over pred_buf in data
// order, the same in-order summation style as sse_current. Weights: Dataset stores
// sqrt(w_i) (see least_squares_problem.hpp), so the moment weight is
// sqrt_weights[i]^2, or 1.0 when unweighted. The fitted (a, b) are written to
// a_out/b_out when non-null (the finalize pass reads them to materialise the wrap).
double sse_linear_scaled(const Tree& tree, const Dataset& data,
                         std::vector<double>& pool, std::vector<double*>& stk,
                         std::vector<double>& pred_buf,
                         double* a_out = nullptr, double* b_out = nullptr) {
    const std::vector<double> c = initial_constants(tree);
    const std::size_t m = data.y.size();
    const double* sw = data.sqrt_weights.empty() ? nullptr : data.sqrt_weights.data();
    pred_buf.resize(m);
    double Sw = 0.0, Swf = 0.0, Swy = 0.0, Swff = 0.0, Swfy = 0.0;
    for (std::size_t lo = 0; lo < m; lo += kStride) {
        const std::size_t P = std::min(kStride, m - lo);
        const double* pred =
            evaluate_soa_residual(tree, data.Xcol, c.data(), lo, P, pool, stk);
        for (std::size_t p = 0; p < P; ++p) {
            const double f = pred[p];
            if (!std::isfinite(f)) return kInf;
            pred_buf[lo + p] = f;
            const double w  = sw ? sw[lo + p] * sw[lo + p] : 1.0;
            const double yv = data.y[lo + p];
            Sw   += w;
            Swf  += w * f;
            Swy  += w * yv;
            Swff += w * f * f;
            Swfy += w * f * yv;
        }
    }
    // Closed-form weighted least squares of y on f. Sw > 0 holds in every real run:
    // unweighted data gives Sw = m (>= 1 row, enforced by the wrappers) and weights are
    // validated finite and NON-NEGATIVE — so an all-zero weight vector is the one
    // degenerate input that reaches Sw == 0. Guard it with a = 0, b = 0: every residual
    // then carries zero weight, so the returned loss is 0 either way.
    double a = 0.0, b = 0.0;
    if (Sw > 0.0) {
        const double var_f = Swff - (Swf * Swf) / Sw;
        const double cov   = Swfy - (Swf * Swy) / Sw;
        a = cov / var_f;
        b = Swy / Sw - a * (Swf / Sw);
        if (!(var_f > 0.0) || !std::isfinite(a) || !std::isfinite(b)) {
            // Degenerate predictor (constant f => var_f == 0, or a numerically hopeless
            // solve): the least-squares limit is the constant model a = 0,
            // b = weighted mean(y), keeping the loss definition "always the best affine
            // fit" (here: the best constant fit, the weighted SST).
            a = 0.0;
            b = Swy / Sw;
        }
    }
    double sse = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        double r = a * pred_buf[i] + b - data.y[i];
        if (sw) r *= sw[i];
        sse += r * r;
    }
    if (a_out) *a_out = a;
    if (b_out) *b_out = b;
    return sse;
}

// The raw forward-pass loss under the active scorer: plain weighted SSE by default, or
// the Keijzer-2003 best-affine-fit SSE when the opt-in linear_scaling is on. This is
// the single dispatch point between the two loss definitions — everything routed
// through score_sse (init, evolution, the batching finalize) sees the same one. With
// the flag off the call collapses to the former direct sse_current call.
double raw_forward_sse(Island& isl, const Tree& tree, const Dataset& data) {
    return isl.linear_scaling
        ? sse_linear_scaled(tree, data, isl.sse_pool, isl.sse_stk, isl.pred_buf)
        : sse_current(tree, data, isl.sse_pool, isl.sse_stk);
}

// Forward-pass scoring used at every former sse_current call site: the active scorer
// (raw_forward_sse) plus the evaluation-counter charge, routed through the island's
// opt-in duplicate-evaluation cache (options.eval_cache). With the cache inactive (the
// default, and always under batching) this is exactly the former inline pattern — the
// scorer followed by ++eval_count/++n_forward_evals — byte-identical code path. With
// the cache active, an evaluation-identical tree's loss is returned from the memo
// instead of re-evaluated; a cached value is whatever the ACTIVE scorer defines (the
// linear_scaling flag is fixed for the whole run, so memoised values never mix
// scorers; hash/equality are unchanged). CRITICAL for bit-identity: a cache hit is
// charged to eval_count/n_forward_evals exactly like a real evaluation — otherwise a
// max_evals-budgeted run would diverge between cache on and off. The returned value is
// the RAW scorer output (before add_dim_penalty, which stays at the call sites); kInf
// results are cached like any other (re-evaluating them would only waste the same
// forward pass again).
double score_sse(Island& isl, const Tree& tree, const Dataset& data) {
    if (!isl.cache.active()) {
        const double sse = raw_forward_sse(isl, tree, data);
        ++isl.eval_count;
        ++isl.n_forward_evals;
        return sse;
    }
    const std::uint64_t key = tree_hash(tree);
    if (const double* cached = isl.cache.lookup(key, tree)) {
        ++isl.cache.hits;
        ++isl.eval_count;        // a hit still counts as one forward-pass evaluation
        ++isl.n_forward_evals;   // (max_evals budget parity with the cache off)
        return *cached;
    }
    const double sse = raw_forward_sse(isl, tree, data);
    ++isl.cache.misses;
    isl.cache.store(key, tree, sse);
    ++isl.eval_count;
    ++isl.n_forward_evals;
    return sse;
}

// y_norm, the NMSE denominator. Unweighted: sum((y - mean(y))^2). Weighted (PySR
// `weights`, w non-empty): sum(w_i (y_i - wmean)^2) with wmean = sum(w_i y_i)/sum(w_i),
// so y_norm scales with the same overall weight magnitude as the weighted loss and the
// loss/y_norm ratio stays invariant to rescaling all weights. Returns 1.0 when the
// denominator would be zero (constant y, single point, or non-positive total weight).
double compute_y_norm(const std::vector<double>& y, const std::vector<double>& w) {
    if (y.size() <= 1) return 1.0;
    if (w.empty()) {
        const double mean_y =
            std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
        double ss = 0.0;
        for (const double v : y) { const double d = v - mean_y; ss += d * d; }
        return ss > 0.0 ? ss : 1.0;
    }
    double sw = 0.0, swy = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) { sw += w[i]; swy += w[i] * y[i]; }
    if (sw <= 0.0) return 1.0;
    const double mean_y = swy / sw;
    double ss = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
        const double d = y[i] - mean_y;
        ss += w[i] * d * d;
    }
    return ss > 0.0 ? ss : 1.0;
}

// Build a batched dataset for one PySR-batching evolution or optimisation pass (SR.jl
// batch(): SymbolicRegression.jl Dataset.jl::batch). Samples `batch_size` row indices
// uniformly WITH REPLACEMENT, then rescales each sampled point's weight by n_full/batch_size
// so the weighted SSE over the batch is an unbiased estimate of the full-dataset SSE. Paired
// with the FULL-dataset y_norm in the selection cost, this makes the batched cost equal
//   (SSE_batch / batch_size) / (SST_full / n_full),
// which is bit-for-bit SR.jl's mean_loss_batch / full_baseline (loss_to_cost reads the full
// dataset's baseline even for a SubDataset). Folding the rescale into the per-point weight
// means the hot-path evaluators (sse_current / fit) need no batching-specific code — they
// already apply sqrt(weight) — and member.loss stays in full-equivalent SSE units, so it is
// directly comparable to the full-data losses written by finalize_costs_and_merge and to
// target_loss. Rows are gathered into a fresh owned Dataset (small: batch_size rows), shared
// by const pointer exactly like the full dataset.
std::shared_ptr<const Dataset> make_batch(const Dataset& full, std::size_t batch_size,
                                          std::mt19937_64& rng) {
    const std::size_t n  = full.y.size();
    const std::size_t bs = std::min(std::max<std::size_t>(1, batch_size), n);
    const double scale   = static_cast<double>(n) / static_cast<double>(bs);
    const bool weighted  = !full.sqrt_weights.empty();
    std::vector<std::vector<double>> Xb(bs);
    std::vector<double> yb(bs), wb(bs);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    for (std::size_t i = 0; i < bs; ++i) {
        const std::size_t idx = pick(rng);
        Xb[i] = full.X[idx];
        yb[i] = full.y[idx];
        // The Dataset ctor takes raw weights (it sqrt's them); recover the original weight
        // w = sqrt_weights^2 (1 when unweighted) and scale it. With replacement, a repeated
        // index contributes its weight once per draw — matching SR.jl's repeated indices.
        const double w = weighted ? full.sqrt_weights[idx] * full.sqrt_weights[idx] : 1.0;
        wb[i] = w * scale;
    }
    return std::make_shared<const Dataset>(
        Dataset{std::move(Xb), std::move(yb), std::move(wb)});
}

// Base selection cost: loss/y_norm + parsimony*complexity.
// When parsimony=0 this collapses to raw loss, preserving pre-B2 behaviour. Used
// by migration and as the base for the frequency-adjusted tournament cost.
inline double base_cost(const PopMember& m, double parsimony, double y_norm) {
    if (parsimony <= 0.0) return m.loss;
    return m.loss / y_norm + parsimony * static_cast<double>(m.complexity);
}

// Frequency-adjusted cost used in tournament selection. With scaling<=0 it is
// exactly base_cost (frequency penalty disabled — byte-identical to pre-adaptive
// behaviour). Otherwise the base is taken in normalized-loss units
// (loss/y_norm + parsimony*complexity) so the multiplicative factor is comparable
// across problems, and multiplied by exp(scaling * normalized_frequency[complexity]).
// The exponent is clamped to avoid overflow to +inf, which would tie every
// candidate. See docs/24.
inline double adjusted_cost(const PopMember& m, double parsimony, double y_norm,
                            const RunningSearchStatistics& stats, double scaling) {
    if (scaling <= 0.0) return base_cost(m, parsimony, y_norm);
    const double base =
        m.loss / y_norm + parsimony * static_cast<double>(m.complexity);
    double e = scaling * stats.freq_at(m.complexity);
    if (e > 50.0) e = 50.0;
    return base * std::exp(e);
}

double fit(Tree& tree, const std::shared_ptr<const Dataset>& data,
           const ConstantOptimizer& optimizer,
           const StopRequested& stop_requested,
           Island& isl) {
    const std::vector<double> init = initial_constants(tree);
    // Pass stop_requested through so the residual/Jacobian closures can poll it
    // mid-evaluation and signal an abort via problem.aborted (docs/22 Phase 1).
    // The dataset is shared (not copied) so each fit() avoids the per-call dataset
    // copy that was the dominant heap-contention source (docs/23 §4).
    OptimizationProblem problem = make_least_squares_problem(tree, data, init,
                                                             stop_requested);
    const OptimizationResult result = optimizer.optimize(problem, stop_requested);
    // Evaluation accounting (also the max_evals budget unit, PySR): charge the
    // optimiser's reported residual evaluations (at least 1 even on failure — work was
    // still done). Jacobian builds are reported only, never charged to eval_count.
    const std::uint64_t resid_evals =
        result.evaluations > 0 ? static_cast<std::uint64_t>(result.evaluations) : 1;
    isl.eval_count       += resid_evals;
    isl.n_lm_resid_evals += resid_evals;
    isl.n_lm_jac_evals   += static_cast<std::uint64_t>(result.jacobian_evaluations);
    if (!result.success || !std::isfinite(result.loss)) return kInf;
    if (!result.constants.empty()) set_constants(tree, result.constants);
    return result.loss;
}

std::size_t sample_index(std::size_t size, std::mt19937_64& rng) {
    return std::uniform_int_distribution<std::size_t>(0, size - 1)(rng);
}

// Parent selection. With p >= 1 this is a deterministic best-of-k tournament (the
// original behaviour, fewest RNG draws). With 0 < p < 1 it is PySR's probabilistic
// tournament (`tournament_selection_p`): sample k members, rank them by cost, and
// pick the rank-r best with probability p*(1-p)^r (normalised over the k ranks), so a
// slightly-worse parent is occasionally chosen — a diversity pressure PySR relies on.
std::size_t tournament_best(const std::vector<PopMember>& pop, std::size_t k,
                            std::mt19937_64& rng, double parsimony, double y_norm,
                            const RunningSearchStatistics& stats, double scaling,
                            double p) {
    if (p >= 1.0) {
        std::size_t best = sample_index(pop.size(), rng);
        double best_cost = adjusted_cost(pop[best], parsimony, y_norm, stats, scaling);
        for (std::size_t i = 1; i < k; ++i) {
            const std::size_t cand = sample_index(pop.size(), rng);
            const double c = adjusted_cost(pop[cand], parsimony, y_norm, stats, scaling);
            if (c < best_cost) { best = cand; best_cost = c; }
        }
        return best;
    }

    struct Cand { std::size_t idx; double cost; };
    std::vector<Cand> cand(k);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t idx = sample_index(pop.size(), rng);
        cand[i] = {idx, adjusted_cost(pop[idx], parsimony, y_norm, stats, scaling)};
    }
    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b) { return a.cost < b.cost; });

    // Pick a rank from the geometric-style weights with a single uniform draw.
    // Cumulative weight through rank r is 1-(1-p)^(r+1); total mass is 1-(1-p)^k.
    const double q      = 1.0 - p;
    const double total  = 1.0 - std::pow(q, static_cast<double>(k));
    const double target = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * total;
    std::size_t rank = k - 1;  // default to the worst sampled if rounding leaves a gap
    double cum = 0.0;
    for (std::size_t r = 0; r < k; ++r) {
        cum += p * std::pow(q, static_cast<double>(r));
        if (target < cum) { rank = r; break; }
    }
    return cand[rank].idx;
}

// Index of the oldest population member (smallest birth stamp), the regularized-evolution
// replacement target (SR.jl reg_evol_cycle::argmin_fast over member.birth). Linear scan;
// population_size is small (PySR default 27).
std::size_t oldest_index(const std::vector<PopMember>& pop) {
    std::size_t oldest = 0;
    std::uint64_t min_birth = pop[0].birth;
    for (std::size_t i = 1; i < pop.size(); ++i) {
        if (pop[i].birth < min_birth) { min_birth = pop[i].birth; oldest = i; }
    }
    return oldest;
}

// Build the per-search dimensional-analysis context from the options and the full
// dataset. Disabled (a no-op for the loss path) unless X_units or y_units were declared.
// penalty_sse folds in the normalisation weight W (= n, or sum of weights when weighted)
// so that adding it to an SSE-scale loss reproduces PySR's penalty on the loss/baseline
// scale (see add_dim_penalty in dimensional_analysis.hpp).
DimAnalysis build_dim_analysis(const SearchOptions& opts, const std::vector<double>& y) {
    DimAnalysis da;
    da.enabled = !opts.space.x_units.empty() || opts.space.y_units.has_value();
    if (!da.enabled) return da;
    da.x_units = opts.space.x_units;
    da.y_units = opts.space.y_units;
    da.allow_wildcards = !opts.space.dimensionless_constants_only;
    double W = static_cast<double>(y.size());
    if (!opts.weights.empty()) {
        W = 0.0;
        for (const double w : opts.weights) W += w;
    }
    da.penalty_sse = opts.dimensional_constraint_penalty * W;
    return da;
}

void initialize_island(Island& isl,
                       const std::shared_ptr<const Dataset>& data,
                       const SearchOptions& opts,
                       const StopRequested& stop_requested,
                       const DimAnalysis& da) {
    isl.population.clear();
    isl.population.reserve(opts.population_size);
    isl.stats.init(opts.space.max_nodes, opts.parsimony_window);
    for (std::size_t i = 0; i < opts.population_size; ++i) {
        // Empty-population guard: always create at least one member. The downstream
        // tournament selection calls sample_index(pop.size(), ...) which computes
        // uniform_int_distribution(0, size-1) — undefined for size == 0. We only
        // check the predicate before the second member onward, so every island that
        // exists has at least one member and is safe for evolve_island.
        if (i > 0 && stop_requested()) break;
        // SR.jl seeds the initial population with gen_random_tree(nlength=3): a small tree of
        // ~3 operators grown by append_random_op, NOT a depth-bounded recursive tree. nlength=3
        // is hardcoded in SR.jl (Configure.jl / SymbolicRegression.jl), not a PySR-tunable knob.
        // Phase-0 seeding oracle hook (diagnostic only): if a seed tree is supplied for this
        // member slot, inject a copy of it instead of a random tree. Empty seed_trees (the
        // default) never takes this branch, so the default search path is unaffected.
        Tree tree = (i < opts.seed_trees.size()) ? opts.seed_trees[i]
                                                  : gen_random_tree(3, opts.space, isl.rng);
        // Score the initial member with its random constants via a single forward pass —
        // NOT an LM fit. SR.jl builds the initial population with eval_cost only (its
        // PopMember constructor; Population.jl), never optimising constants at birth; the
        // first optimisation opportunity is the end-of-iteration population pass. The
        // previous per-member init fit() diverged from PySR (docs/31). sse_current draws no
        // island RNG, so the per-island sequence is unchanged from the prior init.
        // score_sse charges the forward-pass eval counters (max_evals units) itself.
        const double raw_loss = RSYM_TIMED(isl.prof.init_fit_sec, isl.prof.init_fit_n,
                                           score_sse(isl, tree, *data));
        // Opt-in dimensional penalty (no-op when da.enabled is false; docs/46).
        const double loss = add_dim_penalty(raw_loss, tree, da);
        // No simplify here: SR.jl does not simplify the initial population. Simplification
        // first happens in the per-iteration optimize_and_simplify_population pass below,
        // mirroring SR.jl.
        PopMember m{std::move(tree), loss, 0};
        m.complexity = static_cast<int>(m.tree.size());
        m.birth = isl.next_birth++;  // initial members age 0..population_size-1
        isl.hof.update(m);
        isl.stats.update(m.complexity);
        isl.population.push_back(std::move(m));
    }
}

using Clock    = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

// Returns elapsed seconds since `start`.
inline double elapsed_sec(const TimePoint& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Evolve island for exactly n_gens generations (or until early stop / timeout).
// Returns true if the early-stop threshold was reached.
// `data` is the dataset children are scored on (the full dataset normally, or an epoch
// batch under PySR batching) and `y_norm` is the matching selection-cost denominator
// (always the full-data y_norm — see make_batch). `hof_target` receives every evaluated
// candidate: it is the island's own hall of fame on the full-data path, or a per-epoch
// best-seen archive under batching (whose members are later recomputed on the full dataset
// by finalize_costs_and_merge before reaching isl.hof). The early-stop test still reads
// isl.hof, which under batching holds only full-data losses, so the search never stops on a
// lucky batch.
bool evolve_island(Island& isl,
                   const std::shared_ptr<const Dataset>& data,
                   const SearchOptions& opts,
                   std::size_t n_gens,
                   const TimePoint& t_start,
                   bool has_deadline,
                   double deadline_sec,
                   double y_norm,
                   std::size_t island_eval_budget,
                   int cur_maxsize,
                   HallOfFame& hof_target,
                   const DimAnalysis& da) {
    const std::size_t tournament =
        opts.tournament_size == 0 ? 1 : opts.tournament_size;
    // PySR warmup_maxsize_by: the mutation/crossover size cap for this epoch is the warmed-up
    // cur_maxsize, not opts.space.max_nodes. A local SearchSpace carries it into mutate() (its
    // room/can_add/can_insert filter == SR.jl condition_mutation_weights! size zeroing) and the
    // randomize mutation (gen_random_tree over 1..cur_maxsize); the crossover size check takes
    // cur_maxsize directly. The frequency histogram (isl.stats) and the initial population are
    // sized by the full max_nodes elsewhere and are deliberately left untouched (docs/42). At
    // the default warmup_maxsize_by=0 the caller passes cur_maxsize == max_nodes, so cur_space
    // equals opts.space and the path is byte-for-byte the pre-warmup behaviour.
    SearchSpace cur_space   = opts.space;
    cur_space.max_nodes     = cur_maxsize;
    const double parsimony = opts.parsimony;
    const double scaling   = opts.adaptive_parsimony_scaling;
    const double tsel_p    = opts.tournament_selection_p;
    // Effective early-stop threshold: stop once the best loss falls below either
    // target_loss (the near-exact default) or early_stop_condition (PySR `max_evals`'s
    // companion knob); the larger threshold dominates, so it stops the run sooner.
    // Under the opt-in linear_scaling the compared loss is the best-affine-fit SSE,
    // which is <= the raw SSE (a=1, b=0 is one affine candidate), so early stop can
    // only fire sooner, never later.
    const double target    = std::max(opts.target_loss, opts.early_stop_condition);
    // Evaluation counting itself is unconditional (evaluation accounting); only the
    // budget enforcement below is gated by max_evals > 0.
    const bool   enforce_budget = island_eval_budget > 0;  // max_evals enabled
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Build the stop closure once. It reads t_start (immutable after search start)
    // and the wall clock — safe to call from any thread under the OpenMP parallel-for.
    const StopRequested stop = [&] {
        return has_deadline && elapsed_sec(t_start) >= deadline_sec;
    };

    // Evaluate a freshly built child with its INHERITED constants (single forward pass, no
    // LM — SR.jl crossover_generation uses eval_cost only) and, if finite, insert it by
    // regularized-evolution replacement of the current oldest member. A non-finite loss is
    // SR.jl's NaN-cost rejection: the child is discarded (rsymbolic2 keeps no NaN-loss members
    // in the population). Used for the two crossover children: the second call sees the first
    // child as the newest member, so the two replacements land on the two oldest members
    // (SR.jl reg_evol_cycle's oldest1/oldest2). No use_frequency test — the crossover path has
    // none in SR.jl (next_generation applies it only to mutations).
    auto insert_evaluated_child = [&](Tree&& t) {
        const double raw_loss =
            RSYM_TIMED(isl.prof.evolve_sse_sec, isl.prof.evolve_sse_n,
                       score_sse(isl, t, *data));
        if (!std::isfinite(raw_loss)) return;
        const double loss = add_dim_penalty(raw_loss, t, da);  // no-op when units off
        PopMember child{std::move(t), loss, 0};
        child.complexity = static_cast<int>(child.tree.size());
        const int child_complexity = child.complexity;
        child.birth = isl.next_birth++;
        RSYM_TIMED(isl.prof.hof_sec, isl.prof.hof_n, hof_target.update(child));
        const std::size_t oldest = oldest_index(isl.population);
        isl.population[oldest] = std::move(child);
        isl.stats.update(child_complexity);
    };

    for (std::size_t gen = 0; gen < n_gens; ++gen) {
        if (!isl.hof.empty() && isl.hof.best().loss < target)
            return true;
        if (stop())
            return false;
        // max_evals budget: stop this island once it has spent its fair share of the
        // global evaluation budget. Checked at the generation boundary (not the optimizer
        // stop closure), so a fit is never aborted mid-solve and counting stays clean.
        if (enforce_budget && isl.eval_count >= island_eval_budget)
            return false;

        // Refresh the normalized frequency snapshot once per generation (cheap,
        // O(max_nodes)). Both the tournament-cost penalty (scaling) and the use_frequency
        // acceptance test read this snapshot; the raw counts keep accumulating and are
        // re-normalized next generation (matches SR.jl's per-cycle cadence; deterministic).
        if (scaling > 0.0 || opts.use_frequency) isl.stats.normalize();

        for (std::size_t step = 0; step < opts.population_size; ++step) {
            // Coarse step-boundary check: catches overshoot between fit() calls.
            // The finer check inside fit() -> optimizer catches overshoot within a
            // single bloated-tree optimization. Together they bound overshoot to one
            // LM step rather than one full minimize() call. See docs/20.
            if (stop())
                return false;

            const std::size_t parent = RSYM_TIMED(
                isl.prof.tournament_sec, isl.prof.tournament_n,
                tournament_best(isl.population, tournament, isl.rng,
                                parsimony, y_norm, isl.stats, scaling, tsel_p));
            const int parent_complexity = isl.population[parent].complexity;
            Tree child_tree;
            CrossoverChildren pair;
            bool is_crossover = false;
            bool mutated      = true;
            // Child construction (parent copy + mutate, or two-child subtree crossover). The
            // crossover branch's parent2 tournament draw is rare (crossover_probability
            // default 0.0259) and is counted within this block, not tournament.
            RSYM_PROF_BEGIN(isl.prof.mutate_sec, isl.prof.mutate_n)
            if (opts.crossover_probability > 0.0 &&
                unit(isl.rng) < opts.crossover_probability) {
                is_crossover = true;
                const std::size_t parent2 = tournament_best(isl.population,
                                                            tournament, isl.rng,
                                                            parsimony, y_norm,
                                                            isl.stats, scaling, tsel_p);
                pair = subtree_crossover_pair(
                    isl.population[parent].tree,
                    isl.population[parent2].tree,
                    isl.rng, cur_space.max_nodes);
            } else {
                child_tree = isl.population[parent].tree;
                mutated = mutate(child_tree, cur_space, isl.rng, opts.const_perturb_scale,
                                 opts.mutation_weights, opts.probability_negate_constant);
            }
            RSYM_PROF_END(isl.prof.mutate_sec, isl.prof.mutate_n)

            // Crossover path: SR.jl crossover_generation produces TWO children from one breed
            // and reg_evol_cycle replaces the two oldest members with them. A breed that could
            // not satisfy the constraints after 10 tries is skipped entirely
            // (skip_mutation_failures) — no member is replaced.
            if (is_crossover) {
                if (!pair.accepted)
                    continue;
                insert_evaluated_child(std::move(pair.child1));
                insert_evaluated_child(std::move(pair.child2));
                continue;
            }

            // SR.jl skip_mutation_failures: a sampled mutation that could not apply leaves
            // the parent unchanged and is skipped — the oldest member is NOT replaced this
            // step (a no-op cycle), rather than injecting a randomized tree.
            if (!mutated)
                continue;

            // Evaluate the child with its INHERITED constants — a single forward pass
            // (sse_current), never an LM fit. This matches SR.jl's next_generation, which
            // only calls eval_cost on a mutated child and never optimises constants during
            // the regularized-evolution cycle. Constant optimisation is confined to the
            // once-per-iteration population pass (optimize_and_simplify_population below),
            // exactly as in SR.jl. The previous per-child LM (probability optimize_probability)
            // diverged from PySR and dominated compute (docs/30, docs/31).
            const double raw_loss =
                RSYM_TIMED(isl.prof.evolve_sse_sec, isl.prof.evolve_sse_n,
                           score_sse(isl, child_tree, *data));
            // A non-finite loss is SR.jl's NaN-cost rejection: discard the child (no
            // replacement, no archive update), faithful to next_generation.
            if (!std::isfinite(raw_loss))
                continue;
            const double loss = add_dim_penalty(raw_loss, child_tree, da);  // no-op when off
            // No per-child simplify: SR.jl's s_r_cycle evolves raw trees and simplifies
            // only once per iteration (optimize_and_simplify_population, called per epoch
            // below) plus via the weighted `simplify` mutation. The child's complexity
            // reflects its raw form during evolution, matching SR.jl.
            PopMember child{std::move(child_tree), loss, 0};
            child.complexity = static_cast<int>(child.tree.size());

            // use_frequency acceptance (SR.jl next_generation; mutation path only — the
            // crossover path returned above, as SR.jl applies no frequency test there). Accept
            // the child with probability old_freq/new_freq (clamped to <= 1 by the comparison):
            // a child drifting toward an over-represented complexity is probabilistically
            // rejected, leaving the parent in place. annealing is off (PySR default), so only
            // the frequency factor applies.
            if (opts.use_frequency) {
                const double prob_change =
                    isl.stats.freq_for_accept(parent_complexity) /
                    isl.stats.freq_for_accept(child.complexity);
                if (prob_change < unit(isl.rng))
                    continue;  // rejected: parent survives, oldest not replaced
            }

            // Accepted: regularized-evolution replacement — overwrite the OLDEST member
            // unconditionally (SR.jl reg_evol_cycle). Selection pressure comes from the
            // parent tournament; survival is age-bounded, which keeps the population
            // churning and preserves diversity. The hall of fame independently retains the
            // best member per complexity, so no quality is lost by an unconditional
            // overwrite (CLAUDE.md Correctness; cf. migration's same rationale).
            const int child_complexity = child.complexity;  // child is moved below
            child.birth = isl.next_birth++;
            RSYM_TIMED(isl.prof.hof_sec, isl.prof.hof_n, hof_target.update(child));
            const std::size_t oldest = oldest_index(isl.population);
            isl.population[oldest] = std::move(child);

            // Record the accepted child's complexity in the running histogram. Updating
            // after the acceptance keeps every comparison in this generation consistent
            // with the start-of-generation normalized snapshot; the new count only affects
            // subsequent generations. No RNG is consumed, so determinism is preserved.
            isl.stats.update(child_complexity);
        }
    }
    return false;
}

// Once-per-iteration population pass, faithful to SR.jl's optimize_and_simplify_population
// (SingleIteration.jl): for each population member, simplify it and then — with probability
// optimize_probability, gated by should_optimize_constants — LM-optimise its constants. This
// is the ONLY place constant optimisation happens during evolution: the reg-evol cycle
// (evolve_island) never optimises children. The previous design optimised per child plus a
// per-epoch HOF re-optimisation; both diverged from PySR and inflated optimizer calls ~28x
// (docs/30, docs/31). The optimise decision draws one uniform from the island RNG per member,
// so it is deterministic and thread-count-independent (share-nothing per island).
//
// simplify() preserves the member's value under exact arithmetic, but combine_operators
// reassociates constants, so the loss can shift by a floating-point rounding step; we
// recompute it (SR.jl recomputes via finalize_costs). When a member is optimised, fit()
// returns the optimised loss and writes the tuned constants back; otherwise we evaluate the
// simplified tree's SSE with its inherited constants. Members whose simplify/optimise step
// produced a non-finite loss are left unchanged. Every resulting member is offered to the
// hall of fame. Deadline-guarded via `stop` (LM polls it mid-evaluation).
// `data` is the dataset constants are optimised on (full data normally, an epoch batch under
// batching) and `hof_target` receives each finalised member, or is nullptr under batching —
// where the hall-of-fame merge is deferred to finalize_costs_and_merge so that only
// full-data losses ever enter the archive (SR.jl optimises on the batch, then finalize_costs
// recomputes the population on the full dataset before the global hall of fame is updated).
void optimize_and_simplify_population(Island& isl,
                                      const std::shared_ptr<const Dataset>& data,
                                      const SearchOptions& opts,
                                      const StopRequested& stop,
                                      std::size_t island_eval_budget,
                                      HallOfFame* hof_target,
                                      const DimAnalysis& da) {
    // Evaluation counting is unconditional; only the budget check is gated (max_evals).
    const bool enforce_budget = island_eval_budget > 0;  // max_evals enabled
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (PopMember& m : isl.population) {
        // max_evals budget: stop polishing once this island has spent its fair share.
        if (enforce_budget && isl.eval_count >= island_eval_budget) break;
        Tree s = opts.simplify_expressions
                     ? RSYM_TIMED(isl.prof.simplify_sec, isl.prof.simplify_n, simplify(m.tree))
                     : m.tree;

        // SR.jl draws rand() < optimizer_probability per member (do_optimization), gated by
        // should_optimize_constants; trees with no constants are no-ops in the optimiser.
        const bool do_optimize =
            opts.should_optimize_constants && !stop() &&
            unit(isl.rng) < opts.optimize_probability && count_constants(s) > 0;

        double loss;
        if (do_optimize) {
            Tree t = s;
            loss = RSYM_TIMED(isl.prof.reopt_sec, isl.prof.reopt_n,
                              fit(t, data, *isl.optimizer, stop, isl));
            if (!std::isfinite(loss)) continue;  // keep the pre-optimise member
            // Opt-in linear_scaling: fit() minimises (and returns) the UNSCALED SSE —
            // the LM optimiser deliberately stays on raw residuals (v1). Scaled and
            // unscaled losses must never mix in the hall-of-fame/tournament ordering,
            // so re-score the optimised tree through the active (scaled) scorer: one
            // extra charged forward pass per optimised member. No-op branch when off.
            if (opts.linear_scaling) {
                loss = score_sse(isl, t, *data);
                if (!std::isfinite(loss)) continue;  // keep the pre-optimise member
            }
            m.tree = std::move(t);
        } else {
            loss = score_sse(isl, s, *data);
            if (!std::isfinite(loss)) continue;  // keep the unsimplified member if it degraded
            m.tree = std::move(s);
        }
        // Recompute the dimensional penalty on the finalised (possibly simplified) tree:
        // simplify() can change structure and thus violation status (no-op when units off).
        m.loss = add_dim_penalty(loss, m.tree, da);
        m.complexity = static_cast<int>(m.tree.size());
        if (hof_target) hof_target->update(m);
    }
}

// PySR batching finalize: SR.jl finalize_costs (Population.jl) plus the best_seen full-data
// recompute (SymbolicRegression.jl:1129). After a batched epoch the population members and
// the per-epoch best_seen archive carry batched (full-equivalent-estimate) losses; recompute
// every one on the FULL dataset — a forward pass only, no re-optimisation, exactly as SR.jl's
// eval_cost — and merge the full-data members into the island's hall of fame. This is the
// single point where batched-epoch results reach the archive, guaranteeing the hall of fame,
// early-stop test and final result are decided on the full data just as in PySR. Members
// whose full-data loss is non-finite are skipped. The population's own member.loss is updated
// to the full-data value so the next epoch's tournament starts from full-data parent costs
// (SR.jl: finalize_costs writes member.cost/loss back into the population).
void finalize_costs_and_merge(Island& isl,
                              const std::shared_ptr<const Dataset>& full,
                              const HallOfFame& best_seen,
                              const DimAnalysis& da) {
    // Routed through score_sse for uniformity; this function only runs under batching,
    // where the cache is inactive, so every call is a real full-data evaluation.
    for (PopMember& m : isl.population) {
        const double loss = score_sse(isl, m.tree, *full);
        if (!std::isfinite(loss)) continue;
        m.loss = add_dim_penalty(loss, m.tree, da);  // no-op when units off
        m.complexity = static_cast<int>(m.tree.size());
        isl.hof.update(m);
    }
    for (PopMember bm : best_seen.members()) {
        const double loss = score_sse(isl, bm.tree, *full);
        if (!std::isfinite(loss)) continue;
        bm.loss = add_dim_penalty(loss, bm.tree, da);  // no-op when units off
        isl.hof.update(bm);
    }
}

// PySR-parity migration primitive (SymbolicRegression.jl 1.11.0 Migration.jl::migrate!):
// draw poisson(|dst| * frac) migrants, then for each one pick a uniformly random
// destination slot and a uniformly random migrant from `pool` (both with replacement)
// and overwrite UNCONDITIONALLY — migration is a stochastic mixing operator, not an
// elitist one. The hall-of-fame archive preserves the global best independently, so an
// unconditional overwrite never loses the incumbent solution (CLAUDE.md Correctness).
void inject_migrants(std::vector<PopMember>& dst_pop,
                     const std::vector<PopMember>& pool, double frac,
                     std::mt19937_64& rng) {
    if (pool.empty() || dst_pop.empty() || frac <= 0.0) return;
    const double mean = static_cast<double>(dst_pop.size()) * frac;
    std::poisson_distribution<long long> poisson(mean);
    const long long draw = poisson(rng);
    std::size_t num = draw > 0 ? static_cast<std::size_t>(draw) : 0;
    num = std::min(num, dst_pop.size());
    if (num == 0) return;

    std::uniform_int_distribution<std::size_t> slot(0, dst_pop.size() - 1);
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    for (std::size_t r = 0; r < num; ++r) {
        dst_pop[slot(rng)] = pool[pick(rng)];  // unconditional overwrite
    }
}

// Ring migration: snapshot the top-`topn` members of each island (the migrant pool,
// bounded by PySR's topn), then stochastically inject island i's pool into island
// (i+1)%n. Called serially after each epoch barrier — no concurrent access. Ranking uses
// base_cost (no frequency factor): migration transfers quality between islands and must
// not depend on any single island's complexity histogram. At the PySR default
// (fraction_replaced=0.00036, population_size=27) the Poisson mean is ~0.01, so ring
// migration almost never fires — faithful to PySR, which is likewise near-inert here.
void migrate(std::vector<Island>& islands, double frac, std::size_t topn,
             double parsimony, double y_norm, std::mt19937_64& rng) {
    const std::size_t n = islands.size();
    if (n < 2 || frac <= 0.0 || topn == 0) return;

    // Snapshot before any injection so emigrants are never stale.
    std::vector<std::vector<PopMember>> pool(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto& pop = islands[i].population;
        if (pop.empty()) continue;
        std::vector<const PopMember*> ptrs(pop.size());
        for (std::size_t j = 0; j < pop.size(); ++j) ptrs[j] = &pop[j];
        const std::size_t take = std::min(topn, pop.size());
        std::partial_sort(ptrs.begin(), ptrs.begin() + static_cast<std::ptrdiff_t>(take),
                          ptrs.end(),
                          [parsimony, y_norm](const PopMember* a, const PopMember* b) {
                              return base_cost(*a, parsimony, y_norm) <
                                     base_cost(*b, parsimony, y_norm);
                          });
        pool[i].reserve(take);
        for (std::size_t j = 0; j < take; ++j) pool[i].push_back(*ptrs[j]);
    }

    for (std::size_t i = 0; i < n; ++i)
        inject_migrants(islands[(i + 1) % n].population, pool[i], frac, rng);
}

// Hall-of-fame migration (PySR fraction_replaced_hof, hof_migration=True). Merge every
// island's archive into one global elite, take its Pareto front (= PySR's `dominating`
// set), then stochastically inject it into every island's population. Runs serially
// after the evolution barrier (no concurrent access) and for any island count, so even a
// single population regains elites it had discarded. The Poisson count uses the
// destination population size, matching Migration.jl.
void migrate_hof(std::vector<Island>& islands, double frac, std::mt19937_64& rng) {
    if (frac <= 0.0) return;

    HallOfFame global;
    for (const auto& isl : islands) global.merge(isl.hof);
    if (global.empty()) return;
    const std::vector<PopMember> elite = global.pareto_front();
    if (elite.empty()) return;

    for (auto& isl : islands)
        inject_migrants(isl.population, elite, frac, rng);
}

}  // namespace

SearchResult run_evolution(const std::vector<std::vector<double>>& X,
                           const std::vector<double>& y,
                           const SearchOptions& options) {
    const std::size_t n = std::max<std::size_t>(1, options.n_populations);

    // Start the clock before island initialisation so that init time counts against
    // the timeout budget. Previously t_start was set after initialize_island, meaning
    // a long init was not charged to the deadline. See docs/20.
    const TimePoint   t_start      = Clock::now();
    const bool        has_deadline = options.timeout_seconds > 0.0;
    const double      deadline_sec = options.timeout_seconds;

    const StopRequested init_stop = [&] {
        return has_deadline && elapsed_sec(t_start) >= deadline_sec;
    };

    // Copy the dataset once into shared, immutable storage. Every island's fit()
    // references this single Dataset instead of copying X/y per evaluation, which was
    // the dominant multi-island heap-contention source (docs/23 §4). Read-only and
    // shared by const pointer, so concurrent island workers never write to it.
    const auto data = std::make_shared<const Dataset>(Dataset{X, y, options.weights});

    // Opt-in dimensional-analysis context (docs/46). Disabled unless X_units/y_units were
    // declared, in which case every add_dim_penalty call below is a no-op and the search is
    // byte-identical to the units-off PySR-parity default. Built once and shared read-only.
    const DimAnalysis dim_analysis = build_dim_analysis(options, y);

    // Build islands. Island 0 always uses options.seed so that n_populations=1
    // produces exactly the same RNG sequence as the pre-island implementation.
    //
    // Island initialisation (population_size full fit()s per island) is the same
    // share-nothing, embarrassingly parallel work as the evolution region: each
    // island touches only islands[i] and reads the shared const dataset/options.
    // Running it serially made init wall grow linearly with n_populations and
    // directly starved the per-run generation budget — measured as the dominant
    // multi-island scaling loss once SelfLM removed the per-fit heap contention
    // (docs/23 §4 follow-up). Parallelise it with the same team-size cap as the
    // evolution loop. Determinism is unaffected: every island is seeded
    // deterministically before its (independent) initialisation, so the result is
    // identical regardless of thread count or completion order.
    std::vector<Island> islands(n);
#ifdef _OPENMP
    const int init_threads = resolve_team_size(n, options.n_threads, omp_get_max_threads());
#   pragma omp parallel for schedule(dynamic) num_threads(init_threads) if(n > 1)
#endif
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        islands[idx].rng.seed(idx == 0
            ? options.seed
            : options.seed + idx * UINT64_C(0x9e3779b97f4a7c15));
        // Seed each island's optimizer deterministically for its constant-optimization
        // restart perturbations (SelfLM multi-start). A per-island, idx-only seed keeps the
        // restart RNG reproducible and thread-count independent (test_island_model), and a
        // distinct salt decorrelates it from the island evolution RNG and the migration RNG.
        OptimizerConfig isl_opt_config = options.optimizer_config;
        isl_opt_config.seed = options.seed ^
                              (idx * UINT64_C(0x9e3779b97f4a7c15)) ^
                              UINT64_C(0xA5A5A5A5A5A5A5A5);
        islands[idx].optimizer =
            OptimizerFactory::create(options.optimizer_type, isl_opt_config);
        // Per-island batch-sampling RNG (PySR batching). Distinct salt from the evolution,
        // optimizer-restart and migration streams so the batch index sequence is decorrelated
        // from them yet fully determined by (seed, island index) — reproducible and
        // thread-count independent. Unused when options.batching is false.
        islands[idx].batch_rng.seed(options.seed ^
                                    (idx * UINT64_C(0x9e3779b97f4a7c15)) ^
                                    UINT64_C(0x243F6A8885A308D3));
        // Opt-in duplicate-evaluation cache: active iff requested AND batching is off
        // (batches change every pass, so a batched SSE is never reusable). Decided once
        // here; the default-constructed inactive cache allocates nothing. Activated
        // before initialize_island so the init forward passes are cached too.
        if (options.eval_cache && !options.batching)
            islands[idx].cache = EvalCache(kEvalCacheSlots);
        // Opt-in Keijzer-2003 linear scaling: copy the flag onto the island before
        // initialize_island so the init forward passes already use the scaled scorer.
        islands[idx].linear_scaling = options.linear_scaling;
        initialize_island(islands[idx], data, options, init_stop, dim_analysis);
    }

    const std::size_t interval = std::max<std::size_t>(1, options.migration_interval);
    const double      y_norm    = compute_y_norm(y, options.weights);  // selection-cost denominator

    // Effective early-stop threshold (target_loss or the looser early_stop_condition).
    const double      target    = std::max(options.target_loss, options.early_stop_condition);

    // max_evals budget (PySR `max_evals`), split into a deterministic per-island fair
    // share so islands self-limit without any shared hot-path counter. 0 = no limit.
    // ceil-divide so the shares sum to >= max_evals (never starve the budget by rounding).
    const std::size_t island_budget =
        options.max_evals > 0
            ? std::max<std::size_t>(1, (options.max_evals + n - 1) / n)
            : 0;

    // Dedicated, deterministic RNG for stochastic migration (PySR draws Poisson counts
    // and random destination slots / migrants). A separate stream from the per-island
    // RNGs keeps migration reproducible and independent of island evolution.
    std::mt19937_64 migration_rng(options.seed ^ UINT64_C(0xD1B54A32D192ED03));

    bool reached_target = false;
    bool timed_out      = false;
    bool evals_exhausted = false;  // max_evals budget spent (global sum across islands)
    std::size_t epoch   = 0;

#ifdef RSYMBOLIC2_PROFILE
    double        migrate_sec = 0.0;  // serial ring + HOF migration (main thread)
    std::uint64_t migrate_n   = 0;
#endif

    for (std::size_t done = 0;
         done < options.generations && !reached_target && !timed_out && !evals_exhausted;
         done += interval, ++epoch) {
        const std::size_t gens =
            std::min(interval, options.generations - done);

        // PySR warmup_maxsize_by: size cap for this epoch's evolution. fraction_elapsed uses
        // the generations completed at the epoch START (done), matching SR.jl, where the first
        // iteration sees fraction 0 (cur_maxsize = 3 when warmup is on). At the default
        // warmup_maxsize_by=0 this is always max_nodes, so evolve_island's path is unchanged.
        const double fraction_elapsed =
            options.generations > 0 ? static_cast<double>(done) / options.generations : 0.0;
        const int cur_maxsize = get_cur_maxsize(
            options.space.max_nodes, fraction_elapsed, options.warmup_maxsize_by);

#ifdef _OPENMP
        // One worker thread per island, capped at the hardware. The default
        // OpenMP team size is the core count; when that exceeds the number of
        // islands the surplus threads have no loop iteration and busy-wait at the
        // implicit barrier, starving the island workers. Under that starvation a
        // single node evaluation's wall time can balloon to tens of seconds, so
        // the deadline poll (which runs on the starved worker) cannot fire and the
        // run grossly overshoots its timeout (docs/22). Islands are the only unit
        // of parallelism here, so capping the team at n loses no useful concurrency.
        // Team size = resolve_team_size(islands, options.n_threads, omp_get_max_threads()).
        // options.n_threads == 0 (default) falls back to omp_get_max_threads() (not
        // omp_get_num_procs()): it honours OMP_NUM_THREADS / omp_set_num_threads, so a user
        // (or a benchmark equalizing the compute budget against another tool) can cap the team
        // via the standard env var, and with the env unset it defaults to the core count — so
        // the default behaviour is unchanged. A positive options.n_threads (the explicit
        // R/Python n_threads argument) overrides that. Either way the team is capped at the
        // island count (surplus threads only busy-wait — docs/22).
        const int omp_threads = resolve_team_size(n, options.n_threads, omp_get_max_threads());
#       pragma omp parallel for schedule(dynamic) num_threads(omp_threads) if(n > 1)
#endif
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(n); ++i) {
            Island& isl = islands[static_cast<std::size_t>(i)];
            const StopRequested opt_stop = [&] {
                return has_deadline && elapsed_sec(t_start) >= deadline_sec;
            };
            // PySR optimize_and_simplify_population: once per iteration, simplify every
            // population member and (with probability optimize_probability, gated by
            // should_optimize_constants) LM-optimise its constants — the single place
            // constant optimisation runs during evolution. Same order as SR.jl (simplify
            // before optimise), inside the share-nothing parallel region; the deadline
            // closure reads only immutable state, safe across threads.
            if (!options.batching) {
                // Full-data path (default): score and optimise on the whole dataset, and
                // update the island's hall of fame directly — byte-for-byte the pre-batching
                // behaviour.
                evolve_island(isl, data, options, gens, t_start, has_deadline,
                              deadline_sec, y_norm, island_budget, cur_maxsize, isl.hof,
                              dim_analysis);
                optimize_and_simplify_population(isl, data, options, opt_stop,
                                                 island_budget, &isl.hof, dim_analysis);
            } else {
                // PySR batching (SR.jl SingleIteration.jl): one epoch = one PySR iteration.
                // Draw a fresh batch for the reg-evol cycle, evolving against a per-epoch
                // best_seen archive (NOT isl.hof); draw a second batch for the constant
                // optimisation pass (hof_target = nullptr); then recompute the population and
                // best_seen on the full dataset and merge into isl.hof. The full y_norm is
                // used throughout because make_batch rescales the batch weights to keep the
                // batched cost on the full-data scale.
                const auto batch_reg = make_batch(*data, options.batch_size, isl.batch_rng);
                HallOfFame best_seen;
                evolve_island(isl, batch_reg, options, gens, t_start, has_deadline,
                              deadline_sec, y_norm, island_budget, cur_maxsize, best_seen,
                              dim_analysis);
                const auto batch_opt = make_batch(*data, options.batch_size, isl.batch_rng);
                optimize_and_simplify_population(isl, batch_opt, options, opt_stop,
                                                 island_budget, nullptr, dim_analysis);
                finalize_costs_and_merge(isl, data, best_seen, dim_analysis);
            }
        }

        if (n > 1) {
            // PySR ring migration replaces poisson(population_size * fraction_replaced)
            // members each iteration (Migration.jl), drawing migrants from the topn source
            // pool (migration_size). At the PySR default (0.00036, population_size=27) the
            // Poisson mean is ~0.01, so ring migration almost never fires — faithful.
            RSYM_PROF_BEGIN(migrate_sec, migrate_n)
            migrate(islands, options.fraction_replaced, options.migration_size,
                    options.parsimony, y_norm, migration_rng);
            RSYM_PROF_END(migrate_sec, migrate_n)
        }
        RSYM_PROF_BEGIN(migrate_sec, migrate_n)
        migrate_hof(islands, options.fraction_replaced_hof, migration_rng);
        RSYM_PROF_END(migrate_sec, migrate_n)

        // Optional progress observer (docs/53): all islands are quiescent here (the
        // OpenMP island-parallel region above has joined, and both ring and HOF
        // migration for this epoch are done), so this is a safe serial point to merge
        // a read-only snapshot of the global Pareto front and hand it to the caller.
        // Pure observation — no RNG use, no mutation of search state — so with the
        // callback unset (the default) this is a single untaken branch and the run is
        // unaffected (PySR Default Parity, CLAUDE.md).
        if (options.progress_callback) {
            HallOfFame snapshot_hof;
            for (const auto& isl : islands) snapshot_hof.merge(isl.hof);
            const std::vector<PopMember> front = snapshot_hof.pareto_front();
            ProgressSnapshot snap;
            // `epoch` is still the 0-based index of the iteration that just finished
            // its evolution/migration (the loop's `++epoch` runs after this body), so
            // +1 reports the count of COMPLETED outer iterations, matching the field's
            // doc comment in evolutionary_search.hpp.
            snap.epoch = epoch + 1;
            snap.complexity.reserve(front.size());
            snap.loss.reserve(front.size());
            for (const auto& m : front) {
                snap.complexity.push_back(m.complexity);
                snap.loss.push_back(m.loss);
            }
            options.progress_callback(snap);
        }

        // Global early stop: once any island reaches the target loss the answer is
        // found, so stop instead of grinding the remaining islands and epochs through
        // their full budget. Within-epoch early stop is handled inside evolve_island.
        for (const auto& isl : islands) {
            if (!isl.hof.empty() && isl.hof.best().loss < target) {
                reached_target = true;
                break;
            }
        }

        if (has_deadline && elapsed_sec(t_start) >= deadline_sec)
            timed_out = true;

        // max_evals budget (PySR): sum the per-island counters at the epoch boundary and
        // stop the run once the global total reaches max_evals. The per-island fair-share
        // check inside evolve_island bounds within-epoch overshoot; this is the global cap.
        if (options.max_evals > 0) {
            std::uint64_t total_evals = 0;
            for (const auto& isl : islands) total_evals += isl.eval_count;
            if (total_evals >= options.max_evals) evals_exhausted = true;
        }

        // Per-epoch diagnostic log (verbosity >= 1).
        if (options.verbosity >= 1) {
            // Gather stats across all islands' populations.
            double best_loss = std::numeric_limits<double>::infinity();
            std::vector<int> sizes, nconsts;
            for (const auto& isl : islands) {
                if (!isl.hof.empty()) {
                    const double l = isl.hof.best().loss;
                    if (l < best_loss) best_loss = l;
                }
                for (const auto& m : isl.population) {
                    sizes.push_back(m.complexity);
                    nconsts.push_back(count_constants(m.tree));
                }
            }
            const std::size_t sz = sizes.size();
            int size_med = 0, size_max = 0, nc_med = 0, nc_max = 0;
            if (sz > 0) {
                std::nth_element(sizes.begin(),
                                 sizes.begin() + static_cast<std::ptrdiff_t>(sz / 2),
                                 sizes.end());
                size_med = sizes[sz / 2];
                size_max = *std::max_element(sizes.begin(), sizes.end());
                std::nth_element(nconsts.begin(),
                                 nconsts.begin() + static_cast<std::ptrdiff_t>(sz / 2),
                                 nconsts.end());
                nc_med = nconsts[sz / 2];
                nc_max = *std::max_element(nconsts.begin(), nconsts.end());
            }
            rsym_eprintf(
                "[epoch %zu  t=%.1fs] best=%.3e  size med/max=%d/%d"
                "  nconst med/max=%d/%d\n",
                epoch, elapsed_sec(t_start), best_loss,
                size_med, size_max, nc_med, nc_max);
        }
    }

#ifdef RSYMBOLIC2_PROFILE
    {
        // Sum the per-island phase accumulators. These are work-seconds summed
        // across islands (NOT wall): with islands running in parallel, the sum tells
        // where the compute goes, which is what we optimise. The harness reports wall
        // and cpu/wall for the parallel-health picture.
        IslandProfile t;
        for (const auto& isl : islands) {
            t.init_fit_sec   += isl.prof.init_fit_sec;   t.init_fit_n   += isl.prof.init_fit_n;
            t.evolve_fit_sec += isl.prof.evolve_fit_sec; t.evolve_fit_n += isl.prof.evolve_fit_n;
            t.evolve_sse_sec += isl.prof.evolve_sse_sec; t.evolve_sse_n += isl.prof.evolve_sse_n;
            t.reopt_sec      += isl.prof.reopt_sec;      t.reopt_n      += isl.prof.reopt_n;
            t.simplify_sec   += isl.prof.simplify_sec;   t.simplify_n   += isl.prof.simplify_n;
            t.mutate_sec     += isl.prof.mutate_sec;     t.mutate_n     += isl.prof.mutate_n;
            t.tournament_sec += isl.prof.tournament_sec; t.tournament_n += isl.prof.tournament_n;
            t.hof_sec        += isl.prof.hof_sec;        t.hof_n        += isl.prof.hof_n;
        }
        const double total = t.init_fit_sec + t.evolve_fit_sec + t.evolve_sse_sec +
                             t.reopt_sec + t.simplify_sec + t.mutate_sec +
                             t.tournament_sec + t.hof_sec + migrate_sec;
        const double inv = total > 0.0 ? 100.0 / total : 0.0;
        rsym_eprintf(
            "\n=== rsymbolic2 phase profile (summed work-seconds over %zu islands) ===\n",
            n);
        rsym_eprintf("  %-13s %11s %7s %14s %11s\n",
                     "phase", "seconds", "pct", "calls", "us/call");
        auto row = [&](const char* name, double sec, std::uint64_t cnt) {
            rsym_eprintf("  %-13s %11.3f %6.1f%% %14llu %11.3f\n", name, sec, sec * inv,
                         static_cast<unsigned long long>(cnt),
                         cnt ? sec / static_cast<double>(cnt) * 1e6 : 0.0);
        };
        // init_sse and evolve_sse are forward-pass evaluations (no LM); popopt_fit is the
        // once-per-iteration population constant optimisation — the only LM phase now that
        // per-child fit and HOF re-optimisation are gone (docs/31).
        row("init_sse",     t.init_fit_sec,   t.init_fit_n);
        row("evolve_sse",   t.evolve_sse_sec, t.evolve_sse_n);
        row("popopt_fit",   t.reopt_sec,      t.reopt_n);
        row("simplify",     t.simplify_sec,   t.simplify_n);
        row("mutate_xover", t.mutate_sec,     t.mutate_n);
        row("tournament",   t.tournament_sec, t.tournament_n);
        row("hof_update",   t.hof_sec,        t.hof_n);
        row("migration",    migrate_sec,      migrate_n);
        rsym_eprintf("  %-13s %11.3f %6.1f%%\n", "TOTAL", total, total * inv);
        rsym_eprintf("  (work-seconds; divide by thread count for ~wall contribution)\n");

        // Within-fit split (all fit phases combined): residual vs Jacobian closures.
        const double resid_s = static_cast<double>(g_resid_ns.load()) * 1e-9;
        const double jac_s   = static_cast<double>(g_jac_ns.load()) * 1e-9;
        const std::uint64_t resid_c = g_resid_calls.load();
        const std::uint64_t jac_c   = g_jac_calls.load();
        const double fit_eval = resid_s + jac_s;
        const double finv = fit_eval > 0.0 ? 100.0 / fit_eval : 0.0;
        rsym_eprintf("  within fit() eval: residual %.3fs (%.1f%%, %llu calls, %.3f us)"
                     "  jacobian %.3fs (%.1f%%, %llu calls, %.3f us)\n",
                     resid_s, resid_s * finv,
                     static_cast<unsigned long long>(resid_c),
                     resid_c ? resid_s / static_cast<double>(resid_c) * 1e6 : 0.0,
                     jac_s, jac_s * finv,
                     static_cast<unsigned long long>(jac_c),
                     jac_c ? jac_s / static_cast<double>(jac_c) * 1e6 : 0.0);
    }
#endif

    // Merge per-island halls of fame into a single global result.
    HallOfFame global;
    for (auto& isl : islands) global.merge(isl.hof);

    // Opt-in linear_scaling: materialise the fitted affine map into every reported
    // tree (Keijzer 2003). Serial, deterministic, post-merge and RNG-free. For each
    // archived member in ascending-complexity order: refit (a, b) on the FULL dataset,
    // wrap the tree as a*f + b unless (a, b) is the identity to numerical precision,
    // then recompute the loss as the PLAIN SSE of the wrapped tree — equal to the
    // scaled SSE up to FP rounding — so member.loss, the expression string and
    // predict() agree exactly. A fresh HallOfFame is rebuilt because the +4-node wrap
    // shifts complexity buckets; a wrapped tree may exceed max_nodes by up to 4, which
    // is allowed (reporting-time materialisation, not a search move). These few
    // finalize evaluations run on the main thread after all islands are done, so they
    // are charged to no island counter; they are added directly to the reported
    // n_evals/n_forward_evals totals below.
    std::uint64_t finalize_forward_evals = 0;
    if (options.linear_scaling) {
        std::vector<double>  fin_pool, fin_pred;
        std::vector<double*> fin_stk;
        const double mean_y = y.empty()
            ? 0.0
            : std::accumulate(y.begin(), y.end(), 0.0) / static_cast<double>(y.size());
        HallOfFame rebuilt;
        for (PopMember m : global.members()) {  // ascending complexity, deterministic
            double a = 1.0, b = 0.0;
            const double scaled = sse_linear_scaled(m.tree, *data, fin_pool, fin_stk,
                                                    fin_pred, &a, &b);
            ++finalize_forward_evals;
            if (!std::isfinite(scaled)) continue;  // drop a non-finite archived member
            // Skip the wrap when (a, b) is numerically the identity, so a candidate
            // that already matches y is reported unmodified (b's tolerance is scaled
            // by the target's magnitude).
            const bool identity =
                std::abs(a - 1.0) <= 1e-12 &&
                std::abs(b) <= 1e-12 * (1.0 + std::abs(mean_y));
            if (!identity) {
                // Postfix wrap: [f..., a, Mul, b, Add] = (f * a) + b. Constant slots
                // are re-indexed afterwards with the existing helper, preserving the
                // contiguous-slot invariant every consumer relies on.
                m.tree.push_back(constant_node(0, a));
                m.tree.push_back(binary_node(BinaryOp::Mul));
                m.tree.push_back(constant_node(0, b));
                m.tree.push_back(binary_node(BinaryOp::Add));
                reindex_constants(m.tree);
            }
            const double loss = sse_current(m.tree, *data, fin_pool, fin_stk);
            ++finalize_forward_evals;
            if (!std::isfinite(loss)) continue;
            m.loss       = loss;
            m.complexity = static_cast<int>(m.tree.size());
            rebuilt.update(m);
        }
        // Keep the unmaterialised archive only if every member failed the refit (an
        // all-non-finite degenerate run), so the result below is never empty.
        if (!rebuilt.empty()) global = std::move(rebuilt);
    }

    const PopMember best = global.best();
    SearchResult result;
    result.tree       = best.tree;
    result.loss       = best.loss;
    result.complexity = best.complexity;
    result.expression = to_string(best.tree);
    // Display-only companion (docs/52): operates on a COPY of best.tree; never written
    // back, never used by predict() (docs/48 D2 frozen-expression rule).
    result.expression_simplified = to_string(display_simplify(best.tree));
    result.pareto_front = global.pareto_front();
    result.best_index =
        static_cast<int>(select_best(result.pareto_front, options.model_selection));
    // Evaluation accounting: sum the per-island counters. eval_count is the max_evals
    // unit (forward passes + LM residual evals), so n_evals == n_forward_evals +
    // n_lm_resid_evals by construction; Jacobian builds are reported separately.
    for (const auto& isl : islands) {
        result.n_evals          += isl.eval_count;
        result.n_forward_evals  += isl.n_forward_evals;
        result.n_lm_resid_evals += isl.n_lm_resid_evals;
        result.n_lm_jac_evals   += isl.n_lm_jac_evals;
        result.cache_hits       += isl.cache.hits;
        result.cache_misses     += isl.cache.misses;
    }
    // Main-thread finalize evaluations (opt-in linear_scaling materialisation): charged
    // to no island counter, counted directly here. 0 when the option is off.
    result.n_evals         += finalize_forward_evals;
    result.n_forward_evals += finalize_forward_evals;
    return result;
}

}  // namespace rsymbolic
