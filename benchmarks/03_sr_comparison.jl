# Reference comparison: SymbolicRegression.jl (the engine behind PySR), driven directly
# from Julia. PySR is only a thin Python/scikit-learn wrapper over SymbolicRegression.jl;
# the search algorithm and constant optimisation are entirely in the Julia package, so
# benchmarking SymbolicRegression.jl directly is an equivalent — and cleaner — comparison
# than going through PySR's juliacall bridge (which is broken on this machine's Microsoft
# Store Python; see docs/15). The algorithm version is what matters and is reported below.
#
# Data: bit-identical to the rsymbolic2 gate, exported by export_nguyen_data.R.
# Protocol (see docs/15 for the full reporting checklist):
#   - JIT is warmed up once before timing (compilation is amortised once per session,
#     exactly as in real PySR usage — not gaming the clock).
#   - Equal wall-clock budget per run (TIMEOUT_S) and identical early-stop threshold,
#     to put both tools on the same footing despite different internal population config.
#   - PySR's default population configuration is mirrored.
#   - Recovery uses the SAME criterion as benchmarks/utils.R: NMSE = SSE / ((n-1)*var(y)),
#     recovered iff NMSE < 1e-4, computed from the best member's predictions.
#
# Usage (from project root):
#   julia +release benchmarks/03_sr_comparison.jl
# Optional first arg: "base" (default, no sqrt) or "sqrt" (adds sqrt, includes N8).

using SymbolicRegression
using DelimitedFiles
using Statistics
using Random
using Printf
using Dates

# Hyperparameters = PySR's VERIFIED defaults, read directly from the installed
# pysr/sr.py (PySRRegressor.__init__). An earlier version of this file mislabelled
# SymbolicRegression.jl's defaults (populations=15, population_size=33) as "PySR
# defaults" and used niterations=40 — both wrong. The values below are PySR's actual
# defaults, so this is genuinely "PySR at its default settings", run via the same engine
# (SymbolicRegression.jl) but without the broken Python wrapper. See docs/15.
#
# The ONLY deliberate deviations from PySR's defaults are documented and necessary:
#   1. operators are matched to the rsymbolic2 gate set (a same-search-space comparison
#      requires it; PySR's own default has no unary operators);
#   2. serial execution + a wall-clock backstop (reproducibility and a safety bound;
#      PySR defaults to multithreaded, no timeout).
# In particular early stopping is left at PySR's default (None): each run uses its full
# NITERATIONS budget, exactly as default PySR would. rsymbolic2's gate, by contrast, uses
# its own default early stop — i.e. each tool runs at its own defaults (defaults vs
# defaults), which is the honest cross-tool comparison.
const NITERATIONS    = 100       # PySR default (sr.py)
const POPULATIONS    = 31        # PySR default (sr.py)
const POPULATION_SZ  = 27        # PySR default (sr.py)
const NCYCLES        = 380       # PySR default ncycles_per_iteration (sr.py)
const TOURNAMENT_N   = 15        # PySR default tournament_selection_n (sr.py)
const PARSIMONY      = 0.0       # PySR default (sr.py)
const MAXSIZE        = 30        # PySR default (sr.py); also bounds per-iteration cost
const TIMEOUT_S      = 120.0     # backstop only (not a PySR default); NITERATIONS is the bound
const N_SEEDS        = 5
const RECOVERY_THRESHOLD = 1e-4  # identical to utils.R

# ---- operator set (maps the rsymbolic2 gate set) ----------------------------
# rsymbolic2 binary {add,sub,mul,div} -> [+,-,*,/]; unary {neg,exp,log,sin,cos} ->
# [exp,log,sin,cos]. `neg` is omitted because unary negation is expressible via binary
# subtraction (0 - x); documented in docs/15. The "sqrt" mode adds sqrt and N8.
mode = length(ARGS) >= 1 ? ARGS[1] : "base"
unary_ops = mode == "sqrt" ? [exp, log, sin, cos, sqrt] : [exp, log, sin, cos]

# Problems to run. N8 only makes sense with sqrt enabled.
base_problems = ["N1", "N2", "N3", "N4", "N5", "N6", "N7", "N9", "N10"]
problems = mode == "sqrt" ? vcat(base_problems, "N8") : base_problems

script_dir = @__DIR__
data_dir   = joinpath(script_dir, "data")
results_dir = joinpath(script_dir, "results")
isdir(results_dir) || mkdir(results_dir)

function load_problem(pid)
    path = joinpath(data_dir, "nguyen_$(pid).csv")
    raw, header = readdlm(path, ','; header = true)
    ncol = size(raw, 2)
    p = ncol - 1
    X = permutedims(Float64.(raw[:, 1:p]))   # SR.jl wants (n_features, n_samples)
    y = Float64.(raw[:, ncol])
    return X, y
end

# One search; returns (nmse, complexity, expr, elapsed_s, timed_out).
function run_one(X, y, options, seed)
    Random.seed!(seed)
    t0 = time()
    # verbosity=0, progress=false: SR.jl is otherwise extremely chatty (per-iteration
    # progress bars + Hall-of-Fame tables). When that stdout is captured through a pipe,
    # a full pipe buffer BLOCKS the Julia process — observed as ~19 min "runs" that used
    # almost no CPU (see docs/15). Silencing the engine makes wall-time == compute time.
    hof = equation_search(X, y; niterations = NITERATIONS, options = options,
                          parallelism = :serial, verbosity = 0, progress = false)
    elapsed = time() - t0
    dom = calculate_pareto_frontier(hof)
    # lowest-loss member on the frontier
    losses = [m.loss for m in dom]
    best = dom[argmin(losses)]
    pred, ok = eval_tree_array(best.tree, X, options)
    n = length(y)
    denom = (n - 1) * var(y)
    nmse = (ok && all(isfinite, pred) && denom > 0) ?
           sum((pred .- y) .^ 2) / denom : Inf
    expr = string_tree(best.tree, options)
    complexity = compute_complexity(best, options)
    timed_out = elapsed >= TIMEOUT_S - 1
    return nmse, complexity, expr, elapsed, timed_out
end

function make_options()
    Options(
        binary_operators = [+, -, *, /],
        unary_operators  = unary_ops,
        # PySR's verified defaults (see constants above):
        populations           = POPULATIONS,
        population_size        = POPULATION_SZ,
        ncycles_per_iteration  = NCYCLES,
        tournament_selection_n = TOURNAMENT_N,
        parsimony              = PARSIMONY,
        maxsize                = MAXSIZE,
        # Deviations (documented): backstop timeout only. early stop left at PySR default
        # (None) so each run uses its full NITERATIONS budget.
        timeout_in_seconds = TIMEOUT_S,
    )
end

println("SymbolicRegression.jl comparison  (mode=", mode, ")")
println("SR.jl version: ", pkgversion(SymbolicRegression))
println("Julia version: ", VERSION)
println("Date: ", today())
println("unary_ops: ", unary_ops)
println(repeat("-", 60))

options = make_options()

# ---- JIT warm-up (untimed) --------------------------------------------------
print("warming up JIT ... ")
let
    Xw = reshape(collect(range(-1.0, 1.0; length = 20)), 1, :)
    yw = 1.5 .* Xw[1, :] .+ 0.3
    warmopts = Options(binary_operators = [+, -, *, /], unary_operators = unary_ops,
                       populations = 4, population_size = 20, maxsize = MAXSIZE,
                       timeout_in_seconds = 10.0)
    Random.seed!(0)
    equation_search(Xw, yw; niterations = 3, options = warmopts, parallelism = :serial,
                    verbosity = 0, progress = false)
end
println("done")

rows = String[]
push!(rows, "problem_id,seed,elapsed_sec,nmse,recovered,timed_out,complexity,expression")

for pid in problems
    X, y = load_problem(pid)
    for seed in 1:N_SEEDS
        nmse, cplx, expr, elapsed, timed_out = run_one(X, y, options, seed)
        recovered = isfinite(nmse) && nmse < RECOVERY_THRESHOLD
        @printf("  %-4s seed=%d  NMSE=%.1e  time=%.0fs  %s%s\n",
                pid, seed, nmse, elapsed,
                recovered ? "RECOVERED" : "not recovered",
                timed_out ? "  [TIMED OUT]" : "")
        flush(stdout)   # stream progress to the log so the run is observable
        expr_clean = replace(expr, ',' => ';', '\n' => ' ')
        push!(rows, @sprintf("%s,%d,%.2f,%.6e,%s,%s,%d,%s",
              pid, seed, elapsed, nmse, recovered, timed_out, cplx, expr_clean))
    end
end

stamp = Dates.format(today(), "yyyymmdd")
outpath = joinpath(results_dir, "sr_comparison_$(mode)_$(stamp).csv")
open(outpath, "w") do io
    for r in rows
        println(io, r)
    end
end
println(repeat("-", 60))
println("Results written to: ", outpath)
