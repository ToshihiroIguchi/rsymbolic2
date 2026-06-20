# Reference comparison on the Feynman dev set: SymbolicRegression.jl (the engine
# behind PySR), driven directly from Julia. PySR is only a thin Python/scikit-learn
# wrapper over SymbolicRegression.jl; the search algorithm and constant optimisation
# live entirely in the Julia package, so benchmarking SymbolicRegression.jl directly
# is an equivalent — and cleaner — comparison than going through PySR's juliacall
# bridge (broken on this machine's Microsoft Store Python; see docs/15). This mirrors
# benchmarks/03_sr_comparison.jl (Nguyen) for the 25-equation Feynman dev set (docs/19).
#
# Data: bit-identical to the rsymbolic2 gate, exported by export_feynman_data.R into
# benchmarks/data/feynman_<I|II|III>_<key>_train.csv (1000 rows, columns x1..xp, y).
#
# Protocol (see docs/15 and docs/19 for the full reporting checklist):
#   - JIT is warmed up once before timing (compilation amortised once per session,
#     exactly as in real PySR usage — not gaming the clock).
#   - PySR strict defaults: parallelism=:multithreading, no timeout_in_seconds.
#     Thread budget is equalized on rsymbolic2's side via launch flag (-t 4).
#   - Operators are the shared problem input given identically to both tools (CLAUDE.md):
#     binary [+,-,*,/,^], unary [exp,log,sin,cos,sqrt,tanh,abs,square]. `neg` is omitted
#     because unary negation is expressible as 0 - x; `pow` maps to `^`, `square` to
#     SR.jl's built-in `square`. Output suppression (progress=false, verbosity=0) is
#     non-algorithmic; recorded as such.
#   - maxsize=30 is PySR's default and is NOT equalized to rsymbolic2's max_nodes=50:
#     each tool keeps its own complexity budget (defaults vs defaults). Recorded as a
#     caveat in docs/26.
#   - Recovery uses the SAME criterion as benchmarks/utils.R: NMSE = SSE / ((n-1)*var(y)),
#     recovered iff NMSE < 1e-4, computed from the best member's predictions.
#   - safe_pow caveat: SR.jl `^` and rsymbolic2 safe_pow differ for non-positive bases
#     with non-integer exponents. The Feynman dev domains keep `^` bases positive, so
#     this does not bite here; documented in docs/26 rather than altering PySR.
#
# Usage (from project root, T=4 threads to match rsymbolic2 gate n_populations=4):
#   julia +release -t 4 benchmarks/05_feynman_pysr_comparison.jl          > benchmarks/results/_feynman_sr.log 2>&1
#   julia +release -t 4 benchmarks/05_feynman_pysr_comparison.jl seeds=1  > benchmarks/results/_feynman_sr_sanity.log 2>&1
# Optional arg seeds=N overrides the seed count (default 5) for a fast sanity pass.

using SymbolicRegression
using DelimitedFiles
using Statistics
using Random
using Printf
using Dates

# Hyperparameters = PySR's VERIFIED defaults, read directly from the installed
# pysr/sr.py (PySRRegressor.__init__); identical to benchmarks/03_sr_comparison.jl.
# This is genuinely "PySR at its default settings", run via the same engine
# (SymbolicRegression.jl) but without the broken Python wrapper. See docs/15.
const NITERATIONS    = 100       # PySR default (sr.py)
const POPULATIONS    = 31        # PySR default (sr.py)
const POPULATION_SZ  = 27        # PySR default (sr.py)
const NCYCLES        = 380       # PySR default ncycles_per_iteration (sr.py)
const TOURNAMENT_N   = 15        # PySR default tournament_selection_n (sr.py)
const PARSIMONY      = 0.0       # PySR default (sr.py)
const MAXSIZE        = 30        # PySR default (sr.py); NOT equalized to rsymbolic2 max_nodes=50
const TIMEOUT_S      = 300.0     # wall-clock analysis flag only; NOT passed to SR.jl (PySR has no timeout default)
const RECOVERY_THRESHOLD = 1e-4  # identical to utils.R

# ---- seed count (optional ARG seeds=N; default 5) ---------------------------
# Parsed in a function so the assignment is not a script-level soft-scope local
# (a top-level `for` loop would create a new local `N_SEEDS` and silently keep
# the default — Julia warns about exactly this ambiguity).
function parse_seeds(args, default)
    n = default
    for a in args
        m = match(r"seeds=([0-9]+)", a)
        m !== nothing && (n = parse(Int, m.captures[1]))
    end
    n
end
const N_SEEDS = parse_seeds(ARGS, 5)

# ---- operator set (the shared problem input, per CLAUDE.md) -----------------
# Maps the rsymbolic2 gate set: binary {add,sub,mul,div,pow} -> [+,-,*,/,^];
# unary {neg,exp,log,sin,cos,sqrt,tanh,abs,square} -> [exp,log,sin,cos,sqrt,tanh,abs,square].
# `neg` omitted (0 - x); `pow` -> `^`; `square` is SR.jl's built-in x -> x*x.
binary_ops = [+, -, *, /, ^]
unary_ops  = [exp, log, sin, cos, sqrt, tanh, abs, square]

# ---- Feynman dev problems (key, data file). Order matches feynman_dev_keys. -
# Filenames follow export_feynman_data.R: feynman_<prefix>_<key>_train.csv, where
# the Roman-numeral prefix is derived from each equation's ID.
problems = [
    ("gaussian",       "feynman_I_gaussian_train.csv"),
    ("rel_mass",       "feynman_I_rel_mass_train.csv"),
    ("coulomb",        "feynman_I_coulomb_train.csv"),
    ("spring_pe",      "feynman_I_spring_pe_train.csv"),
    ("lorentz_x",      "feynman_I_lorentz_x_train.csv"),
    ("rel_mom",        "feynman_I_rel_mom_train.csv"),
    ("harmonic_ke",    "feynman_I_harmonic_ke_train.csv"),
    ("interference",   "feynman_I_interference_train.csv"),
    ("bohr_radius",    "feynman_I_bohr_radius_train.csv"),
    ("larmor_rad",     "feynman_I_larmor_rad_train.csv"),
    ("planck",         "feynman_I_planck_train.csv"),
    ("center_mass",    "feynman_I_center_mass_train.csv"),
    ("torque",         "feynman_I_torque_train.csv"),
    ("larmor_freq",    "feynman_I_larmor_freq_train.csv"),
    ("doppler_rel",    "feynman_I_doppler_rel_train.csv"),
    ("einstein_smol",  "feynman_I_einstein_smol_train.csv"),
    ("driven_osc",     "feynman_I_driven_osc_train.csv"),
    ("heat_conduct",   "feynman_II_heat_conduct_train.csv"),
    ("boltzmann_dist", "feynman_II_boltzmann_dist_train.csv"),
    ("clausius_moss",  "feynman_II_clausius_moss_train.csv"),
    ("clausius_moss2", "feynman_II_clausius_moss2_train.csv"),
    ("bohr_magneton",  "feynman_II_bohr_magneton_train.csv"),
    ("bose_einstein",  "feynman_III_bose_einstein_train.csv"),
    ("lens_eq",        "feynman_I_lens_eq_train.csv"),
    ("newtons_grav",   "feynman_I_newtons_grav_train.csv"),
]

script_dir  = @__DIR__
data_dir    = joinpath(script_dir, "data")
results_dir = joinpath(script_dir, "results")
isdir(results_dir) || mkdir(results_dir)

function load_problem(fname)
    path = joinpath(data_dir, fname)
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
    # progress=false, verbosity=0: output suppression only (non-algorithmic).
    # OS-level redirect (> file) + these flags keep wall-time == compute time
    # (a captured progress pipe stalled SR.jl ~19 min; see docs/15).
    hof = equation_search(X, y; niterations = NITERATIONS, options = options,
                          parallelism = :multithreading, verbosity = 0, progress = false)
    elapsed = time() - t0
    dom = calculate_pareto_frontier(hof)
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
        binary_operators = binary_ops,
        unary_operators  = unary_ops,
        # PySR's verified defaults (see constants above):
        populations           = POPULATIONS,
        population_size        = POPULATION_SZ,
        ncycles_per_iteration  = NCYCLES,
        tournament_selection_n = TOURNAMENT_N,
        parsimony              = PARSIMONY,
        maxsize                = MAXSIZE,
    )
end

println("SymbolicRegression.jl comparison — Feynman dev set")
println("SR.jl version: ", pkgversion(SymbolicRegression))
println("Julia version: ", VERSION)
println("Date: ", today())
println("seeds: ", N_SEEDS, "   threads: ", Threads.nthreads())
println("binary_ops: ", binary_ops)
println("unary_ops: ", unary_ops)
println("maxsize=", MAXSIZE, " (PySR default; NOT equalized to rsymbolic2 max_nodes=50)")
println(repeat("-", 60))

options = make_options()

# ---- JIT warm-up (untimed) --------------------------------------------------
print("warming up JIT ... ")
let
    Xw = reshape(collect(range(1.0, 3.0; length = 20)), 1, :)
    yw = 1.5 .* Xw[1, :] .+ 0.3
    warmopts = Options(binary_operators = binary_ops, unary_operators = unary_ops,
                       populations = 4, population_size = 20, maxsize = MAXSIZE)
    Random.seed!(0)
    equation_search(Xw, yw; niterations = 3, options = warmopts, parallelism = :multithreading,
                    verbosity = 0, progress = false)
end
println("done")

rows = String[]
push!(rows, "problem_id,seed,elapsed_sec,nmse,recovered,timed_out,complexity,expression")

for (key, fname) in problems
    X, y = load_problem(fname)
    for seed in 1:N_SEEDS
        nmse, cplx, expr, elapsed, timed_out = run_one(X, y, options, seed)
        recovered = isfinite(nmse) && nmse < RECOVERY_THRESHOLD
        @printf("  %-15s seed=%d  NMSE=%.1e  time=%4.0fs  %s%s\n",
                key, seed, nmse, elapsed,
                recovered ? "RECOVERED" : "not recovered",
                timed_out ? "  [TIMED OUT]" : "")
        flush(stdout)   # stream progress to the log so the run is observable
        expr_clean = replace(expr, ',' => ';', '\n' => ' ')
        push!(rows, @sprintf("%s,%d,%.2f,%.6e,%s,%s,%d,%s",
              key, seed, elapsed, nmse, recovered, timed_out, cplx, expr_clean))
    end
end

stamp = Dates.format(today(), "yyyymmdd")
outpath = joinpath(results_dir, "sr_comparison_feynman_$(stamp).csv")
open(outpath, "w") do io
    for r in rows
        println(io, r)
    end
end
println(repeat("-", 60))
println("Results written to: ", outpath)
