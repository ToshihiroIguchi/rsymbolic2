# Mechanism screen (Phase 0 diagnostic, docs/43-style): for the opt-in
# high-accuracy layer (CLAUDE.md "Opt-in high-accuracy options"), measure whether
# each CANDIDATE mechanism lifts SymbolicRegression.jl's recovery rate above p=0
# on the three hardest-known Feynman dev problems (planck, bose_einstein,
# interference — see docs/38's multi-seed audit) BEFORE implementing it in the
# C++ engine. This is NOT a PySR-default-parity comparison (benchmarks/05 is):
# every arm here deliberately turns on exactly one non-default SR.jl mechanism,
# all other settings staying at the verified PySR defaults used in 05. A mechanism
# that cannot lift recovery here is not worth building in C++.
#
# Data: same bit-identical CSVs as 05, from benchmarks/data/.
#
# Arms (one candidate mechanism each; "domain" combines the plausible
# physicist-authored constraints per problem):
#   baseline    - pure PySR defaults, nothing enabled (control).
#   nested      - generic nesting caps: exp!(exp), sin/cos!(sin|cos), sqrt!(sqrt).
#   constraints - pow argument-size cap (-1, 1): left subtree unrestricted, right
#                 subtree complexity <= 1 (PySR's own documented `constraints` example).
#   annealing   - annealing=true with SR.jl's OWN default alpha (3.17; verified
#                 below), instead of PySR's annealing=false (sr.py:981).
#   domain      - per-problem combination a physicist might plausibly write:
#                   planck, bose_einstein: nested exp!(exp|sin|cos|tanh) + pow (-1,1)
#                   interference:          constraints cos<=5, sqrt<=7 (argument
#                                          complexity caps) + nested cos!(cos|sin|exp)
#
# SR.jl kwarg syntax verified against the installed source
# (C:\Users\toshi\.julia\packages\SymbolicRegression\3nKj1\src\Options.jl):
#   - `nested_constraints`: Vector of `op => [op2 => max_nesting, ...]` pairs
#     (Options.jl:477-484 docstring; e.g. `[sin => [cos => 0], cos => [cos => 2]]`).
#   - `constraints`: Vector of pairs; unary entries are `op => Int` (max subtree
#     size of the argument), binary entries are `op => (Int, Int)` (max subtree
#     size of each argument, -1 = unrestricted) (Options.jl:292-299, PySR's
#     documented `constraints={"pow": (-1, 1)}` example). A single `constraints=`
#     array may mix unary and binary entries; `build_constraints` (Options.jl:50-98)
#     filters each against the relevant operator set via `haskey`, so unrelated
#     entries are silently ignored for the other arity.
#   - `annealing::Bool` / `alpha::Real` (Options.jl:547-548, 787-788). SR.jl's OWN
#     default (used whenever `defaults=` is left at `nothing`, i.e. every arm in
#     this script and in 05): `annealing=true, alpha=3.17` (Options.jl:1120-1121,
#     the `>= v"1.0.0"` branch of `default_options`). This is the value PySR
#     overrides to `annealing=false` (sr.py:981) for its own default; the
#     `annealing` arm here restores SR.jl's un-overridden default alpha.
#   - `timeout_in_seconds::Union{Nothing,Real}`: an `Options()` kwarg (NOT an
#     `equation_search` kwarg), checked in `SearchUtils.jl:366-367`
#     (`time() - start_time > options.timeout_in_seconds`). Unlike 05 (where PySR
#     has no timeout default and TIMEOUT_S is a harness-side wall-clock flag only),
#     this screen is diagnostic and bounds every run explicitly so a stuck arm
#     cannot stall the whole sweep.
#
# Usage (from project root; -t 4 matches rsymbolic2 gate n_populations=4, per 05):
#   julia +release -t 4 benchmarks/06_srjl_mechanism_screen.jl arms=baseline                    > benchmarks/results/_screen_baseline.log 2>&1
#   julia +release -t 4 benchmarks/06_srjl_mechanism_screen.jl arms=nested                       > benchmarks/results/_screen_nested.log 2>&1
#   julia +release -t 4 benchmarks/06_srjl_mechanism_screen.jl arms=constraints                  > benchmarks/results/_screen_constraints.log 2>&1
#   julia +release -t 4 benchmarks/06_srjl_mechanism_screen.jl arms=annealing                    > benchmarks/results/_screen_annealing.log 2>&1
#   julia +release -t 4 benchmarks/06_srjl_mechanism_screen.jl arms=domain                       > benchmarks/results/_screen_domain.log 2>&1
# Optional args (all overrides, harness-side only; SR.jl settings stay as coded):
#   arms=a,b,...      which arms to run (default: all five, comma-separated, in order above)
#   seeds=N           seed count 1..N (default 10)
#   keys=k1,k2,...    restrict to a subset of {planck, bose_einstein, interference}
#   niter_override=N  override niterations (for a fast syntax/sanity pass only)
# The CSV is opened in APPEND mode so arms can be run in separate invocations
# without clobbering earlier results; each invocation still writes the header
# once (only if the file does not already exist).

using SymbolicRegression
using DelimitedFiles
using Statistics
using Random
using Printf
using Dates

# ---- PySR-default baseline hyperparameters, identical to benchmarks/05 ------
const NITERATIONS_DEFAULT = 100      # PySR default (sr.py)
const POPULATIONS    = 31            # PySR default (sr.py)
const POPULATION_SZ  = 27            # PySR default (sr.py)
const NCYCLES        = 380           # PySR default ncycles_per_iteration (sr.py)
const TOURNAMENT_N   = 15            # PySR default tournament_selection_n (sr.py)
const PARSIMONY      = 0.0           # PySR default (sr.py)
const MAXSIZE        = 30            # PySR default (sr.py); NOT equalized to rsymbolic2 max_nodes=50
const TIMEOUT_S      = 600.0         # actually enforced (Options timeout_in_seconds), unlike 05
const RECOVERY_THRESHOLD = 1e-4      # identical to utils.R / 05
const SR_DEFAULT_ALPHA = 3.17        # SR.jl's own default alpha (Options.jl:1121), verified above

# ---- CLI arg parsing (each in its own function: a top-level `for` loop would
# create a new soft-scope local and silently keep the default; see 05) ---------
function parse_seeds(args, default)
    n = default
    for a in args
        m = match(r"seeds=([0-9]+)", a)
        m !== nothing && (n = parse(Int, m.captures[1]))
    end
    n
end
const N_SEEDS = parse_seeds(ARGS, 10)

function parse_niter_override(args, default)
    n = default
    for a in args
        m = match(r"niter_override=([0-9]+)", a)
        m !== nothing && (n = parse(Int, m.captures[1]))
    end
    n
end
const N_ITER = parse_niter_override(ARGS, NITERATIONS_DEFAULT)

function parse_keys(args)
    for a in args
        m = match(r"keys=([A-Za-z0-9_,]+)", a)
        m !== nothing && return String.(split(m.captures[1], ','))
    end
    return String[]
end
const ONLY_KEYS = parse_keys(ARGS)

const VALID_ARMS = ["baseline", "nested", "constraints", "annealing", "domain"]
function parse_arms(args, default)
    for a in args
        m = match(r"arms=([A-Za-z0-9_,]+)", a)
        m !== nothing && return String.(split(m.captures[1], ','))
    end
    return default
end
const ARMS = parse_arms(ARGS, VALID_ARMS)
for a in ARMS
    a in VALID_ARMS || error("unknown arm '$a' (valid: $(join(VALID_ARMS, ", ")))")
end

# ---- operator set (the shared problem input, per CLAUDE.md) -----------------
# Identical to 05: rsymbolic2 gate set, `neg` omitted (0 - x), `pow` -> `^`,
# `square` -> SR.jl's built-in x -> x*x.
binary_ops = [+, -, *, /, ^]
unary_ops  = [exp, log, sin, cos, sqrt, tanh, abs, square]

# ---- fixed problem set: only the three mechanism-screen targets -------------
problems = [
    ("interference",  "feynman_I_interference_train.csv"),
    ("planck",        "feynman_I_planck_train.csv"),
    ("bose_einstein", "feynman_III_bose_einstein_train.csv"),
]
if !isempty(ONLY_KEYS)
    sel = filter(p -> p[1] in ONLY_KEYS, problems)
    if isempty(sel)
        @warn "keys= matched none of the three screen problems; running all three." ONLY_KEYS
    else
        global problems = sel
    end
end

script_dir  = @__DIR__
data_dir    = joinpath(script_dir, "data")
results_dir = joinpath(script_dir, "results")
isdir(results_dir) || mkdir(results_dir)

function load_problem(fname)
    path = joinpath(data_dir, fname)
    raw, header = readdlm(path, ','; header = true)
    ncol = size(raw, 2)
    p = ncol - 1
    # PySR default precision=32 (sr.py:1030,:1870); see 05 for the rationale.
    X = permutedims(Float32.(raw[:, 1:p]))
    y = Float32.(raw[:, ncol])
    return X, y
end

# ---- per-arm Options() construction ------------------------------------------
# base_kwargs is the PySR-default layer (identical to 05's make_options); each
# arm merges in exactly the one mechanism it screens. `merge` (not kwarg splat +
# explicit kwarg) is used so an arm can override a key already in base_kwargs
# (e.g. `annealing`) without Julia's "duplicate keyword argument" error.
function base_kwargs()
    (
        binary_operators = binary_ops,
        unary_operators  = unary_ops,
        populations             = POPULATIONS,
        population_size         = POPULATION_SZ,
        ncycles_per_iteration   = NCYCLES,
        tournament_selection_n  = TOURNAMENT_N,
        parsimony               = PARSIMONY,
        maxsize                 = MAXSIZE,
        annealing               = false,  # PySR default (sr.py:981); overridden by the `annealing` arm
        timeout_in_seconds      = TIMEOUT_S,
    )
end

function arm_extra_kwargs(arm::String, key::String)
    if arm == "baseline"
        return NamedTuple()
    elseif arm == "nested"
        # Generic, domain-agnostic nesting caps.
        return (nested_constraints = [
            exp  => [exp => 0],
            sin  => [sin => 0, cos => 0],
            cos  => [sin => 0, cos => 0],
            sqrt => [sqrt => 0],
        ],)
    elseif arm == "constraints"
        # PySR's own documented `constraints` example: pow left unrestricted,
        # right (exponent) subtree complexity <= 1.
        return (constraints = [(^) => (-1, 1)],)
    elseif arm == "annealing"
        return (annealing = true, alpha = SR_DEFAULT_ALPHA)
    elseif arm == "domain"
        if key in ("planck", "bose_einstein")
            return (
                nested_constraints = [exp => [exp => 0, sin => 0, cos => 0, tanh => 0]],
                constraints        = [(^) => (-1, 1)],
            )
        elseif key == "interference"
            return (
                constraints        = [cos => 5, sqrt => 7],
                nested_constraints = [cos => [cos => 0, sin => 0, exp => 0]],
            )
        else
            error("domain arm has no mapping for key=$key")
        end
    else
        error("unknown arm: $arm")
    end
end

function build_options(arm::String, key::String)
    merged = merge(base_kwargs(), arm_extra_kwargs(arm, key))
    Options(; merged...)
end

# One search; returns (loss, nmse, expr, elapsed_s, timed_out).
function run_one(X, y, options, seed)
    Random.seed!(seed)
    t0 = time()
    hof = equation_search(X, y; niterations = N_ITER, options = options,
                          parallelism = :multithreading, verbosity = 0, progress = false)
    elapsed = time() - t0
    dom = calculate_pareto_frontier(hof)
    losses = [m.loss for m in dom]
    best = dom[argmin(losses)]
    pred, ok = eval_tree_array(best.tree, X, options)
    n = length(y)
    pred64 = Float64.(pred)
    y64 = Float64.(y)
    denom = (n - 1) * var(y64)
    nmse = (ok && all(isfinite, pred64) && denom > 0) ?
           sum((pred64 .- y64) .^ 2) / denom : Inf
    expr = string_tree(best.tree, options)
    timed_out = elapsed >= TIMEOUT_S - 1
    return best.loss, nmse, expr, elapsed, timed_out
end

println("SymbolicRegression.jl mechanism screen — Phase 0 diagnostic (not a parity run)")
println("SR.jl version: ", pkgversion(SymbolicRegression))
println("Julia version: ", VERSION)
println("Date: ", today())
println("arms: ", join(ARMS, ", "))
println("seeds: ", N_SEEDS, "   niterations: ", N_ITER, "   threads: ", Threads.nthreads())
println("problems: ", join(first.(problems), ", "))
println("binary_ops: ", binary_ops)
println("unary_ops: ", unary_ops)
println("maxsize=", MAXSIZE, "  timeout_in_seconds=", TIMEOUT_S, "  SR.jl default alpha=", SR_DEFAULT_ALPHA)
println(repeat("-", 60))

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

stamp = Dates.format(today(), "yyyymmdd")
outpath = joinpath(results_dir, "srjl_mechanism_screen_$(stamp).csv")

# Append mode: separate arm invocations accumulate into the same day's file.
# Header is written only once (file did not already exist before this run).
write_header = !isfile(outpath)
open(outpath, "a") do io
    if write_header
        println(io, "problem_id,key,arm,seed,elapsed_sec,loss,nmse,recovered,timed_out,expression")
    end
    flush(io)
    for (key, fname) in problems
        # problem_id is the full data-file stem (matches export_feynman_data.R /
        # 05's filename convention); key is the short screen-problem name.
        problem_id = replace(fname, "_train.csv" => "")
        X, y = load_problem(fname)
        for arm in ARMS
            options = build_options(arm, key)
            for seed in 1:N_SEEDS
                loss, nmse, expr, elapsed, timed_out = run_one(X, y, options, seed)
                recovered = isfinite(nmse) && nmse < RECOVERY_THRESHOLD
                @printf("  %-15s arm=%-11s seed=%d  NMSE=%.1e  time=%4.0fs  %s%s\n",
                        key, arm, seed, nmse, elapsed,
                        recovered ? "RECOVERED" : "not recovered",
                        timed_out ? "  [TIMED OUT]" : "")
                flush(stdout)
                expr_clean = replace(expr, ',' => ';', '\n' => ' ')
                println(io, @sprintf("%s,%s,%s,%d,%.2f,%.6e,%.6e,%s,%s,%s",
                      problem_id, key, arm, seed, elapsed, loss, nmse, recovered, timed_out, expr_clean))
                flush(io)
            end
        end
    end
end
println(repeat("-", 60))
println("Results written to: ", outpath)
