# Feynman Stage 0 (smoke) and Stage 1 (dev gate) benchmark runner.
#
# Stage 0 — smoke: 10 pow-heavy equations, 3 runs each, 3-minute timeout.
#   Gate: every run returns finite NMSE (no NaN/Inf crash); pipeline stable.
# Stage 1 — dev gate: 25 dev-subset equations, 5 runs each, 5-minute timeout.
#   Gate: >= 18/25 problems recovered (majority across 5 seeds).
#
# Usage (from project root, rsymbolic2 installed):
#   Rscript benchmarks/02_feynman_gate.R               # Stage 0 smoke
#   Rscript benchmarks/02_feynman_gate.R stage=1       # Stage 1 dev gate (5 runs)
#   Rscript benchmarks/02_feynman_gate.R stage=1 runs=1  # 1-seed sanity pass
#
# `runs=N` overrides the per-problem seed count (Stage 1 only); the default is 5.
# A 1-seed pass de-risks the ~8h full gate by first checking whether a change
# moved recovery at all before committing to all 5 seeds.
#
# `scaling=X` overrides adaptive_parsimony_scaling (default: shipped value, 0/off).
# Used for the docs/27 A/B at 31x27 (scaling=0 vs the PySR-aligned ~20).
#
# `units` enables the opt-in dimensional-analysis screen (docs/46 section 6): each
# problem's SI units (feynman_datasets.R `units`/`y_unit`) are passed as
# X_units/y_units; every other setting stays at the frozen parity values. Results
# are written under a separate '<label>_units' name so a units run never
# overwrites the authoritative units-off CSV.
#
# See docs/19 for full protocol.

# ---- Setup ------------------------------------------------------------------

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
source(file.path(script_dir, "feynman_datasets.R"))
source(file.path(script_dir, "utils.R"))

if (!requireNamespace("rsymbolic2", quietly = TRUE)) {
  stop(
    "rsymbolic2 is not installed. ",
    "Run: install.packages('r-package/rsymbolic2', repos = NULL, type = 'source')"
  )
}
library(rsymbolic2)

# ---- Stage selection --------------------------------------------------------

args  <- commandArgs(trailingOnly = TRUE)
STAGE <- 0L
for (a in args) {
  m <- regmatches(a, regexpr("stage=[01]", a))
  if (length(m) > 0L) STAGE <- as.integer(sub("stage=", "", m))
}

# Fail-fast: abort the whole gate the moment a run grossly overshoots its budget
# (a runaway evaluation, not a problem that merely used its full budget). This
# stops PDCA cycles from wasting hours on a known-broken build. Pass "nofailfast"
# to run the full gate to completion regardless (e.g. final validation).
FAIL_FAST        <- !any(grepl("nofailfast", args, fixed = TRUE))
OVERSHOOT_FACTOR <- 1.5  # a run is a runaway iff elapsed > timeout * this

if (STAGE == 0L) {
  KEYS      <- feynman_smoke_keys
  N_RUNS    <- 3L
  label     <- "feynman_smoke"
  gate_desc <- "Stage 0 — smoke test"
} else {
  KEYS      <- feynman_dev_keys
  N_RUNS    <- 5L
  label     <- "feynman_gate"
  gate_desc <- "Stage 1 — dev gate"

  # Optional runs=N override (Stage 1 only): reduce the seed count for a fast
  # sanity pass. Default stays 5. Out-of-range or unparseable values are ignored.
  for (a in args) {
    m <- regmatches(a, regexpr("runs=[0-9]+", a))
    if (length(m) > 0L) {
      n <- as.integer(sub("runs=", "", m))
      if (!is.na(n) && n >= 1L) N_RUNS <- n
    }
  }
}

# Optional target=key1,key2 filter (any stage): restrict the run to specific
# problems for isolated diagnosis (e.g. a single regressed target). Unknown keys
# are warned and dropped; if none match, the full set runs. A subset run writes to
# a separate '<label>_diag' CSV so it never overwrites the authoritative dated
# full-gate result, and its overall gate verdict (vs the 18/25 threshold) is not
# meaningful — read the per-problem PASS/FAIL lines instead.
TARGET_SUBSET <- FALSE
for (a in args) {
  m <- regmatches(a, regexpr("target=[A-Za-z0-9_,]+", a))
  if (length(m) > 0L) {
    want    <- strsplit(sub("target=", "", m), ",", fixed = TRUE)[[1]]
    unknown <- setdiff(want, KEYS)
    if (length(unknown) > 0L)
      cat("WARNING: target= ignoring unknown key(s): ",
          paste(unknown, collapse = ", "), "\n", sep = "")
    sel <- intersect(KEYS, want)  # intersect preserves canonical KEYS order
    if (length(sel) > 0L) {
      KEYS          <- sel
      TARGET_SUBSET <- TRUE
      label         <- paste0(label, "_diag")
      cat("Target subset (diagnostic): ", paste(KEYS, collapse = ", "), "\n",
          sep = "")
    } else {
      cat("WARNING: target= matched no known keys; running the full set.\n")
    }
  }
}

# ---- Fixed hyperparameters (docs/19 §6; PySR default parity per docs/28) ----
# Every algorithmic setting equals PySR's installed default (docs/28 §A): these are
# also rsymbolic2's shipped defaults now, but they are pinned explicitly here so the
# run is self-documenting and immune to any future default drift. The operator set is
# the shared problem input (CLAUDE.md), identical to 05_feynman_pysr_comparison.jl:
# 8 unary {exp,log,sin,cos,sqrt,tanh,abs,square} + 5 binary {add,sub,mul,div,pow}.
# `neg` is intentionally OMITTED to match the PySR comparison (05_…jl): unary negation
# is expressible as 0 - x via `sub`, so no expressiveness is lost. Including it would
# (a) violate "operators given identically to both tools" and (b) skew the SR.jl op-arity
# split nbin/(nuna+nbin) from 5/13 to 5/14 (docs/29 A#13).
# The ONLY rsymbolic2-side equalization is timeout_seconds=300 (compute budget); SR.jl
# runs its full niterations=100 to completion. generations=2800 is the PySR
# niterations*ncycles mapping (docs/28 §C) and is timeout-bound here for hard problems.
# max_nodes=30 now matches PySR maxsize=30 (the old 50 was an un-equalized caveat in
# docs/26); it also sizes the adaptive-parsimony histogram for the 1040 scaling.

BENCH_PARAMS <- list(
  population_size            = 27L,
  n_populations              = 31L,
  generations                = 2800L,
  tournament_size            = 15L,
  unary_ops                  = c("exp", "log", "sin", "cos",
                                 "sqrt", "tanh", "abs", "square"),
  binary_ops                 = c("add", "sub", "mul", "div", "pow"),
  max_depth                  = 30L,
  max_nodes                  = 30L,
  target_loss                = 1e-10,
  simplify                   = TRUE,
  crossover_probability      = 0.0259,
  parsimony                  = 0.0,
  adaptive_parsimony_scaling = 1040.0,
  optimize_probability       = 0.14,
  tournament_selection_p     = 0.982,
  fraction_replaced_hof      = 0.0614,
  timeout_seconds            = 300L
)

# Stage 0 is a pipeline-health check, not a recovery measurement (docs/19 §6).
# A full-budget smoke run overshoots its time limit several-fold because the
# deadline cannot interrupt one in-flight LM fit(); use a lighter profile so the
# smoke run stays fast and predictable. Lower recovery here is expected and
# irrelevant — the Stage 0 gate is "all runs finite NMSE", not a threshold.
if (STAGE == 0L) {
  BENCH_PARAMS <- modifyList(BENCH_PARAMS, list(
    population_size = 200L,
    n_populations   = 2L,
    generations     = 100L,
    timeout_seconds = 60L
  ))
}

# Optional scaling=X override for the adaptive-parsimony A/B (docs/27). When
# absent the shipped default (0/off) is used. Parsed as a double so 0, 20, etc.
# all work; negative or unparseable values are ignored.
for (a in args) {
  m <- regmatches(a, regexpr("scaling=[0-9.eE+-]+", a))
  if (length(m) > 0L) {
    s <- suppressWarnings(as.numeric(sub("scaling=", "", m)))
    if (!is.na(s) && s >= 0) {
      BENCH_PARAMS$adaptive_parsimony_scaling <- s
      cat(sprintf("Override: adaptive_parsimony_scaling = %g\n", s))
    }
  }
}

# Optional parsimony=X override (docs/27). The shipped default keeps a small
# linear parsimony (1e-3) for bloat control; PySR ships parsimony=0 and relies on
# the frequency penalty alone. Set parsimony=0 to test PySR's faithful ON config
# (parsimony=0 + scaling=20). Negative/unparseable values are ignored.
for (a in args) {
  m <- regmatches(a, regexpr("parsimony=[0-9.eE+-]+", a))
  if (length(m) > 0L) {
    pv <- suppressWarnings(as.numeric(sub("parsimony=", "", m)))
    if (!is.na(pv) && pv >= 0) {
      BENCH_PARAMS$parsimony <- pv
      cat(sprintf("Override: parsimony = %g\n", pv))
    }
  }
}

# Optional `nostop` flag: disable rsymbolic2's loss-threshold early stop by setting
# target_loss = -Inf (the `loss < target_loss` test in evolutionary_search.cpp is then
# always false), forcing every problem to run the full `generations` budget. This is a
# throughput-measurement aid only — it makes rsymbolic2's halt behaviour match PySR's
# default (early_stop_condition=None, which runs all niterations to completion), so a
# wall-clock comparison reflects pure compute and not the differing stop criteria.
# It is NOT a parity/search setting: recovery is unaffected (an exact fit is recovered
# whether the search halts early or keeps running). Use with target=/runs= for a fast
# subset throughput probe.
if (any(grepl("nostop", args, fixed = TRUE))) {
  BENCH_PARAMS$target_loss <- -Inf
  cat("Override: target_loss = -Inf (early stop disabled; full-budget throughput run)\n")
}

# Optional gens=N / timeout=N overrides: raise the generation budget (and its
# wall-clock cap to match) for the docs/44 section-3 budget arm (e.g. gens=14000
# timeout=1500 = 5x the Stage-1 budget). A gens override suffixes the label so a
# budget run never overwrites the authoritative fixed-budget CSV.
for (a in args) {
  m <- regmatches(a, regexpr("gens=[0-9]+", a))
  if (length(m) > 0L) {
    g <- as.integer(sub("gens=", "", m))
    if (!is.na(g) && g >= 1L) {
      BENCH_PARAMS$generations <- g
      label <- paste0(label, "_g", g)
      cat(sprintf("Override: generations = %d\n", g))
    }
  }
}
for (a in args) {
  m <- regmatches(a, regexpr("timeout=[0-9]+", a))
  if (length(m) > 0L) {
    tv <- as.integer(sub("timeout=", "", m))
    if (!is.na(tv) && tv >= 1L) {
      BENCH_PARAMS$timeout_seconds <- tv
      cat(sprintf("Override: timeout_seconds = %d\n", tv))
    }
  }
}

# Optional `units` flag: opt-in dimensional analysis (docs/46). The per-problem
# X_units/y_units are added inside the run loop; nothing else changes.
UNITS_ON <- any(args == "units")
if (UNITS_ON) {
  label <- paste0(label, "_units")
  missing_units <- KEYS[vapply(
    KEYS, function(k) is.null(feynman_problems[[k]]$units), logical(1L)
  )]
  if (length(missing_units) > 0L)
    stop("units requested but no unit metadata for: ",
         paste(missing_units, collapse = ", "))
  cat("Override: dimensional analysis ON (X_units/y_units from feynman_datasets.R)\n")
}

DATA_SEED <- 42L   # Matches export_feynman_data.R DATA_SEED_TRAIN

# ---- Run --------------------------------------------------------------------

cat(gate_desc, "\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat("Date:", format(Sys.Date()), "\n")
cat(sprintf("Problems: %d  Runs/problem: %d  Timeout: %ds\n",
            length(KEYS), N_RUNS, BENCH_PARAMS$timeout_seconds))
cat(sprintf("pop=%d  islands=%d  gens=%d  tournament=%d  parsimony=%g  scaling=%g\n",
            BENCH_PARAMS$population_size,
            BENCH_PARAMS$n_populations,
            BENCH_PARAMS$generations,
            BENCH_PARAMS$tournament_size,
            if (is.null(BENCH_PARAMS$parsimony)) 1e-3
            else BENCH_PARAMS$parsimony,
            if (is.null(BENCH_PARAMS$adaptive_parsimony_scaling)) 0
            else BENCH_PARAMS$adaptive_parsimony_scaling))
cat(rep("-", 70), "\n", sep = "")

rows <- vector("list", length(KEYS) * N_RUNS)
row_idx <- 0L
aborted <- FALSE
overshoot_limit <- BENCH_PARAMS$timeout_seconds * OVERSHOOT_FACTOR

for (key in KEYS) {
  ds   <- feynman_dataset(key, n = 1000L, data_seed = DATA_SEED, split = "train")
  prob <- feynman_problems[[key]]

  run_params <- BENCH_PARAMS
  if (UNITS_ON) {
    run_params$X_units <- unname(prob$units)
    run_params$y_units <- prob$y_unit
  }

  for (seed in seq_len(N_RUNS)) {
    t0 <- proc.time()[["elapsed"]]

    result <- tryCatch(
      do.call(symbolic_regression, c(
        list(X = ds$X, y = ds$y, seed = as.integer(seed)),
        run_params
      )),
      error = function(e) {
        message("  ERROR key=", key, " seed=", seed, ": ", conditionMessage(e))
        NULL
      }
    )

    elapsed   <- proc.time()[["elapsed"]] - t0
    timed_out <- isTRUE(elapsed >= BENCH_PARAMS$timeout_seconds - 1L)

    if (is.null(result)) {
      nmse     <- Inf
      expr_str <- NA_character_
      loss_val <- Inf
    } else {
      nmse     <- compute_nmse(result$loss, ds$y)
      expr_str <- result$expression
      loss_val <- result$loss
    }

    rec <- is_recovered(nmse)

    row_idx <- row_idx + 1L
    rows[[row_idx]] <- data.frame(
      problem_id  = prob$id,
      key         = key,
      seed        = seed,
      elapsed_sec = round(elapsed, 2L),
      loss        = loss_val,
      nmse        = nmse,
      recovered   = rec,
      timed_out   = timed_out,
      overshoot_sec = if (timed_out)
                        round(max(0.0, elapsed - BENCH_PARAMS$timeout_seconds), 2L)
                      else NA_real_,
      n_vars      = prob$n_vars,
      expression  = expr_str,
      stringsAsFactors = FALSE
    )

    cat(sprintf("  %-14s  seed=%d  NMSE=%.1e  time=%4.0fs  %s%s\n",
                key, seed, nmse, elapsed,
                if (rec) "RECOVERED" else "not recovered",
                if (timed_out) "  [TIMED OUT]" else ""))

    if (FAIL_FAST && elapsed > overshoot_limit) {
      cat(sprintf(
        "\nABORT: %s seed=%d overshot the budget (%.0fs > %.0fs = %dx%g). ",
        key, seed, elapsed, overshoot_limit,
        BENCH_PARAMS$timeout_seconds, OVERSHOOT_FACTOR))
      cat("Fail-fast enabled; stopping the gate to avoid wasting time.\n")
      cat("(Pass 'nofailfast' to run the full gate to completion.)\n")
      aborted <- TRUE
      break
    }
  }
  if (aborted) break
}

# ---- Summary and gate -------------------------------------------------------

df <- do.call(rbind, rows[!vapply(rows, is.null, logical(1L))])

cat("\n", gate_desc, " — Summary ", format(Sys.Date()), "\n", sep = "")
cat(rep("=", 70), "\n", sep = "")

problems  <- unique(df$key)
n_pass    <- 0L
formulas  <- setNames(
  vapply(problems, function(k) feynman_problems[[k]]$formula, character(1L)),
  problems
)

for (key in problems) {
  sub  <- df[df$key == key, ]
  prob <- feynman_problems[[key]]
  rec  <- sum(sub$recovered, na.rm = TRUE)

  prob_pass <- if (STAGE == 0L) {
    all(is.finite(sub$nmse))   # Stage 0: all runs finite (no crash)
  } else {
    rec >= ceiling(N_RUNS / 2L)  # Stage 1: majority recovered
  }
  if (prob_pass) n_pass <- n_pass + 1L

  med_nmse <- median(sub$nmse[is.finite(sub$nmse)])
  med_time <- median(sub$elapsed_sec)

  cat(sprintf("  %-14s  %-30s  %d/%d  med NMSE=%.1e  med time=%3.0fs  %s\n",
              key,
              substr(prob$formula, 1L, 30L),
              rec, N_RUNS,
              med_nmse, med_time,
              if (prob_pass) "PASS" else "FAIL"))
}

cat(rep("=", 70), "\n", sep = "")

if (TARGET_SUBSET) {
  cat("NOTE: diagnostic target subset — the overall gate verdict below is not\n")
  cat("      meaningful; read the per-problem PASS/FAIL lines above.\n")
}

if (STAGE == 0L) {
  gate_pass <- n_pass == length(problems)
  cat(sprintf("Gate: %d/%d problems all-finite NMSE  -->  %s\n\n",
              n_pass, length(problems),
              if (gate_pass) "PASS" else "FAIL"))
} else {
  gate_threshold <- 18L
  gate_pass <- n_pass >= gate_threshold
  cat(sprintf("Gate: %d/%d problems recovered (threshold: %d/%d)  -->  %s\n\n",
              n_pass, length(problems), gate_threshold, length(problems),
              if (gate_pass) "PASS" else "FAIL"))
}

if (aborted) {
  gate_pass <- FALSE
  cat("NOTE: gate ABORTED early on a runaway run; results are PARTIAL and the\n")
  cat("      gate is recorded as FAIL. Investigate the overshoot before retrying.\n")
}

if (any(df$timed_out, na.rm = TRUE)) {
  cat("WARNING: one or more runs hit the timeout — check for search regressions.\n")
}

save_results(df, label, out_dir = file.path(script_dir, "results"))

invisible(list(results = df, gate_pass = gate_pass, aborted = aborted))
