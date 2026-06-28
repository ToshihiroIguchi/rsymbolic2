# Phase 2 Gate Benchmark: Nguyen Symbolic Regression Suite
#
# Purpose: Determine whether rsymbolic2 can recover symbolic equations at
# sufficient rate to pass the Phase 2 gate (see docs/11_benchmark_implementation_plan.md).
#
# Usage (from project root, with rsymbolic2 installed):
#   Rscript benchmarks/01_nguyen_gate.R
#
# Or from an R session:
#   source("benchmarks/01_nguyen_gate.R")
#
# Output: CSV in benchmarks/results/ and a printed summary.

# ---- Setup ------------------------------------------------------------------

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
source(file.path(script_dir, "nguyen_datasets.R"))
source(file.path(script_dir, "utils.R"))

if (!requireNamespace("rsymbolic2", quietly = TRUE)) {
  stop(
    "rsymbolic2 is not installed. ",
    "Run: install.packages('r-package/rsymbolic2', repos = NULL, type = 'source')"
  )
}
library(rsymbolic2)

# ---- Fixed Hyperparameters --------------------------------------------------
# These are set BEFORE running and are not tuned on benchmark results.
# See docs/11_benchmark_implementation_plan.md Section 4 for rationale.

BENCH_PARAMS <- list(
  population_size       = 500L,
  n_populations         = 4L,
  generations           = 200L,
  tournament_size       = 5L,
  unary_ops             = c("neg", "exp", "log", "sin", "cos"),
  binary_ops            = c("add", "sub", "mul", "div"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5,
  # Safety net only: guards against a re-emergent runtime blowup (an unbounded
  # gate is what allowed the pre-B1/B2 96-minute hang; see docs/12, docs/14).
  # Post-B1/B2 the diagnosed slow cases finish in <60s, so no run should reach
  # this. A run that DOES hit it is a regression to investigate, not a pass
  # (the timed_out column below flags it). NOTE: optimize_probability and
  # parsimony are intentionally NOT set here so the gate exercises the shipped
  # defaults (0.1 / 1e-3).
  timeout_seconds       = 180
)

DATA_SEED <- 123L    # Fixed: same data across all SR seeds (only SR seed varies)
N_RUNS    <- 5L      # Seeds 1:5
PROBLEMS  <- names(nguyen_problems)

# ---- Run Benchmark ----------------------------------------------------------

cat("Nguyen Gate Benchmark\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat("Date:", format(Sys.Date()), "\n")
cat("Hyperparameters: pop=", BENCH_PARAMS$population_size,
    " islands=", BENCH_PARAMS$n_populations,
    " gens=", BENCH_PARAMS$generations, "\n")
cat(rep("-", 60), "\n", sep = "")

rows <- vector("list", length(PROBLEMS) * N_RUNS)
row_idx <- 0L
formulas <- list()

for (pid in PROBLEMS) {
  ds <- nguyen_dataset(pid, n = 20L, data_seed = DATA_SEED)
  formulas[[pid]] <- ds$formula

  if (ds$skip) {
    cat(sprintf("SKIP  %s  (%s)\n", pid, ds$skip_reason))
    next
  }

  for (seed in seq_len(N_RUNS)) {
    t0 <- proc.time()[["elapsed"]]

    result <- tryCatch(
      do.call(symbolic_regression, c(
        list(X = ds$X, y = ds$y, seed = as.integer(seed)),
        BENCH_PARAMS
      )),
      error = function(e) {
        message("  ERROR pid=", pid, " seed=", seed, ": ", conditionMessage(e))
        NULL
      }
    )

    elapsed <- proc.time()[["elapsed"]] - t0
    timed_out <- isTRUE(elapsed >= BENCH_PARAMS$timeout_seconds - 1)

    if (is.null(result)) {
      nmse <- Inf
      expr_str <- NA_character_
      loss <- Inf
    } else {
      nmse     <- compute_nmse(result$loss, ds$y)
      expr_str <- result$expression
      loss     <- result$loss
    }

    rec <- is_recovered(nmse)

    row_idx <- row_idx + 1L
    rows[[row_idx]] <- data.frame(
      problem_id  = pid,
      seed        = seed,
      elapsed_sec = round(elapsed, 2),
      loss        = loss,
      nmse        = nmse,
      recovered   = rec,
      timed_out   = timed_out,
      expression  = expr_str,
      stringsAsFactors = FALSE
    )

    cat(sprintf("  %s  seed=%d  NMSE=%.1e  time=%.0fs  %s%s\n",
                pid, seed, nmse, elapsed,
                if (rec) "RECOVERED" else "not recovered",
                if (timed_out) "  [TIMED OUT]" else ""))
  }
}

# ---- Summarize and Save -----------------------------------------------------

df <- do.call(rbind, rows[!vapply(rows, is.null, logical(1L))])
attr(df, "formulas") <- formulas

gate_pass <- print_summary(df, title = paste("Nguyen Gate Benchmark", Sys.Date()))

if (any(df$timed_out, na.rm = TRUE)) {
  cat("WARNING: one or more runs hit the safety timeout — investigate (see docs/14).\n")
}

save_results(df, "nguyen_gate", out_dir = file.path(script_dir, "results"))

invisible(list(results = df, gate_pass = gate_pass))
