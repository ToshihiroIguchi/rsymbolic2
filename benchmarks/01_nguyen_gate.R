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
  crossover_probability = 0.5
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
      expression  = expr_str,
      stringsAsFactors = FALSE
    )

    cat(sprintf("  %s  seed=%d  NMSE=%.1e  time=%.0fs  %s\n",
                pid, seed, nmse, elapsed,
                if (rec) "RECOVERED" else "not recovered"))
  }
}

# ---- Summarize and Save -----------------------------------------------------

df <- do.call(rbind, rows[!vapply(rows, is.null, logical(1L))])
attr(df, "formulas") <- formulas

gate_pass <- print_summary(df, title = paste("Nguyen Gate Benchmark", Sys.Date()))

save_results(df, "nguyen_gate")

invisible(list(results = df, gate_pass = gate_pass))
