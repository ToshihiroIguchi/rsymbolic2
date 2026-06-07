# Phase 2/3 Gate Benchmark — Operator-Extension Run (sqrt enabled)
#
# Purpose: the orthogonal lever to 01_nguyen_gate.R. That script measures whether
# B1/B2 (probabilistic constant optimization + static parsimony) regressed recovery
# with the ORIGINAL operator set. This script measures whether ADDING the sqrt
# operator (committed alongside tanh/abs) regresses recovery, and whether N8 = sqrt(x)
# — previously unrecoverable — now recovers. See docs/14, Step 1, Run 2.
#
# It deliberately mirrors 01_nguyen_gate.R exactly except for two changes:
#   1. "sqrt" is added to unary_ops (enlarging the search space for ALL problems).
#   2. N8's skip flag is cleared so it actually runs.
# Keeping this separate (rather than parameterising the main runner) preserves
# 01_nguyen_gate.R as a frozen, release-over-release regression artifact.
#
# Usage (from project root, with rsymbolic2 installed):
#   Rscript benchmarks/01b_nguyen_gate_sqrt.R

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

# Enable N8 for this run: sqrt is now in the operator set below, so the skip
# (which exists only because the default set omits sqrt) no longer applies.
nguyen_problems$N8$skip <- FALSE

# ---- Fixed Hyperparameters --------------------------------------------------
# Identical to 01_nguyen_gate.R BENCH_PARAMS except unary_ops gains "sqrt".

BENCH_PARAMS <- list(
  population_size       = 500L,
  n_populations         = 4L,
  generations           = 200L,
  tournament_size       = 5L,
  unary_ops             = c("neg", "exp", "log", "sin", "cos", "sqrt"),
  binary_ops            = c("add", "sub", "mul", "div"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5,
  timeout_seconds       = 180   # same safety net as the main runner (see docs/14)
)

DATA_SEED <- 123L
N_RUNS    <- 5L
PROBLEMS  <- names(nguyen_problems)

# ---- Run Benchmark ----------------------------------------------------------

cat("Nguyen Gate Benchmark — Operator-Extension Run (sqrt enabled)\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat("Date:", format(Sys.Date()), "\n")
cat("Hyperparameters: pop=", BENCH_PARAMS$population_size,
    " islands=", BENCH_PARAMS$n_populations,
    " gens=", BENCH_PARAMS$generations,
    " unary=", paste(BENCH_PARAMS$unary_ops, collapse = "/"), "\n")
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

gate_pass <- print_summary(df, title = paste("Nguyen Gate (sqrt) ", Sys.Date()))

if (any(df$timed_out, na.rm = TRUE)) {
  cat("WARNING: one or more runs hit the safety timeout — investigate (see docs/14).\n")
}

save_results(df, "nguyen_gate_sqrt", out_dir = file.path(script_dir, "results"))

invisible(list(results = df, gate_pass = gate_pass))
