# Blowup Probe Benchmark
#
# Purpose: (1) Complete the gate table (N9 seed 5, N10 all seeds).
#          (2) Collect per-epoch instrumentation on the exact cases known to be slow,
#              to discriminate H1 (bloat) / H2 (LM ill-conditioning) / H3 (constant count).
#
# Usage:
#   Rscript benchmarks/02_blowup_probe.R
#
# Output:
#   benchmarks/results/blowup_probe_YYYYMMDD.csv   — per-run results
#   benchmarks/results/blowup_probe_YYYYMMDD.log   — full per-epoch stderr logs
#
# After running, read the log and apply the decision rule in docs/12 Step 5.

script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "benchmarks")
source(file.path(script_dir, "nguyen_datasets.R"))
source(file.path(script_dir, "utils.R"))

if (!requireNamespace("rsymbolic2", quietly = TRUE))
  stop("rsymbolic2 not installed")
library(rsymbolic2)

# Same hyperparameters as the gate benchmark, plus:
#   verbosity = 1     -> per-epoch log printed to stderr (captured below)
#   timeout_seconds   -> 300s per run (5 min hard ceiling)
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
  verbosity             = 1L,
  timeout_seconds       = 300   # 5-minute ceiling; any blowup case hits this
)

DATA_SEED  <- 123L
RUNS       <- list(
  list(id = "N9",  seed = 5L),    # DNF in gate run
  list(id = "N10", seed = 1L),    # not run at all
  list(id = "N10", seed = 2L),
  list(id = "N10", seed = 3L),
  list(id = "N10", seed = 4L),
  list(id = "N10", seed = 5L)
)

# Set up log file — capture stderr from C++ verbosity output.
results_dir <- file.path(script_dir, "results")
if (!dir.exists(results_dir)) dir.create(results_dir, recursive = TRUE)
stamp   <- format(Sys.Date(), "%Y%m%d")

# Note on epoch logs: the per-epoch diagnostics are written by C++ directly to the
# process stderr (fprintf). R-level sink() cannot capture C-level stderr, so we do NOT
# try to. Instead, redirect the whole Rscript stderr to a file at launch, e.g.:
#   Rscript benchmarks/02_blowup_probe.R 2> benchmarks/results/blowup_probe_YYYYMMDD.log
# The result CSV (below) is written from R and is independent of that capture.

cat("Blowup Probe Benchmark\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat("Date:", format(Sys.Date()), "\n")
cat("timeout_seconds = 300, verbosity = 1\n")
cat("Epoch logs go to stderr (redirect at launch to capture them).\n")
cat(rep("-", 60), "\n", sep = "")

rows <- list()

for (run in RUNS) {
  pid   <- run$id
  seed  <- run$seed
  ds    <- nguyen_dataset(pid, data_seed = DATA_SEED)

  cat(sprintf("\n--- %s seed=%d ---\n", pid, seed))
  # Emit a run separator on stderr too, so it interleaves with the C++ epoch lines.
  message(sprintf("=== %s seed=%d ===", pid, seed))

  t0 <- proc.time()[["elapsed"]]
  result <- tryCatch(
    do.call(symbolic_regression, c(
      list(X = ds$X, y = ds$y, seed = as.integer(seed)),
      BENCH_PARAMS
    )),
    error = function(e) {
      message("ERROR: ", conditionMessage(e))
      NULL
    }
  )
  elapsed <- proc.time()[["elapsed"]] - t0

  if (is.null(result)) {
    nmse <- Inf; expr_str <- NA_character_; loss <- Inf
  } else {
    nmse     <- compute_nmse(result$loss, ds$y)
    expr_str <- result$expression
    loss     <- result$loss
  }
  rec <- is_recovered(nmse)

  timed_out <- (!is.null(result)) && (elapsed >= BENCH_PARAMS$timeout_seconds * 0.95)

  cat(sprintf("  %s  seed=%d  NMSE=%.1e  time=%.0fs  %s%s\n",
              pid, seed, nmse, elapsed,
              if (rec) "RECOVERED" else "not recovered",
              if (timed_out) "  [TIMEOUT]" else ""))

  rows[[length(rows) + 1L]] <- data.frame(
    problem_id    = pid,
    seed          = seed,
    elapsed_sec   = round(elapsed, 1),
    loss          = loss,
    nmse          = nmse,
    recovered     = rec,
    timed_out     = timed_out,
    expression    = expr_str,
    stringsAsFactors = FALSE
  )
}

df <- do.call(rbind, rows)
csv_path <- file.path(results_dir, paste0("blowup_probe_", stamp, ".csv"))
write.csv(df, csv_path, row.names = FALSE)
cat("\nResults written to:", csv_path, "\n")
cat("Epoch logs were sent to stderr (see the launcher's redirect target).\n")
cat("\nNext step: read the log and apply the decision rule in docs/12 Step 5.\n")

invisible(df)
