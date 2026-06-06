# B1.8 sweep: optimize_probability in {1.0, 0.5, 0.2, 0.1}
# Harness: slow cases (N9 s5, N10 s3) + fast recovery checks (N1, N5, N7)
# For each p: record wall time, median tree size, median nconst, recovery.
# Decision rule: pick the smallest p whose fast-set recovery is unchanged from p=1.

library(rsymbolic2)
script_dir <- "C:/Users/toshi/python/rsymbolic2/benchmarks"
source(file.path(script_dir, "nguyen_datasets.R"))
source(file.path(script_dir, "utils.R"))

PARAMS_BASE <- list(
  population_size = 500L, n_populations = 4L, generations = 200L,
  tournament_size = 5L, binary_ops = c("add", "sub", "mul", "div"),
  max_depth = 6L, max_nodes = 50L, target_loss = 1e-10,
  simplify = TRUE, crossover_probability = 0.5,
  timeout_seconds = 120, verbosity = 1L   # 2-min cap per run; epoch log for size/nconst
)
# Reference loss at p=1.0 for fast set (N1/N5/N7), used to check recovery.
RECOVERY_THRESHOLD <- 1e-4

SWEEP_PROBS <- c(1.0, 0.5, 0.2, 0.1)

# Slow cases: the ones that blew up in the gate run.
SLOW_CASES  <- list(list(id = "N9", seed = 5L),
                    list(id = "N10", seed = 3L))
# Fast recovery checks: simple/trig/log. Fixed seeds for comparability.
FAST_CASES  <- list(list(id = "N1", seed = 1L),
                    list(id = "N5", seed = 1L),
                    list(id = "N7", seed = 1L))

DATA_SEED <- 123L
rows <- list()

for (p in SWEEP_PROBS) {
  cat(sprintf("\n=== optimize_probability = %.1f ===\n", p))
  for (case in c(SLOW_CASES, FAST_CASES)) {
    ds <- nguyen_dataset(case$id, data_seed = DATA_SEED)
    params <- c(PARAMS_BASE, list(seed = case$seed, optimize_probability = p))
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(do.call(symbolic_regression, c(list(X = ds$X, y = ds$y), params)),
                   error = function(e) { message("ERROR: ", e$message); NULL })
    elapsed <- proc.time()[["elapsed"]] - t0
    nmse <- if (is.null(res)) Inf else compute_nmse(res$loss, ds$y)
    rec  <- is_recovered(nmse)
    cat(sprintf("  %s s%d  p=%.1f  NMSE=%.1e  time=%.0fs  %s\n",
                case$id, case$seed, p, nmse, elapsed,
                if (rec) "OK" else "FAIL"))
    rows[[length(rows) + 1L]] <- list(
      prob = p, id = case$id, seed = case$seed,
      elapsed = round(elapsed, 1), nmse = nmse, recovered = rec
    )
  }
}

df <- do.call(rbind, lapply(rows, as.data.frame))
stamp <- format(Sys.Date(), "%Y%m%d")
out <- file.path(script_dir, "results", paste0("b1_sweep_", stamp, ".csv"))
write.csv(df, out, row.names = FALSE)
cat("\nResults written to:", out, "\n")

# Print summary table
cat("\n--- Summary ---\n")
cat(sprintf("%-6s %-5s %-5s %9s %12s %9s\n",
            "prob", "id", "seed", "elapsed_s", "nmse", "recovered"))
for (r in rows) {
  cat(sprintf("%-6.1f %-5s %-5d %9.0f %12.1e %9s\n",
              r$prob, r$id, r$seed, r$elapsed, r$nmse,
              if (r$recovered) "YES" else "NO"))
}
