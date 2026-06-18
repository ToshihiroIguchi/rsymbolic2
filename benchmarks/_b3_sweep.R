# B3 sweep: frequency-based adaptive parsimony (adaptive_parsimony_scaling).
# Fixes parsimony = 1e-3 (shipped) and optimize_probability = 0.1 (shipped); varies
# adaptive_parsimony_scaling. Reuses the docs/13 B2 Nguyen no-regression set:
# slow cases (N9 s5, N10 s3) + fast checks (N1, N5, N7). For each scaling value:
# record wall time, NMSE, recovered. Decision rule: a scaling value is acceptable
# only if every fast case still recovers and no slow case blows up vs scaling=0.
# See docs/24.

library(rsymbolic2)
script_dir <- "C:/Users/toshi/python/rsymbolic2/benchmarks"
source(file.path(script_dir, "nguyen_datasets.R"))
source(file.path(script_dir, "utils.R"))

PARAMS_BASE <- list(
  population_size       = 500L,
  n_populations         = 4L,
  generations           = 200L,
  tournament_size       = 5L,
  binary_ops            = c("add", "sub", "mul", "div"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5,
  parsimony             = 1e-3,    # shipped — fixed throughout this sweep
  optimize_probability  = 0.1,     # shipped — fixed throughout this sweep
  timeout_seconds       = 120,     # 2-min cap per run
  verbosity             = 0L
)
RECOVERY_THRESHOLD <- 1e-4

# Allow overriding the swept values from the command line, e.g.
#   Rscript benchmarks/_b3_sweep.R 0,0.5,1
args <- commandArgs(trailingOnly = TRUE)
SWEEP_SCALING <- if (length(args) >= 1L)
  as.numeric(strsplit(args[[1L]], ",")[[1L]]) else c(0, 0.5)

SLOW_CASES <- list(list(id = "N9",  seed = 5L),
                   list(id = "N10", seed = 3L))
FAST_CASES <- list(list(id = "N1", seed = 1L),
                   list(id = "N5", seed = 1L),
                   list(id = "N7", seed = 1L))

DATA_SEED <- 123L
rows <- list()

for (scal in SWEEP_SCALING) {
  cat(sprintf("\n=== adaptive_parsimony_scaling = %g (parsimony = 1e-3 fixed) ===\n", scal))
  for (case in c(SLOW_CASES, FAST_CASES)) {
    ds <- nguyen_dataset(case$id, data_seed = DATA_SEED)
    params <- c(PARAMS_BASE, list(seed = case$seed,
                                  adaptive_parsimony_scaling = scal,
                                  unary_ops = c("neg", "exp", "log", "sin", "cos")))
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(
      do.call(symbolic_regression, c(list(X = ds$X, y = ds$y), params)),
      error = function(e) { message("ERROR: ", e$message); NULL }
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    nmse <- if (is.null(res)) Inf else compute_nmse(res$loss, ds$y)
    rec  <- is_recovered(nmse)
    cat(sprintf("  %s s%d  scaling=%g  NMSE=%.1e  time=%.0fs  %s\n",
                case$id, case$seed, scal, nmse, elapsed,
                if (rec) "OK" else "FAIL"))
    rows[[length(rows) + 1L]] <- list(
      scaling = scal, id = case$id, seed = case$seed,
      elapsed = round(elapsed, 1), nmse = nmse, recovered = rec
    )
  }
}

df <- do.call(rbind, lapply(rows, as.data.frame))
stamp <- format(Sys.Date(), "%Y%m%d")
out <- file.path(script_dir, "results", paste0("b3_sweep_", stamp, ".csv"))
write.csv(df, out, row.names = FALSE)
cat("\nResults written to:", out, "\n")

cat("\n--- Summary ---\n")
cat(sprintf("%-10s %-5s %-5s %9s %12s %9s\n",
            "scaling", "id", "seed", "elapsed_s", "nmse", "recovered"))
for (r in rows) {
  cat(sprintf("%-10g %-5s %-5d %9.0f %12.1e %9s\n",
              r$scaling, r$id, r$seed, r$elapsed, r$nmse,
              if (r$recovered) "YES" else "NO"))
}
