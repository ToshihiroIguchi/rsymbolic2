# B2.8 sweep: parsimony in {0, 1e-4, 5e-4, 1e-3, 5e-3}
# Harness: slow cases (N9 s5, N10 s3) + fast recovery checks (N1, N5, N7)
# B1 default (optimize_probability = 0.1) is fixed throughout.
# For each parsimony: record wall time, nmse, recovered.
# Decision rule: pick the smallest parsimony whose fast-set recovery is
# unchanged from p=0 AND slow-run wall time is reduced; keep 0.0 if no
# value achieves this without losing recovery.

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
  optimize_probability  = 0.1,     # B1 default — fixed throughout this sweep
  timeout_seconds       = 120,     # 2-min cap per run
  verbosity             = 1L       # epoch log: size med/max per epoch
)
RECOVERY_THRESHOLD <- 1e-4

SWEEP_PARSIMONY <- c(0, 1e-4, 5e-4, 1e-3, 5e-3)

# Slow cases: the ones that blew up in the gate run (pre-B1).
SLOW_CASES <- list(list(id = "N9",  seed = 5L),
                   list(id = "N10", seed = 3L))
# Fast recovery checks: simple/trig/log. Fixed seeds for comparability.
FAST_CASES <- list(list(id = "N1", seed = 1L),
                   list(id = "N5", seed = 1L),
                   list(id = "N7", seed = 1L))

DATA_SEED <- 123L
rows <- list()

for (pars in SWEEP_PARSIMONY) {
  cat(sprintf("\n=== parsimony = %.0e (optimize_probability = 0.1 fixed) ===\n", pars))
  for (case in c(SLOW_CASES, FAST_CASES)) {
    ds <- nguyen_dataset(case$id, data_seed = DATA_SEED)
    params <- c(PARAMS_BASE, list(seed = case$seed, parsimony = pars,
                                   unary_ops = c("neg", "exp", "log", "sin", "cos")))
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(
      do.call(symbolic_regression, c(list(X = ds$X, y = ds$y), params)),
      error = function(e) { message("ERROR: ", e$message); NULL }
    )
    elapsed <- proc.time()[["elapsed"]] - t0
    nmse <- if (is.null(res)) Inf else compute_nmse(res$loss, ds$y)
    rec  <- is_recovered(nmse)
    cat(sprintf("  %s s%d  parsimony=%.0e  NMSE=%.1e  time=%.0fs  %s\n",
                case$id, case$seed, pars, nmse, elapsed,
                if (rec) "OK" else "FAIL"))
    rows[[length(rows) + 1L]] <- list(
      parsimony = pars, id = case$id, seed = case$seed,
      elapsed = round(elapsed, 1), nmse = nmse, recovered = rec
    )
  }
}

df <- do.call(rbind, lapply(rows, as.data.frame))
stamp <- format(Sys.Date(), "%Y%m%d")
out <- file.path(script_dir, "results", paste0("b2_sweep_", stamp, ".csv"))
write.csv(df, out, row.names = FALSE)
cat("\nResults written to:", out, "\n")

# Print summary table
cat("\n--- Summary ---\n")
cat(sprintf("%-8s %-5s %-5s %9s %12s %9s\n",
            "parsimony", "id", "seed", "elapsed_s", "nmse", "recovered"))
for (r in rows) {
  cat(sprintf("%-8.0e %-5s %-5d %9.0f %12.1e %9s\n",
              r$parsimony, r$id, r$seed, r$elapsed, r$nmse,
              if (r$recovered) "YES" else "NO"))
}

cat("\nNext step: apply the decision rule in docs/13 B2.8 and flip the\n")
cat("parsimony default if a value reduces slow-run time without losing recovery.\n")
