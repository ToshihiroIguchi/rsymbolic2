# Diagnostic harness for the `interference` Feynman problem (docs/38).
#
# Purpose: settle whether the rsymbolic2-vs-PySR `interference` recovery gap is
# (a) threshold noise, (b) exploration-bound, or (c) polish-bound. Unlike
# diag_multivar.R (whose defaults are NOT the parity config), this script runs
# the EXACT Stage 1 dev-gate parity hyperparameters (02_feynman_gate.R
# BENCH_PARAMS: pop=27, islands=31, tournament=15, gens=2800, scaling=1040, ...)
# and only overrides `generations` and the seed count from the CLI, so a run here
# reproduces the gate's known per-seed numbers (seed 1 @2800 NMSE ~= 1.19e-3).
#
# For every run it captures, beyond the reported best loss:
#   - the full Pareto front (complexity, loss -> NMSE, expression), written long,
#   - the front-minimum NMSE and that expression (for structural inspection:
#     does the target backbone I1 + I2 + 2*sqrt(I1*I2)*cos(delta) already appear
#     with imperfect constants [polish-bound] or is it absent [exploration-bound]?).
#
# Usage (PowerShell + full-path Rscript; set OMP_NUM_THREADS to physical cores):
#   $env:OMP_NUM_THREADS=6
#   Rscript benchmarks/diag_interference.R seeds=15 gens=2800 tag=p1
#   Rscript benchmarks/diag_interference.R seeds=8  gens=8400 timeout=400 tag=sweep
#
# Recognised key=value args (all optional):
#   seeds    integer, number of seeds 1..N   (default 5)
#   gens     override generations            (default 2800, the parity value)
#   timeout  per-run wall-clock seconds       (default 300)
#   key      problem key                      (default interference)
#   tag      string appended to output files  (default = paste0("g", gens))

# ---- Setup ------------------------------------------------------------------

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
source(file.path(script_dir, "feynman_datasets.R"))
source(file.path(script_dir, "utils.R"))

if (!requireNamespace("rsymbolic2", quietly = TRUE)) {
  stop("rsymbolic2 is not installed. ",
       "Run: install.packages('r-package/rsymbolic2', repos = NULL, type = 'source')")
}
library(rsymbolic2)

# ---- Argument parsing -------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- regmatches(args, regexpr(paste0("^", name, "=.*"), args))
  hit <- hit[nzchar(hit)]
  if (length(hit) == 0L) return(default)
  sub(paste0("^", name, "="), "", hit[[1L]])
}

KEY     <- get_arg("key", "interference")
N_SEEDS <- as.integer(get_arg("seeds", "5"))
GENS    <- as.integer(get_arg("gens", "2800"))
TIMEOUT <- as.numeric(get_arg("timeout", "300"))
TAG     <- get_arg("tag", paste0("g", GENS))

# Parity hyperparameters: a verbatim copy of 02_feynman_gate.R BENCH_PARAMS
# (Stage 1), so this run is faithful to the dev gate. `generations` and
# `timeout_seconds` are the only fields overridden from the CLI. Sourcing the
# gate script would execute the whole gate, so the list is copied here; if the
# gate parity values change, update both (they are pinned for self-documentation).
BENCH_PARAMS <- list(
  population_size            = 27L,
  n_populations              = 31L,
  generations                = GENS,
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
  timeout_seconds            = TIMEOUT
)

DATA_SEED <- 42L  # matches export_feynman_data.R DATA_SEED_TRAIN / the gate

# ---- Run --------------------------------------------------------------------

cat("interference diagnostic (docs/38)\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat(sprintf("key=%s  seeds=%d  gens=%d  timeout=%gs  tag=%s\n",
            KEY, N_SEEDS, GENS, TIMEOUT, TAG))
cat(sprintf("pop=%d islands=%d tournament=%d scaling=%g optprob=%g maxsize=%d\n",
            BENCH_PARAMS$population_size, BENCH_PARAMS$n_populations,
            BENCH_PARAMS$tournament_size, BENCH_PARAMS$adaptive_parsimony_scaling,
            BENCH_PARAMS$optimize_probability, BENCH_PARAMS$max_nodes))
cat(rep("-", 70), "\n", sep = "")

ds   <- feynman_dataset(KEY, n = 1000L, data_seed = DATA_SEED, split = "train")
prob <- feynman_problems[[KEY]]
cat("formula:", prob$formula, "  n_vars:", prob$n_vars, "\n")
cat(rep("-", 70), "\n", sep = "")

run_rows   <- vector("list", N_SEEDS)
front_rows <- list()

for (seed in seq_len(N_SEEDS)) {
  t0 <- proc.time()[["elapsed"]]
  result <- tryCatch(
    do.call(symbolic_regression, c(
      list(X = ds$X, y = ds$y, seed = as.integer(seed)),
      BENCH_PARAMS
    )),
    error = function(e) {
      message("  ERROR seed=", seed, ": ", conditionMessage(e))
      NULL
    }
  )
  elapsed   <- proc.time()[["elapsed"]] - t0
  timed_out <- isTRUE(elapsed >= TIMEOUT - 1)

  if (is.null(result)) {
    best_nmse <- Inf; best_expr <- NA_character_; loss_val <- Inf
    fmin_nmse <- Inf; fmin_expr <- NA_character_; fmin_cx <- NA_integer_
  } else {
    loss_val  <- result$loss
    best_nmse <- compute_nmse(result$loss, ds$y)
    best_expr <- result$expression

    pf <- result$pareto_front
    if (is.data.frame(pf) && nrow(pf) > 0L) {
      pf_nmse <- vapply(pf$loss, compute_nmse, numeric(1L), y = ds$y)
      k <- which.min(pf_nmse)
      fmin_nmse <- pf_nmse[k]
      fmin_expr <- as.character(pf$expression[k])
      fmin_cx   <- as.integer(pf$complexity[k])
      # archive the whole front (long) for structural grep
      front_rows[[length(front_rows) + 1L]] <- data.frame(
        gens = GENS, seed = seed,
        complexity = as.integer(pf$complexity),
        loss = pf$loss, nmse = pf_nmse,
        expression = as.character(pf$expression),
        stringsAsFactors = FALSE
      )
    } else {
      fmin_nmse <- best_nmse; fmin_expr <- best_expr; fmin_cx <- NA_integer_
    }
  }

  rec <- is_recovered(best_nmse)
  run_rows[[seed]] <- data.frame(
    key = KEY, gens = GENS, seed = seed,
    elapsed_sec = round(elapsed, 2L),
    loss = loss_val, nmse = best_nmse,
    front_min_nmse = fmin_nmse, front_min_complexity = fmin_cx,
    recovered = rec, timed_out = timed_out,
    best_expression = best_expr,
    front_min_expression = fmin_expr,
    stringsAsFactors = FALSE
  )

  cat(sprintf("  seed=%2d  NMSE=%.2e  front_min=%.2e (cx=%s)  time=%4.0fs  %s%s\n",
              seed, best_nmse, fmin_nmse,
              if (is.na(fmin_cx)) "?" else as.character(fmin_cx),
              elapsed,
              if (rec) "RECOVERED" else "not recovered",
              if (timed_out) " [TIMED OUT]" else ""))
}

# ---- Summary and save -------------------------------------------------------

df <- do.call(rbind, run_rows[!vapply(run_rows, is.null, logical(1L))])

finite_nmse <- df$nmse[is.finite(df$nmse)]
n_rec <- sum(df$recovered, na.rm = TRUE)
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("recovered %d/%d   median NMSE=%.2e   min NMSE=%.2e   max NMSE=%.2e\n",
            n_rec, nrow(df),
            if (length(finite_nmse)) median(finite_nmse) else Inf,
            if (length(finite_nmse)) min(finite_nmse) else Inf,
            if (length(finite_nmse)) max(finite_nmse) else Inf))
cat(sprintf("median time=%.0fs\n", median(df$elapsed_sec)))

out_dir <- file.path(script_dir, "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
runs_path  <- file.path(out_dir, sprintf("diag_interference_%s_runs.csv", TAG))
front_path <- file.path(out_dir, sprintf("diag_interference_%s_front.csv", TAG))
write.csv(df, runs_path, row.names = FALSE)
message("Runs written to: ", runs_path)
if (length(front_rows) > 0L) {
  write.csv(do.call(rbind, front_rows), front_path, row.names = FALSE)
  message("Fronts written to: ", front_path)
}

invisible(df)
