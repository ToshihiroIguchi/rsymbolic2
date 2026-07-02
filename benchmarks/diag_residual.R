# Residual-solvability runs for the separability oracle (Phase 0-D, part 2).
#
# diag_separability.R exports residual datasets (benchmarks/data/residual_*.csv)
# implied by the detected/oracle decomposition chains of the p = 0 problems.
# This script runs the EXACT Stage 1 parity configuration (the same BENCH_PARAMS
# copy as diag_interference.R) on such a CSV, answering: is the residual in the
# solved class? It is diag_interference.R with the Feynman dataset swapped for a
# generic x1..xn,y CSV; everything else is unchanged.
#
# Usage (PowerShell + full-path Rscript; set OMP_NUM_THREADS to physical cores):
#   Rscript benchmarks/diag_residual.R csv=benchmarks/data/residual_bose_q_oracle_train.csv seeds=5
#
# Recognised key=value args:
#   csv      path to the residual CSV (required; columns x1..xn then y)
#   seeds    integer, number of seeds 1..N   (default 5)
#   gens     override generations            (default 2800, the parity value)
#   timeout  per-run wall-clock seconds      (default 300)
#   tag      string for output file names    (default = CSV basename)

# ---- Setup ------------------------------------------------------------------

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
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

CSV     <- get_arg("csv", "")
if (!nzchar(CSV)) stop("csv= argument is required")
if (!file.exists(CSV)) stop("no such file: ", CSV)
N_SEEDS <- as.integer(get_arg("seeds", "5"))
GENS    <- as.integer(get_arg("gens", "2800"))
TIMEOUT <- as.numeric(get_arg("timeout", "300"))
TAG     <- get_arg("tag", sub("\\.csv$", "", basename(CSV)))

# Parity hyperparameters: verbatim copy of 02_feynman_gate.R BENCH_PARAMS
# (Stage 1), as in diag_interference.R; only `generations`/`timeout_seconds`
# come from the CLI.
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

# ---- Run --------------------------------------------------------------------

dat <- read.csv(CSV)
stopifnot("y" %in% names(dat))
X <- as.matrix(dat[, setdiff(names(dat), "y"), drop = FALSE])
y <- dat$y

cat("residual solvability run (Phase 0-D part 2)\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat(sprintf("csv=%s  n=%d  n_vars=%d  seeds=%d  gens=%d  timeout=%gs\n",
            CSV, nrow(X), ncol(X), N_SEEDS, GENS, TIMEOUT))
cat(rep("-", 70), "\n", sep = "")

run_rows <- vector("list", N_SEEDS)
for (seed in seq_len(N_SEEDS)) {
  t0 <- proc.time()[["elapsed"]]
  result <- tryCatch(
    do.call(symbolic_regression, c(
      list(X = X, y = y, seed = as.integer(seed)),
      BENCH_PARAMS
    )),
    error = function(e) {
      message("  ERROR seed=", seed, ": ", conditionMessage(e))
      NULL
    }
  )
  elapsed <- proc.time()[["elapsed"]] - t0

  if (is.null(result)) {
    best_nmse <- Inf; best_expr <- NA_character_; loss_val <- Inf
  } else {
    loss_val  <- result$loss
    best_nmse <- compute_nmse(result$loss, y)
    best_expr <- result$expression
  }
  rec <- is_recovered(best_nmse)
  run_rows[[seed]] <- data.frame(
    tag = TAG, gens = GENS, seed = seed,
    elapsed_sec = round(elapsed, 2L),
    loss = loss_val, nmse = best_nmse, recovered = rec,
    expression = best_expr, stringsAsFactors = FALSE)
  cat(sprintf("  seed=%2d  NMSE=%.2e  time=%4.0fs  %s\n",
              seed, best_nmse, elapsed,
              if (rec) "RECOVERED" else "not recovered"))
}

df <- do.call(rbind, run_rows[!vapply(run_rows, is.null, logical(1L))])
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("recovered %d/%d\n", sum(df$recovered, na.rm = TRUE), nrow(df)))

out_dir <- file.path(script_dir, "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, sprintf("diag_residual_%s_runs.csv", TAG))
write.csv(df, out_path, row.names = FALSE)
message("Runs written to: ", out_path)

invisible(df)
