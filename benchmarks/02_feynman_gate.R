# Feynman Stage 0 (smoke) and Stage 1 (dev gate) benchmark runner.
#
# Stage 0 — smoke: 10 pow-heavy equations, 3 runs each, 3-minute timeout.
#   Gate: every run returns finite NMSE (no NaN/Inf crash); pipeline stable.
# Stage 1 — dev gate: 25 dev-subset equations, 5 runs each, 5-minute timeout.
#   Gate: >= 18/25 problems recovered (majority across 5 seeds).
#
# Usage (from project root, rsymbolic2 installed):
#   Rscript benchmarks/02_feynman_gate.R            # Stage 0 smoke
#   Rscript benchmarks/02_feynman_gate.R stage=1    # Stage 1 dev gate
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
}

# ---- Fixed hyperparameters (docs/19 §6) ------------------------------------
# Not tuned on Feynman results; set before benchmarking.
# optimize_probability and parsimony: shipped defaults (0.1 / 1e-3).

BENCH_PARAMS <- list(
  population_size       = 500L,
  n_populations         = 4L,
  generations           = 500L,
  tournament_size       = 5L,
  unary_ops             = c("neg", "exp", "log", "sin", "cos",
                            "sqrt", "tanh", "abs", "square"),
  binary_ops            = c("add", "sub", "mul", "div", "pow"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5,
  timeout_seconds       = 300L
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

DATA_SEED <- 42L   # Matches export_feynman_data.R DATA_SEED_TRAIN

# ---- Run --------------------------------------------------------------------

cat(gate_desc, "\n")
cat("rsymbolic2 version:", as.character(packageVersion("rsymbolic2")), "\n")
cat("Date:", format(Sys.Date()), "\n")
cat(sprintf("Problems: %d  Runs/problem: %d  Timeout: %ds\n",
            length(KEYS), N_RUNS, BENCH_PARAMS$timeout_seconds))
cat(sprintf("pop=%d  islands=%d  gens=%d\n",
            BENCH_PARAMS$population_size,
            BENCH_PARAMS$n_populations,
            BENCH_PARAMS$generations))
cat(rep("-", 70), "\n", sep = "")

rows <- vector("list", length(KEYS) * N_RUNS)
row_idx <- 0L
aborted <- FALSE
overshoot_limit <- BENCH_PARAMS$timeout_seconds * OVERSHOOT_FACTOR

for (key in KEYS) {
  ds   <- feynman_dataset(key, n = 1000L, data_seed = DATA_SEED, split = "train")
  prob <- feynman_problems[[key]]

  for (seed in seq_len(N_RUNS)) {
    t0 <- proc.time()[["elapsed"]]

    result <- tryCatch(
      do.call(symbolic_regression, c(
        list(X = ds$X, y = ds$y, seed = as.integer(seed)),
        BENCH_PARAMS
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
