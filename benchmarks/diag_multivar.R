# Diagnostic harness for multivariate Feynman problems that fail recovery
# (center_mass, newtons_grav and friends). See docs/23 and the plan in
# .claude/plans/rsymbolic2-feynman-hazy-grove.md.
#
# This script is a DIAGNOSTIC tool, not a gate. It does not weaken any
# threshold. It runs the same data + hyperparameters as the Stage 1 dev gate
# (02_feynman_gate.R) but additionally captures, per run:
#   - reported NMSE (from result$loss; this already equals the best raw loss
#     ever archived in the HallOfFame across ALL complexities, because
#     hof.update(child) runs for every child — see evolutionary_search.cpp),
#   - the full Pareto front (complexity, loss -> NMSE, expression),
#   - the minimum NMSE anywhere on the front,
#   - wall time and a cpu/wall ratio (health indicator),
#   - optional per-epoch trajectory via verbosity=1 (redirect stderr to a file).
#
# Usage (PowerShell + full-path Rscript):
#   Rscript benchmarks/diag_multivar.R mode=front keys=center_mass,newtons_grav seeds=5 timeout=300
#   Rscript benchmarks/diag_multivar.R mode=front keys=rel_mass seeds=1 timeout=20   # quick self-test / health check
#   Rscript benchmarks/diag_multivar.R mode=traj  keys=center_mass seeds=1 timeout=120 2> results/traj_center_mass.log
#   Rscript benchmarks/diag_multivar.R mode=ablate keys=center_mass,newtons_grav seeds=3 timeout=120 parsimony=0
#
# Recognised key=value args (all optional except where noted):
#   mode      front | traj | ablate         (default front; only affects defaults below)
#   keys      comma-separated problem keys   (default center_mass,newtons_grav)
#   seeds     integer, number of seeds 1..N  (default 5)
#   timeout   per-run wall-clock seconds      (default 300)
#   parsimony override parsimony              (default shipped 1e-3)
#   scaling   override adaptive_parsimony_scaling (default shipped 0)
#   optprob   override optimize_probability   (default shipped 0.1)
#   gens      override generations            (default 500)
#   islands   override n_populations           (default 4, matching the gate)
#   verbosity 0 or 1                          (default 0; mode=traj forces 1)
#   tag       string appended to output file names (default = mode)

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

MODE      <- get_arg("mode", "front")
KEYS      <- strsplit(get_arg("keys", "center_mass,newtons_grav"), ",")[[1L]]
N_SEEDS   <- as.integer(get_arg("seeds", "5"))
TIMEOUT   <- as.numeric(get_arg("timeout", "300"))
PARSIMONY <- as.numeric(get_arg("parsimony", "1e-3"))
SCALING   <- as.numeric(get_arg("scaling", "0"))
OPTPROB   <- as.numeric(get_arg("optprob", "0.1"))
GENS      <- as.integer(get_arg("gens", "500"))
ISLANDS   <- as.integer(get_arg("islands", "4"))
VERBOSITY <- as.integer(get_arg("verbosity", if (MODE == "traj") "1" else "0"))
TAG       <- get_arg("tag", MODE)

# Hyperparameters identical to the Stage 1 dev gate (02_feynman_gate.R), except
# the knobs we may override for diagnosis. Keeping these in lockstep is what
# makes a quick run here reproduce the gate's known 0/5 result.
PARAMS <- list(
  population_size       = 500L,
  n_populations         = ISLANDS,
  generations           = GENS,
  tournament_size       = 5L,
  unary_ops             = c("neg", "exp", "log", "sin", "cos",
                            "sqrt", "tanh", "abs", "square"),
  binary_ops            = c("add", "sub", "mul", "div", "pow"),
  max_depth             = 6L,
  max_nodes             = 50L,
  target_loss           = 1e-10,
  simplify              = TRUE,
  crossover_probability = 0.5,
  timeout_seconds       = TIMEOUT,
  parsimony             = PARSIMONY,
  adaptive_parsimony_scaling = SCALING,
  optimize_probability  = OPTPROB,
  verbosity             = VERBOSITY
)

DATA_SEED <- 42L  # matches 02_feynman_gate.R / export_feynman_data.R train split

cat(sprintf("diag_multivar  mode=%s  tag=%s\n", MODE, TAG))
cat(sprintf("rsymbolic2 version: %s\n", as.character(packageVersion("rsymbolic2"))))
cat(sprintf("keys=%s  seeds=%d  islands=%d  timeout=%gs  parsimony=%g  scaling=%g  optprob=%g  gens=%d  verbosity=%d\n",
            paste(KEYS, collapse = ","), N_SEEDS, ISLANDS, TIMEOUT, PARSIMONY, SCALING, OPTPROB, GENS, VERBOSITY))
cat(rep("-", 78), "\n", sep = "")

# ---- Run --------------------------------------------------------------------

run_rows   <- list()   # one row per (key, seed): summary
front_rows <- list()   # long: one row per (key, seed, front member)
ri <- 0L; fi <- 0L

for (key in KEYS) {
  ds <- feynman_dataset(key, n = 1000L, data_seed = DATA_SEED, split = "train")

  for (seed in seq_len(N_SEEDS)) {
    t0 <- proc.time()
    result <- tryCatch(
      do.call(symbolic_regression,
              c(list(X = ds$X, y = ds$y, seed = as.integer(seed)), PARAMS)),
      error = function(e) {
        message("  ERROR key=", key, " seed=", seed, ": ", conditionMessage(e))
        NULL
      }
    )
    dt   <- proc.time() - t0
    wall <- as.numeric(dt[["elapsed"]])
    cpu  <- sum(dt[c("user.self", "sys.self", "user.child", "sys.child")],
                na.rm = TRUE)

    if (is.null(result)) {
      reported_nmse <- Inf; front_min_nmse <- Inf
      front_min_cx  <- NA_integer_; best_expr <- NA_character_
    } else {
      reported_nmse <- compute_nmse(result$loss, ds$y)
      best_expr     <- result$expression
      pf <- result$pareto_front
      # Per-front-member NMSE; the front is keyed on raw loss (parsimony-free).
      pf_nmse <- vapply(pf$loss, compute_nmse, numeric(1L), y = ds$y)
      front_min_nmse <- if (length(pf_nmse)) min(pf_nmse) else Inf
      front_min_cx   <- if (length(pf_nmse)) pf$complexity[which.min(pf_nmse)] else NA_integer_
      for (k in seq_len(nrow(pf))) {
        fi <- fi + 1L
        front_rows[[fi]] <- data.frame(
          key = key, seed = seed,
          complexity = pf$complexity[k],
          loss = pf$loss[k],
          nmse = pf_nmse[k],
          expression = pf$expression[k],
          stringsAsFactors = FALSE
        )
      }
    }

    ri <- ri + 1L
    run_rows[[ri]] <- data.frame(
      key = key, seed = seed,
      reported_nmse  = reported_nmse,
      front_min_nmse = front_min_nmse,
      front_min_complexity = front_min_cx,
      recovered_reported  = is_recovered(reported_nmse),
      recovered_front     = is_recovered(front_min_nmse),
      wall_sec = round(wall, 1L),
      cpu_sec  = round(cpu, 1L),
      cpu_per_wall = round(cpu / wall, 2L),
      best_expression = best_expr,
      stringsAsFactors = FALSE
    )

    cat(sprintf("  %-14s seed=%d  reportedNMSE=%.2e  frontMinNMSE=%.2e (cx=%s)  wall=%4.0fs  cpu/wall=%.1f%s\n",
                key, seed, reported_nmse, front_min_nmse,
                as.character(front_min_cx), wall, cpu / wall,
                if (is_recovered(front_min_nmse)) "  RECOVERED" else ""))
  }
}

# ---- Save + summarise -------------------------------------------------------

run_df   <- do.call(rbind, run_rows)
front_df <- if (length(front_rows)) do.call(rbind, front_rows) else NULL

save_results(run_df, paste0("diag_", TAG, "_runs"),
             out_dir = file.path(script_dir, "results"))
if (!is.null(front_df)) {
  save_results(front_df, paste0("diag_", TAG, "_front"),
               out_dir = file.path(script_dir, "results"))
}

cat("\n", rep("=", 78), "\n", sep = "")
cat("Per-problem summary (median over seeds)\n")
for (key in KEYS) {
  sub <- run_df[run_df$key == key, ]
  cat(sprintf("  %-14s  med reportedNMSE=%.2e  med frontMinNMSE=%.2e  recovered(front)=%d/%d  med cpu/wall=%.1f\n",
              key,
              median(sub$reported_nmse[is.finite(sub$reported_nmse)]),
              median(sub$front_min_nmse[is.finite(sub$front_min_nmse)]),
              sum(sub$recovered_front, na.rm = TRUE), nrow(sub),
              median(sub$cpu_per_wall, na.rm = TRUE)))
}
cat(rep("=", 78), "\n", sep = "")
cat("Interpretation guide:\n")
cat("  front_min_nmse < 1e-4  -> structure WAS reached; reporting/selection issue (H1).\n")
cat("  front_min_nmse >> 1e-4 -> structure never assembled; reachability issue (H2/H3).\n")
cat("  cpu/wall well below island count -> CPU starvation; fix environment before trusting numbers.\n")

invisible(list(runs = run_df, front = front_df))
