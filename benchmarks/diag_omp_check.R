# Quick OpenMP / parallelism health probe (not a gate).
#
# Runs a FIXED amount of work (no timeout) with n_populations = 1 then 4 and
# compares wall time. Total work scales with n_populations, so:
#   - OpenMP active & healthy : wall(4) ~= wall(1), cpu/wall(4) ~= 4
#   - serial / no OpenMP      : wall(4) ~= 4 * wall(1), cpu/wall ~= 1
#   - parallel but starved    : wall(4) > wall(1) but threads still burn CPU
#
# Usage: Rscript benchmarks/diag_omp_check.R
source("benchmarks/feynman_datasets.R")
suppressMessages(library(rsymbolic2))

ds <- feynman_dataset("center_mass", n = 1000L, data_seed = 42L)
p <- list(
  population_size = 200L, generations = 40L, tournament_size = 5L,
  unary_ops  = c("neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"),
  binary_ops = c("add", "sub", "mul", "div", "pow"),
  max_depth = 6L, max_nodes = 50L, target_loss = 1e-12, simplify = TRUE,
  crossover_probability = 0.5, seed = 1L, parsimony = 1e-3,
  optimize_probability = 0.1, timeout_seconds = 0
)
for (np in c(1L, 4L)) {
  t  <- proc.time()
  r  <- do.call(symbolic_regression, c(list(X = ds$X, y = ds$y, n_populations = np), p))
  dt <- proc.time() - t
  cpu <- dt[["user.self"]] + dt[["sys.self"]]
  cat(sprintf("n_populations=%d  wall=%.1fs  cpu=%.1fs  cpu/wall=%.2f\n",
              np, dt[["elapsed"]], cpu, cpu / dt[["elapsed"]]))
}
