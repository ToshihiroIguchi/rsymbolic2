# One-off: is the serial island-init the scaling bottleneck, or the parallel
# evolution region? Compare np=1 vs np=4 cpu/wall at a small generation count
# (init-heavy) vs a large one (evolution-heavy). If scaling improves a lot when
# generations rises, the serial initialize_island loop dominates. Bounded by timeout.
source("benchmarks/feynman_datasets.R")
suppressMessages(library(rsymbolic2))
ds <- feynman_dataset("center_mass", n = 1000L, data_seed = 42L)
base <- list(
  population_size = 200L, tournament_size = 5L,
  unary_ops = c("neg","exp","log","sin","cos","sqrt","tanh","abs","square"),
  binary_ops = c("add","sub","mul","div","pow"),
  max_depth = 6L, max_nodes = 50L, target_loss = 1e-12, simplify = TRUE,
  crossover_probability = 0.5, seed = 1L, parsimony = 1e-3,
  optimize_probability = 0.1, timeout_seconds = 150
)
for (gens in c(40L, 200L)) {
  cat(sprintf("\n--- generations=%d ---\n", gens))
  w <- numeric(2); idx <- 1
  for (np in c(1L, 4L)) {
    t <- proc.time()
    r <- do.call(symbolic_regression, c(list(X = ds$X, y = ds$y,
              n_populations = np, generations = gens), base))
    dt <- proc.time() - t
    cpu <- dt[["user.self"]] + dt[["sys.self"]]
    w[idx] <- dt[["elapsed"]]; idx <- idx + 1
    cat(sprintf("np=%d  wall=%.1fs  cpu=%.1fs  cpu/wall=%.2f\n",
                np, dt[["elapsed"]], cpu, cpu/dt[["elapsed"]]))
  }
  cat(sprintf("speedup np1->np4 (lower wall=better): wall_np4/wall_np1=%.2f  (ideal ~1.0 for 4x work on 4 threads)\n",
              w[2]/w[1]))
}
