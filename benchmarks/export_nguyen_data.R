# Export the exact Nguyen training data (X, y) used by the rsymbolic2 gate so that
# SymbolicRegression.jl can be benchmarked on bit-identical samples. R's RNG stream
# (Mersenne-Twister, set.seed(123)) cannot be reproduced from Julia, so the only way
# to guarantee both tools see the same data is to materialise it here and have Julia
# read it back. See docs/15.
#
# Output: benchmarks/data/nguyen_<id>.csv with columns x1..xp, y. One file per problem.
# ALL problems are exported (including N8); the skip flag only governs which operator
# set rsymbolic2 uses, not whether the data exists.
#
# Usage (from project root):
#   Rscript benchmarks/export_nguyen_data.R

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
source(file.path(script_dir, "nguyen_datasets.R"))

out_dir <- file.path(script_dir, "data")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

DATA_SEED <- 123L   # identical to 01_nguyen_gate.R / 01b_nguyen_gate_sqrt.R

for (pid in names(nguyen_problems)) {
  ds <- nguyen_dataset(pid, n = 20L, data_seed = DATA_SEED)
  df <- as.data.frame(ds$X)
  names(df) <- paste0("x", seq_len(ncol(ds$X)))
  df$y <- ds$y
  path <- file.path(out_dir, paste0("nguyen_", pid, ".csv"))
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("wrote %s  (n=%d, p=%d, formula=%s)\n",
              path, nrow(df), ncol(ds$X), ds$formula))
}
cat("DONE export\n")
