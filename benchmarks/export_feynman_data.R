# Export Feynman training and test data as CSV so that SymbolicRegression.jl
# can be benchmarked on bit-identical samples. R's RNG stream cannot be reproduced
# from Julia, so data must be materialised here and read back by the comparison
# harness. See docs/19 §2 for the train/test split design.
#
# Output:
#   benchmarks/data/feynman_I_<key>_train.csv  (1000 rows, columns x1..xp, y)
#   benchmarks/data/feynman_I_<key>_test.csv   (500 rows)
#   benchmarks/data/feynman_II_<key>_train.csv  (for Feynman II equations)
#   benchmarks/data/feynman_II_<key>_test.csv
#   benchmarks/data/feynman_III_<key>_train.csv
#   benchmarks/data/feynman_III_<key>_test.csv
#
# The prefix (I, II, III) is derived from the equation ID field.
#
# Usage (from project root):
#   Rscript benchmarks/export_feynman_data.R

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
source(file.path(script_dir, "feynman_datasets.R"))

out_dir <- file.path(script_dir, "data")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

DATA_SEED_TRAIN <- 42L   # identical to feynman_gate / feynman_full runners
DATA_SEED_TEST  <- 43L   # held-out test split

n_train <- 1000L
n_test  <- 500L

# Derive the Roman-numeral prefix from the equation ID (e.g. "I.10.7" -> "I",
# "II.11.3" -> "II", "III.4.33" -> "III").
id_prefix <- function(id) {
  sub("\\..*", "", id)
}

write_split <- function(key, n, seed, suffix) {
  ds   <- feynman_dataset(key, n = n, data_seed = seed, split = suffix)
  prob <- feynman_problems[[key]]
  df   <- as.data.frame(ds$X)
  names(df) <- paste0("x", seq_len(ncol(ds$X)))
  df$y <- ds$y
  prefix <- id_prefix(prob$id)
  fname  <- paste0("feynman_", prefix, "_", key, "_", suffix, ".csv")
  path   <- file.path(out_dir, fname)
  write.csv(df, path, row.names = FALSE)
  cat(sprintf("wrote %s  (n=%d, p=%d, id=%s, formula=%s)\n",
              fname, nrow(df), ncol(ds$X), prob$id,
              substr(prob$formula, 1L, 50L)))
  invisible(path)
}

for (key in feynman_dev_keys) {
  write_split(key, n_train, DATA_SEED_TRAIN, "train")
  write_split(key, n_test,  DATA_SEED_TEST,  "test")
}

cat(sprintf("\nDONE: exported %d equation × 2 splits = %d CSV files to %s\n",
            length(feynman_dev_keys),
            2L * length(feynman_dev_keys),
            out_dir))
