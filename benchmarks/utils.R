# Shared utilities for rsymbolic2 benchmarks.

# Normalized Mean Squared Error.
#
# rsymbolic2 returns loss = SSR = sum((y_hat - y)^2).
# NMSE = SSR / sum((y - mean(y))^2) = loss / ((n-1) * var(y))
#
# Threshold for "recovered": NMSE < 1e-4 (standard in Nguyen/Feynman literature).
compute_nmse <- function(loss, y) {
  denom <- (length(y) - 1L) * var(y)
  if (!is.finite(denom) || denom <= 0) return(Inf)
  loss / denom
}

RECOVERY_THRESHOLD <- 1e-4

is_recovered <- function(nmse) is.finite(nmse) && nmse < RECOVERY_THRESHOLD

# Save a results data frame to a timestamped CSV under benchmarks/results/.
save_results <- function(df, label) {
  dir <- file.path(dirname(sys.frame(1)$ofile %||% "."), "results")
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  stamp <- format(Sys.Date(), "%Y%m%d")
  path <- file.path(dir, paste0(label, "_", stamp, ".csv"))
  write.csv(df, path, row.names = FALSE)
  message("Results written to: ", path)
  invisible(path)
}

# Print a summary table of benchmark results grouped by problem.
#
# @param df  Data frame with columns: problem_id, seed, elapsed_sec, loss,
#            nmse, recovered, expression.
print_summary <- function(df, title = "Benchmark Summary") {
  cat("\n", title, "\n", rep("=", nchar(title) + 2), "\n", sep = "")

  problems <- unique(df$problem_id)
  n_seeds  <- length(unique(df$seed))
  n_pass   <- 0L

  for (pid in problems) {
    sub  <- df[df$problem_id == pid, ]
    rec  <- sum(sub$recovered, na.rm = TRUE)
    prob_pass <- rec >= ceiling(n_seeds / 2)  # majority
    if (prob_pass) n_pass <- n_pass + 1L

    med_nmse <- median(sub$nmse[is.finite(sub$nmse)])
    med_time <- median(sub$elapsed_sec)
    formula  <- attr(df, "formulas")[[pid]] %||% ""

    cat(sprintf("  %-4s  %-28s  %d/%d recovered  med NMSE=%.1e  med time=%.0fs%s\n",
                pid,
                substr(formula, 1L, 28L),
                rec, n_seeds,
                med_nmse, med_time,
                if (prob_pass) "  PASS" else "  FAIL"))
  }

  n_active <- length(problems)
  gate_pass <- n_pass >= 7L
  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("Gate: %d/%d problems recovered (threshold: 7/%d)  -->  %s\n\n",
              n_pass, n_active, n_active, if (gate_pass) "PASS" else "FAIL"))
  invisible(gate_pass)
}

`%||%` <- function(x, y) if (is.null(x)) y else x
