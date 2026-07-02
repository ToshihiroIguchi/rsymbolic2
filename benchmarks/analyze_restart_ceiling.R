# Multi-restart recovery-ceiling analysis (docs/43, Phase 0a/0b).
#
# Purpose: decide, from existing per-seed recovery data and WITHOUT building any
# feature, how much run-level multi-restart could raise Feynman recovery, and on
# which problems. A K-restart bundle "recovers" a problem iff at least one of its K
# independent runs recovers; so the ceiling is a pure post-hoc computation over the
# observed per-seed `recovered` column produced by 02_feynman_gate.R.
#
# Key idea (docs/38): multi-restart can only help a problem whose single-run recovery
# rate p is in (0, 1). A p = 0 problem (every seed fails, e.g. `interference` @2800)
# stays 0 for any K — restart is inert there; it needs a budget/structure lever, not
# more restarts. A p = 1 problem is already solved. The p in (0, 1) set is the target.
#
# Estimator: multi-restart bundles K INDEPENDENT fresh runs, so the right model for
# "K restarts recover" is the binomial complement over independent Bernoulli(p) draws
#     union_K = 1 - (1 - p_hat)^K
# where p_hat = r / S is the single-run recovery rate. (A hypergeometric union over
# the finite pool of S already-computed seeds is a different question — it saturates
# to 1 at K = S whenever r >= 1 — and would overstate what fresh restarts buy; we do
# not use it.) p_hat from few seeds is noisy (0a runs on S = 5), which is exactly why
# 0b re-estimates the responsive problems at S = 20 before any go/no-go.
#
# Usage (from project root):
#   Rscript benchmarks/analyze_restart_ceiling.R                       # latest full-seed gate CSV
#   Rscript benchmarks/analyze_restart_ceiling.R csv=results/feynman_gate_diag_20260701.csv
#   Rscript benchmarks/analyze_restart_ceiling.R csv=<path> kmax=20
#
# Recognised key=value args (all optional):
#   csv    path to a per-seed recovery CSV (default: most recent multi-seed
#          feynman_gate_*.csv under benchmarks/results/)
#   kmax   largest K to tabulate in the union-of-K curve (default 10)

# ---- Setup ------------------------------------------------------------------

# Resolve the script's own directory robustly under Rscript (sys.frame$ofile is
# unavailable there): parse --file= from the raw command line, else fall back.
resolve_script_dir <- function() {
  a <- commandArgs(FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f) == 1L && nzchar(f)) return(dirname(normalizePath(f)))
  d <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) NULL)
  if (!is.null(d) && length(d) == 1L && nzchar(d)) return(d)
  "benchmarks"
}
script_dir <- resolve_script_dir()

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- regmatches(args, regexpr(paste0("^", name, "=.*"), args))
  hit <- hit[nzchar(hit)]
  if (length(hit) == 0L) return(default)
  sub(paste0("^", name, "="), "", hit[[1L]])
}

results_dir <- file.path(script_dir, "results")

# Default CSV: the most recent feynman_gate CSV that has more than one seed
# (a runs=1 sanity pass has a single seed per problem and cannot estimate p).
pick_default_csv <- function() {
  cands <- list.files(results_dir, pattern = "^feynman_gate.*\\.csv$",
                      full.names = TRUE)
  if (length(cands) == 0L)
    stop("No feynman_gate_*.csv found under ", results_dir,
         "; pass csv=<path> explicitly.")
  # newest first by mtime; take the first with >1 distinct seed
  cands <- cands[order(file.info(cands)$mtime, decreasing = TRUE)]
  for (p in cands) {
    d <- tryCatch(read.csv(p, stringsAsFactors = FALSE), error = function(e) NULL)
    if (!is.null(d) && "seed" %in% names(d) &&
        length(unique(d$seed)) > 1L) return(p)
  }
  stop("No multi-seed feynman_gate_*.csv found under ", results_dir,
       "; pass csv=<path> explicitly.")
}

csv_arg <- get_arg("csv", "")
csv_path <- if (nzchar(csv_arg)) {
  # allow a path relative to the project root or to benchmarks/results
  if (file.exists(csv_arg)) csv_arg
  else if (file.exists(file.path(results_dir, basename(csv_arg))))
    file.path(results_dir, basename(csv_arg))
  else stop("csv not found: ", csv_arg)
} else pick_default_csv()

KMAX <- as.integer(get_arg("kmax", "10"))

# ---- Load and validate ------------------------------------------------------

df <- read.csv(csv_path, stringsAsFactors = FALSE)
need <- c("key", "seed", "recovered")
miss <- setdiff(need, names(df))
if (length(miss) > 0L)
  stop("CSV ", csv_path, " is missing column(s): ", paste(miss, collapse = ", "))

# `recovered` may load as logical or character ("TRUE"/"FALSE"); normalise.
df$recovered <- as.logical(df$recovered)

# Drop duplicate (key, seed) rows if any, keeping the first (defensive; a clean
# gate CSV has one row per key/seed).
df <- df[!duplicated(df[c("key", "seed")]), ]

cat("Multi-restart recovery-ceiling analysis (docs/43)\n")
cat("Source CSV:", csv_path, "\n")
cat(sprintf("Problems: %d   seeds/problem: %s   Kmax: %d\n",
            length(unique(df$key)),
            paste(sort(unique(tapply(df$seed, df$key,
                                     function(s) length(unique(s))))),
                  collapse = "/"),
            KMAX))
cat(rep("-", 78), "\n", sep = "")

# ---- Per-problem ceiling ----------------------------------------------------

# Recovery probability of a K-restart bundle under the independent-restart model.
union_binom <- function(p, K) 1 - (1 - p)^K

keys <- unique(df$key)
rows <- vector("list", length(keys))

for (i in seq_along(keys)) {
  key <- keys[[i]]
  sub <- df[df$key == key, ]
  S   <- nrow(sub)
  r   <- sum(sub$recovered, na.rm = TRUE)
  p   <- r / S

  class <- if (r == 0L) "p=0 (restart-INERT)"
           else if (r == S) "p=1 (solved)"
           else "p in (0,1) (RESPONSIVE)"

  # union-of-K curve under the independent-restart (binomial) model.
  uk <- vapply(seq_len(KMAX), function(K) union_binom(p, K), numeric(1L))

  rows[[i]] <- data.frame(
    key = key, n_seeds = S, n_recovered = r, p_hat = round(p, 3L),
    class = class,
    setNames(as.list(round(uk, 3L)), paste0("K", seq_len(KMAX))),
    stringsAsFactors = FALSE, check.names = FALSE
  )
}

tab <- do.call(rbind, rows)
# Order: responsive first (most actionable), then p=0, then solved; by p within.
ord <- order(match(sub("^p=0.*", "2",
                sub("^p in.*", "1",
                sub("^p=1.*", "3", tab$class))), c("1", "2", "3")),
             -tab$p_hat, tab$key)
tab <- tab[ord, ]

# ---- Report -----------------------------------------------------------------

# Compact console table: identity + p + a few union-of-K checkpoints.
show_ks <- intersect(c(1L, 2L, 3L, 5L, 10L), seq_len(KMAX))
kcols   <- paste0("K", show_ks)
cat(sprintf("%-16s %5s %5s %6s   %-24s %s\n",
            "key", "seed", "rec", "p_hat", "class",
            paste(sprintf("%6s", kcols), collapse = "")))
for (j in seq_len(nrow(tab))) {
  cat(sprintf("%-16s %5d %5d %6.2f   %-24s %s\n",
              tab$key[j], tab$n_seeds[j], tab$n_recovered[j], tab$p_hat[j],
              tab$class[j],
              paste(sprintf("%6.2f", as.numeric(tab[j, kcols])), collapse = "")))
}
cat(rep("-", 78), "\n", sep = "")

n_resp   <- sum(grepl("RESPONSIVE", tab$class))
n_inert  <- sum(grepl("INERT", tab$class))
n_solved <- sum(grepl("solved", tab$class))
cat(sprintf("Responsive p in (0,1): %d   |   Inert p=0: %d   |   Solved p=1: %d\n",
            n_resp, n_inert, n_solved))
if (n_resp > 0L) {
  resp_keys <- tab$key[grepl("RESPONSIVE", tab$class)]
  cat("Restart-responsive keys (feed to 0b many-seed / equal-compute test):\n  ",
      paste(resp_keys, collapse = ", "), "\n", sep = "")
} else {
  cat("No p in (0,1) problems in this CSV: multi-restart has no ceiling to exploit\n",
      "here at this budget. Failures are p=0 (need a budget/structure lever), not\n",
      "restart-responsive. See docs/43.\n", sep = "")
}
cat("Model: union_K = 1 - (1 - p_hat)^K (independent restarts). p_hat over ",
    unique(tab$n_seeds)[1L], " seeds is noisy;\n",
    "0b re-estimates the responsive keys at more seeds before any go/no-go.\n", sep = "")

# ---- Save -------------------------------------------------------------------

stamp    <- format(Sys.Date(), "%Y%m%d")
out_path <- file.path(dirname(csv_path),
                      paste0("restart_ceiling_", stamp, ".csv"))
write.csv(tab, out_path, row.names = FALSE)
message("Ceiling table written to: ", out_path)

invisible(tab)
