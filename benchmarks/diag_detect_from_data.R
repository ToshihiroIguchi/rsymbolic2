# Data-only pairwise-merge detector (Phase 0 follow-up to diag_separability.R).
#
# Purpose: docs/44 marked separability-decomposition GO using ORACLE access to
# the true formula `fn`, evaluated at freshly constructed off-sample points.
# A shippable feature only ever has sampled (X, y) -- no fn, no off-sample
# queries. This script asks: can the SAME variable merges (f depends on
# (xi, xj) only through xi*xj / xi/xj / xi+xj / xi-xj) be detected from
# (X, y) alone, reliably enough to fire on true merges without false-firing
# elsewhere?
#
# HARD CONSTRAINT: this script must never call prob$fn / prob$formula, and
# must never evaluate the true expression. It only ever touches
# feynman_dataset(key, ...)$X, $y, and $n_vars. (Grep-verifiable: the string
# "fn(" and "$formula" do not appear anywhere below outside this comment
# block; feynman_problems is referenced only to read n_vars/domains for
# looping over keys and column labels -- see build_default_keys() and the
# "vars" label lookup, which reads names(prob$domains), never prob$fn.)
#
# Primary method -- split-fit comparison (self-normalizing, sample points
# only, no off-sample oracle queries):
#   1. Fit a dependency-free k-NN surrogate (z-score standardized columns,
#      Euclidean distance, k=10, base R only) on a 70/30 train/test split.
#   2. nmse_full = test NMSE of the surrogate fit on all original columns.
#   3. For each pair (i<j) and merge mode {prod, ratio, sum, diff}: replace
#      columns i,j with the single merged column, refit the surrogate on the
#      SAME split -> nmse_merged.
#   4. DETECTED iff the surrogate can actually fit the data at all
#      (nmse_full < fit_floor; else UNTESTABLE) AND merging does not hurt
#      accuracy (nmse_merged <= nmse_full * (1 + slack)).
#
# Secondary method -- surrogate-query invariance (exploratory, kept
# secondary): fit one k-NN surrogate M on ALL data (domain box taken from
# column min/max of the sampled X -- data-derivable, no oracle domains
# used), then port diag_separability.R's merge_test transform logic but
# query M instead of fn, and compare the invariance-transform median rel-err
# against a CONTROL transform that deliberately breaks the merged quantity
# (for prod/ratio the two share a transform family: prod's invariant
# transform is ratio's control and vice versa; likewise sum/diff).
#
# Usage (PowerShell + full-path Rscript):
#   Rscript benchmarks/diag_detect_from_data.R
#   Rscript benchmarks/diag_detect_from_data.R keys=bose_einstein,newtons_grav seeds=42,43
#   Rscript benchmarks/diag_detect_from_data.R noise=1e-3
#
# Recognised key=value args (all optional):
#   keys        comma-separated problem keys (default: all keys with n_vars>=2)
#   n           points per dataset (default 800)
#   seeds       comma-separated data seeds (default 42,43,44)
#   noise       relative Gaussian noise added to y: sd = noise * sd(y) (default 0)
#   k           k-NN neighbors (default 10)
#   slack       merge-detection tolerance on nmse_merged/nmse_full (default 0.25)
#   fit_floor   max nmse_full for the surrogate to be considered "fit" (default 0.05)
#   secondary   run the secondary invariance method too (default true)
#   sec_trials  trials per secondary test (default 40)

# ---- Setup ------------------------------------------------------------------

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) "benchmarks"
)
source(file.path(script_dir, "feynman_datasets.R"))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- regmatches(args, regexpr(paste0("^", name, "=.*"), args))
  hit <- hit[nzchar(hit)]
  if (length(hit) == 0L) return(default)
  sub(paste0("^", name, "="), "", hit[[1L]])
}

# Default key set: every problem with >=2 variables (pairwise merges are
# undefined for 1-variable problems). Reads only n_vars, never fn/formula.
build_default_keys <- function() {
  nv <- vapply(feynman_problems, function(p) p$n_vars, integer(1L))
  names(feynman_problems)[nv >= 2L]
}

raw_keys <- get_arg("keys", "")
KEYS <- if (nzchar(raw_keys)) strsplit(raw_keys, ",")[[1L]] else build_default_keys()

N          <- as.integer(get_arg("n", "800"))
SEEDS      <- as.integer(strsplit(get_arg("seeds", "42,43,44"), ",")[[1L]])
NOISE      <- as.numeric(get_arg("noise", "0"))
K          <- as.integer(get_arg("k", "10"))
SLACK      <- as.numeric(get_arg("slack", "0.25"))
FIT_FLOOR  <- as.numeric(get_arg("fit_floor", "0.05"))
SECONDARY  <- tolower(get_arg("secondary", "true")) %in% c("true", "1", "yes")
SEC_TRIALS <- as.integer(get_arg("sec_trials", "40"))
SPLIT_FRAC <- 0.7
SPLIT_SEED <- 123L
MODES      <- c("prod", "ratio", "sum", "diff")

# ---- k-NN surrogate (base R only; z-score standardized, Euclidean) ---------

standardize <- function(Xtr, Xte) {
  mu  <- colMeans(Xtr)
  sdv <- apply(Xtr, 2L, sd)
  sdv[!is.finite(sdv) | sdv < 1e-12] <- 1e-12
  list(Ztr = scale(Xtr, center = mu, scale = sdv),
       Zte = scale(Xte, center = mu, scale = sdv),
       mu = mu, sdv = sdv)
}

# Vectorized squared-Euclidean k-NN prediction: D2[i,j] via the
# ||a||^2 + ||b||^2 - 2 a.b identity (no O(n^2) explicit loop).
knn_predict <- function(Ztr, ytr, Zte, k) {
  if (any(!is.finite(Ztr)) || any(!is.finite(Zte)))
    return(rep(NA_real_, nrow(Zte)))
  sumA2 <- rowSums(Ztr^2)
  sumB2 <- rowSums(Zte^2)
  D2 <- outer(sumA2, sumB2, "+") - 2 * (Ztr %*% t(Zte))
  D2[D2 < 0] <- 0
  kk <- min(k, nrow(Ztr))
  vapply(seq_len(ncol(D2)), function(col) {
    idx <- order(D2[, col])[seq_len(kk)]
    mean(ytr[idx])
  }, numeric(1L))
}

nmse_of <- function(yhat, ytest) {
  v <- var(ytest)
  if (!is.finite(v) || v < 1e-300) return(NA_real_)
  mean((yhat - ytest)^2) / v
}

surrogate_nmse <- function(X, y, train_idx, test_idx, k) {
  Xtr <- X[train_idx, , drop = FALSE]; ytr <- y[train_idx]
  Xte <- X[test_idx, , drop = FALSE];  yte <- y[test_idx]
  st <- standardize(Xtr, Xte)
  yhat <- knn_predict(st$Ztr, ytr, st$Zte, k)
  nmse_of(yhat, yte)
}

make_split <- function(n, frac, seed) {
  set.seed(seed)
  perm <- sample.int(n)
  ntr  <- floor(n * frac)
  list(train = perm[seq_len(ntr)], test = perm[(ntr + 1L):n])
}

build_merged_X <- function(X, i, j, mode) {
  merged <- switch(mode,
    prod  = X[, i] * X[, j],
    ratio = X[, i] / X[, j],
    sum   = X[, i] + X[, j],
    diff  = X[, i] - X[, j],
    stop("unknown mode: ", mode))
  keep <- setdiff(seq_len(ncol(X)), c(i, j))
  cbind(merged, X[, keep, drop = FALSE])
}

# ---- Secondary method: surrogate-query invariance --------------------------
#
# Ports diag_separability.R's merge_test constructed-point transforms, but
# queries the k-NN surrogate M (built from ALL sampled data) instead of the
# true fn, and judges PASS by comparing against a CONTROL transform from the
# paired mode that deliberately breaks the merged quantity (prod<->ratio and
# sum<->diff share a transform family). Domain box = column min/max of the
# sampled X (data-derivable, not the oracle's analytic domain).
sec_invariance_test <- function(query_M, lo, hi, i, j, mode, trials) {
  errs_inv <- numeric(0); errs_ctrl <- numeric(0)
  attempts <- 0L
  max_attempts <- trials * 6L
  while (length(errs_inv) < trials && attempts < max_attempts) {
    attempts <- attempts + 1L
    x <- vapply(seq_along(lo), function(v) runif(1L, lo[v], hi[v]), numeric(1L))
    ok <- TRUE
    if (mode == "prod") {
      cmin <- max(lo[i] / x[i], x[j] / hi[j]); cmax <- min(hi[i] / x[i], x[j] / lo[j])
      if (!is.finite(cmin) || !is.finite(cmax) || cmin >= cmax) ok <- FALSE
      if (ok) {
        cc <- exp(runif(1L, log(cmin), log(cmax)))
        if (abs(log(cc)) < 0.05) ok <- FALSE
      }
      if (ok) {
        xt <- x; xt[i] <- x[i] * cc; xt[j] <- x[j] / cc   # invariant: keeps xi*xj
        xc <- x; xc[i] <- x[i] * cc; xc[j] <- x[j] * cc   # control: keeps xi/xj, breaks xi*xj
      }
    } else if (mode == "ratio") {
      cmin <- max(lo[i] / x[i], lo[j] / x[j]); cmax <- min(hi[i] / x[i], hi[j] / x[j])
      if (!is.finite(cmin) || !is.finite(cmax) || cmin >= cmax) ok <- FALSE
      if (ok) {
        cc <- exp(runif(1L, log(cmin), log(cmax)))
        if (abs(log(cc)) < 0.05) ok <- FALSE
      }
      if (ok) {
        xt <- x; xt[i] <- x[i] * cc; xt[j] <- x[j] * cc   # invariant: keeps xi/xj
        xc <- x; xc[i] <- x[i] * cc; xc[j] <- x[j] / cc   # control: keeps xi*xj, breaks xi/xj
      }
    } else if (mode == "sum") {
      dmin <- max(lo[i] - x[i], x[j] - hi[j]); dmax <- min(hi[i] - x[i], x[j] - lo[j])
      if (!is.finite(dmin) || !is.finite(dmax) || dmin >= dmax) ok <- FALSE
      if (ok) {
        d <- runif(1L, dmin, dmax)
        if (abs(d) < 0.05) ok <- FALSE
      }
      if (ok) {
        xt <- x; xt[i] <- x[i] + d; xt[j] <- x[j] - d     # invariant: keeps xi+xj
        xc <- x; xc[i] <- x[i] + d; xc[j] <- x[j] + d     # control: keeps xi-xj, breaks xi+xj
      }
    } else { # diff
      dmin <- max(lo[i] - x[i], lo[j] - x[j]); dmax <- min(hi[i] - x[i], hi[j] - x[j])
      if (!is.finite(dmin) || !is.finite(dmax) || dmin >= dmax) ok <- FALSE
      if (ok) {
        d <- runif(1L, dmin, dmax)
        if (abs(d) < 0.05) ok <- FALSE
      }
      if (ok) {
        xt <- x; xt[i] <- x[i] + d; xt[j] <- x[j] + d     # invariant: keeps xi-xj
        xc <- x; xc[i] <- x[i] + d; xc[j] <- x[j] - d     # control: keeps xi+xj, breaks xi-xj
      }
    }
    if (!ok) next
    m0 <- query_M(x); mt <- query_M(xt); mc <- query_M(xc)
    e_inv  <- abs(mt - m0) / (abs(m0) + abs(mt) + 1e-300)
    e_ctrl <- abs(mc - m0) / (abs(m0) + abs(mc) + 1e-300)
    if (!is.finite(e_inv) || !is.finite(e_ctrl)) next
    errs_inv  <- c(errs_inv, e_inv)
    errs_ctrl <- c(errs_ctrl, e_ctrl)
  }
  if (length(errs_inv) < trials %/% 2L)
    return(list(err_inv = NA_real_, err_ctrl = NA_real_, n = length(errs_inv), pass = FALSE))
  mi <- median(errs_inv); mc2 <- median(errs_ctrl)
  # PASS iff the invariance error is much smaller than the control error
  # (invariance error near M's own baseline noise floor is implied by being
  # a small fraction of the control's genuine-dependence error).
  pass <- is.finite(mi) && is.finite(mc2) && mc2 > 1e-12 && (mi < 0.3 * mc2)
  list(err_inv = mi, err_ctrl = mc2, n = length(errs_inv), pass = pass)
}

# ---- Run --------------------------------------------------------------------

cat("Data-only pairwise-merge detector (Phase 0 follow-up)\n")
cat(sprintf("keys=%s\n", paste(KEYS, collapse = ",")))
cat(sprintf("n=%d  seeds=%s  noise=%.1e  k=%d  slack=%.2f  fit_floor=%.3f  secondary=%s\n",
            N, paste(SEEDS, collapse = ","), NOISE, K, SLACK, FIT_FLOOR, SECONDARY))

rows     <- list()
rows_sec <- list()

for (key in KEYS) {
  prob <- feynman_problems[[key]]
  if (is.null(prob)) { warning("unknown key: ", key); next }
  p <- prob$n_vars
  if (p < 2L) { cat(sprintf("%-16s skipped (n_vars=%d < 2)\n", key, p)); next }
  vars <- names(prob$domains)   # labels only, not the formula

  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("%s  (%d vars)\n", key, p))

  for (seed in SEEDS) {
    ds <- feynman_dataset(key, n = N, data_seed = seed)
    X <- ds$X; y <- ds$y

    if (NOISE > 0) {
      set.seed(seed * 1000L + 7L)
      y <- y + rnorm(length(y), mean = 0, sd = NOISE * sd(y))
    }

    split <- make_split(nrow(X), SPLIT_FRAC, SPLIT_SEED)
    nmse_full <- surrogate_nmse(X, y, split$train, split$test, K)
    untestable_full <- !is.finite(nmse_full) || nmse_full >= FIT_FLOOR

    cat(sprintf("  seed=%-4d nmse_full=%.4f%s\n", seed, nmse_full,
                if (untestable_full) "  [UNTESTABLE: surrogate can't fit]" else ""))

    for (i in seq_len(p - 1L)) for (j in seq((i + 1L), p)) {
      for (mode in MODES) {
        Xm <- tryCatch(build_merged_X(X, i, j, mode), error = function(e) NULL)
        if (is.null(Xm) || any(!is.finite(Xm))) {
          nmse_merged <- NA_real_
        } else {
          nmse_merged <- surrogate_nmse(Xm, y, split$train, split$test, K)
        }
        ratio_val <- if (is.finite(nmse_merged) && is.finite(nmse_full) && nmse_full > 0) {
          nmse_merged / nmse_full
        } else NA_real_
        detected <- !untestable_full && is.finite(nmse_merged) &&
          (nmse_merged <= nmse_full * (1 + SLACK))

        rows[[length(rows) + 1L]] <- data.frame(
          key = key, seed = seed, noise = NOISE, test = paste0("merge_", mode),
          vars = paste(vars[i], vars[j], sep = ":"),
          nmse_full = nmse_full, nmse_merged = nmse_merged, ratio = ratio_val,
          detected = detected, untestable = untestable_full,
          stringsAsFactors = FALSE)

        if (detected)
          cat(sprintf("    DETECTED merge_%-5s %-14s nmse_merged=%.4f ratio=%.3f\n",
                      mode, paste(vars[i], vars[j], sep = ":"), nmse_merged, ratio_val))
      }
    }

    if (SECONDARY && !untestable_full) {
      st_all <- standardize(X, X)
      Zall <- st_all$Ztr; mu <- st_all$mu; sdv <- st_all$sdv
      lo <- apply(X, 2L, min); hi <- apply(X, 2L, max)
      kk <- min(K, nrow(Zall))
      query_M <- function(xrow) {
        z <- (xrow - mu) / sdv
        d2 <- rowSums((Zall - matrix(z, nrow(Zall), length(z), byrow = TRUE))^2)
        idx <- order(d2)[seq_len(kk)]
        mean(y[idx])
      }
      for (i in seq_len(p - 1L)) for (j in seq((i + 1L), p)) {
        for (mode in MODES) {
          res <- sec_invariance_test(query_M, lo, hi, i, j, mode, SEC_TRIALS)
          rows_sec[[length(rows_sec) + 1L]] <- data.frame(
            key = key, seed = seed, noise = NOISE, test = paste0("merge_", mode),
            vars = paste(vars[i], vars[j], sep = ":"),
            err_inv = res$err_inv, err_ctrl = res$err_ctrl, n = res$n,
            pass = res$pass, stringsAsFactors = FALSE)
        }
      }
    }
  }
}

df <- do.call(rbind, rows)
out_dir <- file.path(script_dir, "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
out_path <- file.path(out_dir, "detect_from_data.csv")
write.csv(df, out_path, row.names = FALSE)
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("primary results (%d rows) written to %s\n", nrow(df), out_path))
cat(sprintf("  detected=%d  untestable=%d  total=%d\n",
            sum(df$detected), sum(df$untestable & !duplicated(df[, c("key","seed")])), nrow(df)))

if (SECONDARY && length(rows_sec) > 0L) {
  df_sec <- do.call(rbind, rows_sec)
  sec_path <- file.path(out_dir, "detect_from_data_secondary.csv")
  write.csv(df_sec, sec_path, row.names = FALSE)
  cat(sprintf("secondary results (%d rows) written to %s\n", nrow(df_sec), sec_path))
  cat(sprintf("  pass=%d  total=%d\n", sum(df_sec$pass), nrow(df_sec)))
}

invisible(df)
