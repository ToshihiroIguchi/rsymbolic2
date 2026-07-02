# Separability / symmetry oracle for the p = 0 Feynman problems (Phase 0-D).
#
# Purpose: decide, WITHOUT running any search, whether an AI-Feynman-style
# decomposition layer could crack the structure-inert problems (planck,
# bose_einstein, interference; docs/43 "INERT p=0"). Two questions:
#   1. Are the reductions numerically DETECTABLE on the benchmark data domain?
#      (single-variable additive/multiplicative separability; pairwise variable
#      merges: f depends on (xi, xj) only through xi*xj, xi/xj, xi+xj, xi-xj)
#   2. Do the detected reductions leave a residual subproblem that is in the
#      solved class? For planck / bose_einstein the oracle chain ends at the
#      one-variable residual q(r) = r / (exp(r) - 1); this script exports that
#      residual dataset so a later (cheap) search run can confirm solvability.
#
# This is analysis only: the true formula (feynman_datasets.R $fn) is evaluated
# directly. No symbolic_regression() call is made here.
#
# Usage (PowerShell + full-path Rscript):
#   Rscript benchmarks/diag_separability.R
#   Rscript benchmarks/diag_separability.R keys=planck,bose_einstein trials=500
#
# Recognised key=value args (all optional):
#   keys    comma-separated problem keys
#           (default planck,bose_einstein,interference,newtons_grav;
#            newtons_grav is a positive control: its product merges must PASS)
#   trials  random trials per test (default 200)
#   tol     max relative error to call a test PASS (default 1e-6)

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

KEYS   <- strsplit(get_arg("keys",
            "planck,bose_einstein,interference,newtons_grav"), ",")[[1L]]
TRIALS <- as.integer(get_arg("trials", "200"))
TOL    <- as.numeric(get_arg("tol", "1e-6"))
SEED   <- 42L

# ---- Test machinery ---------------------------------------------------------

# Draw one uniform point inside the problem's domains.
draw_point <- function(domains) {
  vapply(domains, function(d) runif(1L, d[1L], d[2L]), numeric(1L))
}

rel_err <- function(a, b) abs(a - b) / (abs(a) + abs(b) + 1e-300)

# Single-variable separability, f(x) = g(xi) (*|+) h(x_-i), via the exchange
# test on two random points x, xp:
#   multiplicative: f(xi,r) * f(xi',r') == f(xi,r') * f(xi',r)
#   additive:       f(xi,r) + f(xi',r') == f(xi,r') + f(xi',r)
sep_test <- function(fn, domains, i, mode, trials, tol) {
  worst <- 0
  for (t in seq_len(trials)) {
    x  <- draw_point(domains)
    xp <- draw_point(domains)
    x_mix1 <- x;  x_mix1[i] <- xp[i]   # (xi', r)
    x_mix2 <- xp; x_mix2[i] <- x[i]    # (xi , r')
    A <- fn(x); B <- fn(xp); C <- fn(x_mix2); D <- fn(x_mix1)
    e <- if (mode == "mul") rel_err(A * B, C * D) else rel_err(A + B, C + D)
    if (!is.finite(e)) return(list(err = Inf, pass = FALSE))
    worst <- max(worst, e)
  }
  list(err = worst, pass = worst < tol)
}

# Pairwise merge invariance: f unchanged under a transform of (xi, xj) that
# preserves the merged quantity. The shift/scale is sampled so the transformed
# point stays inside the domain box (the fn is only trusted there).
merge_test <- function(fn, domains, i, j, mode, trials, tol) {
  lo <- vapply(domains, `[`, numeric(1L), 1L)
  hi <- vapply(domains, `[`, numeric(1L), 2L)
  worst <- 0; used <- 0L
  for (t in seq_len(trials * 3L)) {       # oversample; some draws degenerate
    if (used >= trials) break
    x <- draw_point(domains)
    if (mode == "prod") {                 # (c*xi, xj/c) keeps xi*xj
      cmin <- max(lo[i] / x[i], x[j] / hi[j])
      cmax <- min(hi[i] / x[i], x[j] / lo[j])
      if (cmin >= cmax) next
      cc <- exp(runif(1L, log(cmin), log(cmax)))
      if (abs(log(cc)) < 0.05) next
      xt <- x; xt[i] <- x[i] * cc; xt[j] <- x[j] / cc
    } else if (mode == "ratio") {         # (c*xi, c*xj) keeps xi/xj
      cmin <- max(lo[i] / x[i], lo[j] / x[j])
      cmax <- min(hi[i] / x[i], hi[j] / x[j])
      if (cmin >= cmax) next
      cc <- exp(runif(1L, log(cmin), log(cmax)))
      if (abs(log(cc)) < 0.05) next
      xt <- x; xt[i] <- x[i] * cc; xt[j] <- x[j] * cc
    } else if (mode == "sum") {           # (xi+d, xj-d) keeps xi+xj
      dmin <- max(lo[i] - x[i], x[j] - hi[j])
      dmax <- min(hi[i] - x[i], x[j] - lo[j])
      if (dmin >= dmax) next
      d <- runif(1L, dmin, dmax)
      if (abs(d) < 0.05) next
      xt <- x; xt[i] <- x[i] + d; xt[j] <- x[j] - d
    } else {                              # "diff": (xi+d, xj+d) keeps xi-xj
      dmin <- max(lo[i] - x[i], lo[j] - x[j])
      dmax <- min(hi[i] - x[i], hi[j] - x[j])
      if (dmin >= dmax) next
      d <- runif(1L, dmin, dmax)
      if (abs(d) < 0.05) next
      xt <- x; xt[i] <- x[i] + d; xt[j] <- x[j] + d
    }
    e <- rel_err(fn(x), fn(xt))
    if (!is.finite(e)) return(list(err = Inf, pass = FALSE, n = used))
    worst <- max(worst, e)
    used <- used + 1L
  }
  list(err = worst, pass = (used >= trials %/% 2L) && worst < tol, n = used)
}

# ---- Run --------------------------------------------------------------------

set.seed(SEED)
cat("Separability / symmetry oracle (Phase 0-D)\n")
cat(sprintf("trials=%d  tol=%.0e  keys=%s\n", TRIALS, TOL,
            paste(KEYS, collapse = ",")))

rows <- list()
for (key in KEYS) {
  prob <- feynman_problems[[key]]
  if (is.null(prob)) { warning("unknown key: ", key); next }
  fn   <- prob$fn
  doms <- prob$domains
  vars <- names(doms)
  n    <- prob$n_vars
  cat(rep("=", 70), "\n", sep = "")
  cat(sprintf("%s  (%s)   f = %s\n", key, prob$id, prob$formula))

  # single-variable separability
  for (i in seq_len(n)) {
    for (mode in c("mul", "add")) {
      r <- sep_test(fn, doms, i, mode, TRIALS, TOL)
      rows[[length(rows) + 1L]] <- data.frame(
        key = key, test = paste0("sep_", mode), vars = vars[i],
        max_rel_err = r$err, pass = r$pass, stringsAsFactors = FALSE)
      if (r$pass)
        cat(sprintf("  PASS  %-9s %-14s (max err %.1e)\n",
                    paste0("sep_", mode), vars[i], r$err))
    }
  }
  # pairwise merges
  for (i in seq_len(n - 1L)) for (j in seq((i + 1L), n)) {
    for (mode in c("prod", "ratio", "sum", "diff")) {
      r <- merge_test(fn, doms, i, j, mode, TRIALS, TOL)
      rows[[length(rows) + 1L]] <- data.frame(
        key = key, test = paste0("merge_", mode),
        vars = paste(vars[i], vars[j], sep = ":"),
        max_rel_err = r$err, pass = r$pass, stringsAsFactors = FALSE)
      if (r$pass)
        cat(sprintf("  PASS  %-9s %-14s (max err %.1e, n=%d)\n",
                    paste0("merge_", mode),
                    paste(vars[i], vars[j], sep = ":"), r$err, r$n))
    }
  }
}

df <- do.call(rbind, rows)
out_dir <- file.path(script_dir, "results")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
oracle_path <- file.path(out_dir, "separability_oracle.csv")
write.csv(df, oracle_path, row.names = FALSE)
cat(rep("=", 70), "\n", sep = "")
cat(sprintf("full test matrix (%d rows, incl. FAILs) written to %s\n",
            nrow(df), oracle_path))

# ---- Residual export --------------------------------------------------------
#
# Two tiers per problem, both built from the SAME 1000-row train data as the
# gate (data_seed 42), so later parity-config search runs can test whether the
# residual is in the solved class.
#
# DETECTABLE tier — only reductions the tests above can find:
#   bose_einstein: merge hbar,omega -> s ; kB,T -> u
#                  residual f(s, u) = s / (exp(s/u) - 1)            [2 vars]
#   planck:        c is sep_mul (fit 1/c^2 from data) ; merge k,T -> u
#                  residual g(hbar, omega, u) = hbar*omega^3 /
#                           (pi^2 * (exp(hbar*omega/u) - 1))        [3 vars]
#                  (hbar:omega does NOT prod-merge: the stray omega^2 factor
#                  breaks the invariance — verified FAIL above.)
#
# ORACLE tier — hand-derived upper bound, needs a homogeneity test the feature
# does not (yet) have: both problems end at the SAME 1-var residual
#   q(r) = r / (exp(r) - 1),  r = s/u in ~[1/9, 9]
#   bose_einstein: f = u * q(s/u)
#   planck:        f = (omega^2 / (pi^2 c^2)) * u * q(s/u)
# If even q(r) is not recoverable at defaults, decomposition is a definite
# NO-GO; if q passes but the detectable-tier residuals fail, the feature would
# additionally need subset-homogeneity detection to reach it.

chain_pass <- function(k, tests) {
  sub <- df[df$key == k & df$pass, ]
  all(vapply(tests, function(tv)
    any(sub$test == tv[[1L]] & sub$vars == tv[[2L]]), logical(1L)))
}

export_residual <- function(name, X, y) {
  path <- file.path(script_dir, "data",
                    sprintf("residual_%s_train.csv", name))
  df_out <- as.data.frame(X)
  names(df_out) <- paste0("x", seq_len(ncol(df_out)))
  df_out$y <- y
  write.csv(df_out, path, row.names = FALSE)
  cat(sprintf("residual dataset written to %s\n", path))
}

if ("bose_einstein" %in% KEYS &&
    chain_pass("bose_einstein", list(c("merge_prod", "hbar:omega"),
                                     c("merge_prod", "kB:T")))) {
  ds <- feynman_dataset("bose_einstein", n = 1000L, data_seed = 42L)
  s <- ds$X[, 1L] * ds$X[, 2L]; u <- ds$X[, 3L] * ds$X[, 4L]
  export_residual("bose2v_detectable", cbind(s, u), ds$y)
  export_residual("bose_q_oracle", cbind(s / u), ds$y / u)
}
if ("planck" %in% KEYS &&
    chain_pass("planck", list(c("merge_prod", "k:T"),
                              c("sep_mul", "c")))) {
  ds <- feynman_dataset("planck", n = 1000L, data_seed = 42L)
  s <- ds$X[, 1L] * ds$X[, 2L]; u <- ds$X[, 4L] * ds$X[, 5L]
  export_residual("planck3v_detectable",
                  cbind(ds$X[, 1L], ds$X[, 2L], u), ds$y * ds$X[, 3L]^2)
  export_residual("planck_q_oracle", cbind(s / u),
                  ds$y * pi^2 * ds$X[, 3L]^2 / (ds$X[, 2L]^2 * u))
}

invisible(df)
