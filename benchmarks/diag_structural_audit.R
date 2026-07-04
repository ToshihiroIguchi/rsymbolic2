# Structural-truth audit of "recovered" runs via extrapolation (Phase 0 audit).
#
# Motivation: the recovery criterion NMSE < 1e-4 (train domain) conflates
#   (a) TRUE structural recovery — the expression IS the formula; and
#   (b) "threshold grinders" — wrong structure (typically pow-soup with fitted
#       fractional exponents) polished to just under the threshold on the
#       training box (first seen for interference in docs/38).
# Discriminator: evaluate the recovered expression OUTSIDE the training box.
# The true structure extrapolates exactly (NMSE stays ~machine precision);
# a grinder is a local approximation and degrades by orders of magnitude.
#
# For every recovered=TRUE row of the listed result CSVs this script parses the
# expression (engine syntax: x0..x(n-1), square(), infix ^), evaluates it on
# fresh points drawn from an EXTENDED domain box (lower bound 0.75*lo, upper
# bound hi + 0.5*(hi - lo), with per-problem validity predicates so the true
# formula itself stays well-defined), and reports
#   nmse_extrap = SSE / ((n-1) * var(y_true))   on the extrapolation sample.
# Verdict bands: TRUE_STRUCTURE < 1e-6 <= UNCERTAIN <= 1e-3 < GRINDER.
# A candidate producing non-finite values on > 5% of valid points is GRINDER
# (its structure is not even defined off the training box).
#
# Usage:
#   Rscript benchmarks/diag_structural_audit.R
#   Rscript benchmarks/diag_structural_audit.R n=1000 seed=4242
#
# Single-CSV mode (docs/47 units screen): csv=<file under results/> audits every
# key present in that one gate CSV instead of the frozen docs/44 manifest.
#   Rscript benchmarks/diag_structural_audit.R csv=feynman_gate_units_20260704.csv \
#       gens=2800 out=_units_on
# `gens` only labels the output rows; `out` suffixes the two output CSV names so
# audits of different arms never overwrite each other (or the docs/44 outputs).

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
N_EXTRAP <- as.integer(get_arg("n", "500"))
SEED     <- as.integer(get_arg("seed", "4242"))
CSV_ONLY <- get_arg("csv", "")
CSV_GENS <- as.integer(get_arg("gens", "2800"))
OUT_SUF  <- get_arg("out", "")

# Validity predicates on the extended box: keep the true formula in the regime
# the training data guaranteed (same inequalities as the domain comments in
# feynman_datasets.R), so "extrapolation failure" can only come from the
# candidate's wrong structure, never from the truth becoming singular.
validity <- list(
  lorentz_x      = function(X) (X[, 1] - X[, 2] * X[, 3] >= 0.5) &
                               (X[, 2] / X[, 4] <= 0.9),
  clausius_moss  = function(X) (X[, 1] * X[, 2] <= 2.25),
  # Same regime guards for the keys first audited in the docs/47 all-key csv=
  # mode: keep v/c below 1 (sqrt(1-(v/c)^2) real) and the Clausius-Mossotti
  # denominator positive on the extended box.
  clausius_moss2 = function(X) (X[, 1] * X[, 2] <= 2.25),
  rel_mass       = function(X) (X[, 2] / X[, 3] <= 0.9),
  rel_mom        = function(X) (X[, 2] / X[, 3] <= 0.9),
  doppler_rel    = function(X) (X[, 2] / X[, 3] <= 0.9)
)

# Draw n valid points from the extended box of `key`.
draw_extrap <- function(key, n) {
  prob <- feynman_problems[[key]]
  lo <- vapply(prob$domains, `[`, numeric(1L), 1L)
  hi <- vapply(prob$domains, `[`, numeric(1L), 2L)
  elo <- 0.75 * lo
  ehi <- hi + 0.5 * (hi - lo)
  ok_fun <- validity[[key]]
  X <- matrix(NA_real_, 0L, length(lo))
  while (nrow(X) < n) {
    cand <- vapply(seq_along(lo),
                   function(j) runif(2L * n, elo[j], ehi[j]),
                   numeric(2L * n))
    # discard points inside the original training box: every coordinate
    # interior means no extrapolation is being tested there
    inside <- rowSums(sweep(cand, 2L, lo, ">=") &
                      sweep(cand, 2L, hi, "<=")) == ncol(cand)
    cand <- cand[!inside, , drop = FALSE]
    if (!is.null(ok_fun)) cand <- cand[ok_fun(cand), , drop = FALSE]
    X <- rbind(X, cand)
  }
  X[seq_len(n), , drop = FALSE]
}

# Evaluate an engine expression string on rows of X, replicating the ENGINE's
# operator semantics (dual.hpp / soa_eval.hpp), not R's:
#   sqrt(x) = sqrt(x) if x > 0 else 0                      [safe, neg -> 0]
#   x ^ y   = safe_pow: x>0 -> exp(y*log(x)); x==0,y>0 -> 0;
#             x<0, y within 1e-6 of an integer -> std::pow(x, round(y));
#             otherwise 0
#   log     = unprotected (NaN for x <= 0), like std::log
# Also, the engine's printer emits negative constant leaves WITHOUT parens
# (e.g. "(-1.3 ^ x1)"), which R would parse as -(1.3^x1) because unary minus
# binds looser than ^. Unary negative literals are therefore wrapped in parens
# before parsing (binary minus is always printed with spaces, "a - b", so it
# never has a digit glued to the '-').
engine_sqrt <- function(z) {
  out <- rep(NaN, length(z))
  ok  <- !is.na(z)
  out[ok] <- ifelse(z[ok] > 0, sqrt(pmax(z[ok], 0)), 0)
  out
}
engine_pow <- function(x, y) {
  n <- max(length(x), length(y))
  x <- rep_len(x, n); y <- rep_len(y, n)
  out <- rep(NaN, n)
  ok  <- !is.na(x) & !is.na(y)
  out[ok] <- 0
  pos <- ok & x > 0
  out[pos] <- exp(y[pos] * log(x[pos]))
  negint <- ok & x < 0 & abs(y - round(y)) < 1e-6
  out[negint] <- x[negint]^round(y[negint])
  out
}
eval_expr <- function(expr_str, X) {
  expr_str <- gsub("(?<![\\w.)])(-[0-9.]+(?:[eE][+-]?[0-9]+)?)", "(\\1)",
                   expr_str, perl = TRUE)
  env <- new.env(parent = baseenv())
  assign("square", function(z) z * z, envir = env)
  assign("exp", exp, envir = env); assign("log", log, envir = env)
  assign("sin", sin, envir = env); assign("cos", cos, envir = env)
  assign("tanh", tanh, envir = env); assign("abs", abs, envir = env)
  assign("sqrt", engine_sqrt, envir = env)
  assign("^", engine_pow, envir = env)
  for (j in seq_len(ncol(X)))
    assign(paste0("x", j - 1L), X[, j], envir = env)
  parsed <- tryCatch(parse(text = expr_str), error = function(e) NULL)
  if (is.null(parsed)) return(NULL)
  out <- tryCatch(suppressWarnings(eval(parsed[[1L]], envir = env)),
                  error = function(e) NULL)
  if (is.null(out)) return(NULL)
  if (length(out) == 1L) out <- rep(out, nrow(X))
  out
}

nmse_of <- function(pred, y) {
  if (is.null(pred)) return(Inf)
  bad <- !is.finite(pred)
  if (all(bad)) return(Inf)
  sum((pred[!bad] - y[!bad])^2) /
    (sum((y[!bad] - mean(y[!bad]))^2) + 1e-300)
}

audit_rows <- function(key, gens, df, expr_col) {
  rec <- df[isTRUE_vec(df$recovered), , drop = FALSE]
  if (nrow(rec) == 0L) return(NULL)
  prob <- feynman_problems[[key]]
  # self-validation data: the exact training sample the runs fitted
  tr <- feynman_dataset(key, n = 1000L, data_seed = 42L, split = "train")
  X <- draw_extrap(key, N_EXTRAP)
  y <- apply(X, 1L, prob$fn)
  stopifnot(all(is.finite(y)))
  ss <- sum((y - mean(y))^2)
  out <- vector("list", nrow(rec))
  for (r in seq_len(nrow(rec))) {
    expr_str <- as.character(rec[[expr_col]][r])
    # self-check: re-evaluating on the training data must reproduce the
    # recorded train NMSE, otherwise this evaluator does not faithfully
    # replicate the engine for this expression and the verdict is void
    check <- nmse_of(eval_expr(expr_str, tr$X), tr$y)
    # Faithful if the re-evaluation reproduces the recorded train NMSE within
    # one order of magnitude, or both are essentially zero (< 1e-8): at that
    # level the discrepancy is float association order, not semantics.
    faithful <- is.finite(check) &&
      ((check < 1e-8 && rec$nmse[r] < 1e-8) ||
       (rec$nmse[r] > 0 &&
        abs(log10(check + 1e-300) - log10(rec$nmse[r] + 1e-300)) < 1))
    pred <- eval_expr(expr_str, X)
    if (is.null(pred)) {
      nmse_ex <- Inf; bad_frac <- 1
    } else {
      bad <- !is.finite(pred)
      bad_frac <- mean(bad)
      nmse_ex <- if (all(bad)) Inf else
        sum((pred[!bad] - y[!bad])^2) / (ss * (1 - bad_frac) + 1e-300)
    }
    verdict <- if (!faithful) "EVAL_MISMATCH"
               else if (bad_frac > 0.05 || nmse_ex > 1e-3) "GRINDER"
               else if (nmse_ex < 1e-6) "TRUE_STRUCTURE"
               else "UNCERTAIN"
    out[[r]] <- data.frame(
      key = key, gens = gens, seed = rec$seed[r],
      nmse_train = rec$nmse[r], nmse_train_check = check,
      nmse_extrap = nmse_ex, nonfinite_frac = bad_frac, verdict = verdict,
      stringsAsFactors = FALSE)
  }
  do.call(rbind, out)
}

isTRUE_vec <- function(v) {
  if (is.logical(v)) v & !is.na(v) else toupper(as.character(v)) == "TRUE"
}

# ---- Manifest ---------------------------------------------------------------

res <- file.path(script_dir, "results")
manifest <- list(
  list(key = "newtons_grav",   gens = 14000, csv = "diag_interference_bc_newtons_grav_g14000_runs.csv"),
  list(key = "harmonic_ke",    gens = 14000, csv = "diag_interference_bc_harmonic_ke_g14000_runs.csv"),
  list(key = "newtons_grav",   gens = 5600,  csv = "diag_interference_bc_newtons_grav_g5600_runs.csv"),
  list(key = "harmonic_ke",    gens = 5600,  csv = "diag_interference_bc_harmonic_ke_g5600_runs.csv"),
  list(key = "boltzmann_dist", gens = 5600,  csv = "diag_interference_bc_boltzmann_dist_g5600_runs.csv"),
  list(key = "planck",         gens = 14000, csv = "diag_interference_bc_planck_g14000_runs.csv"),
  list(key = "bose_einstein",  gens = 14000, csv = "diag_interference_bc_bose_einstein_g14000_runs.csv"),
  list(key = "lorentz_x",      gens = 14000, csv = "diag_interference_eqc_lorentz_14000_runs.csv"),
  list(key = "center_mass",    gens = 14000, csv = "diag_interference_eqc_centermass_14000_runs.csv"),
  list(key = "boltzmann_dist", gens = 14000, csv = "diag_interference_eqc_boltzmann_14000_runs.csv"),
  list(key = "interference",   gens = 8400,  csv = "diag_interference_p2_8400_runs.csv")
)
GATE_CSV  <- "feynman_gate_diag_20260702.csv"
GATE_KEYS <- c("clausius_moss", "lens_eq", "lorentz_x", "center_mass",
               "boltzmann_dist", "harmonic_ke", "newtons_grav")

set.seed(SEED)
all_rows <- list(); totals <- list()

summarize_keys <- function(df, keys, gens, expr_col) {
  for (k in keys) {
    sub <- df[df$key == k, , drop = FALSE]
    a <- audit_rows(k, gens, sub, expr_col)
    totals[[length(totals) + 1L]] <<- data.frame(
      key = k, gens = gens, n_total = nrow(sub),
      n_threshold = sum(isTRUE_vec(sub$recovered)),
      n_true = if (is.null(a)) 0L else sum(a$verdict == "TRUE_STRUCTURE"),
      n_uncertain = if (is.null(a)) 0L else sum(a$verdict == "UNCERTAIN"),
      n_mismatch = if (is.null(a)) 0L else sum(a$verdict == "EVAL_MISMATCH"))
    if (!is.null(a)) all_rows[[length(all_rows) + 1L]] <<- a
  }
}

if (nzchar(CSV_ONLY)) {
  # Single-CSV mode: audit every key in the given gate CSV (docs/47 screen).
  df <- read.csv(file.path(res, CSV_ONLY), stringsAsFactors = FALSE)
  expr_col <- if ("best_expression" %in% names(df)) "best_expression" else "expression"
  summarize_keys(df, intersect(feynman_dev_keys, unique(df$key)), CSV_GENS, expr_col)
} else {

for (m in manifest) {
  path <- file.path(res, m$csv)
  if (!file.exists(path)) { warning("missing: ", m$csv); next }
  df <- read.csv(path, stringsAsFactors = FALSE)
  expr_col <- if ("best_expression" %in% names(df)) "best_expression" else "expression"
  a <- audit_rows(m$key, m$gens, df, expr_col)
  totals[[length(totals) + 1L]] <- data.frame(
    key = m$key, gens = m$gens, n_total = nrow(df),
    n_threshold = sum(isTRUE_vec(df$recovered)),
    n_true = if (is.null(a)) 0L else sum(a$verdict == "TRUE_STRUCTURE"),
    n_uncertain = if (is.null(a)) 0L else sum(a$verdict == "UNCERTAIN"),
    n_mismatch = if (is.null(a)) 0L else sum(a$verdict == "EVAL_MISMATCH"))
  if (!is.null(a)) all_rows[[length(all_rows) + 1L]] <- a
}

gate <- read.csv(file.path(res, GATE_CSV), stringsAsFactors = FALSE)
summarize_keys(gate, GATE_KEYS, 2800, "expression")

}  # end manifest mode

rows_df <- do.call(rbind, all_rows)
tot_df  <- do.call(rbind, totals)
tot_df$p_threshold <- round(tot_df$n_threshold / tot_df$n_total, 3L)
tot_df$p_true      <- round(tot_df$n_true / tot_df$n_total, 3L)

cat("Per-row verdicts (recovered rows only):\n")
print(rows_df, digits = 3L, row.names = FALSE)
cat("\nSummary (p_threshold = NMSE<1e-4 on train; p_true = extrapolation-verified):\n")
print(tot_df, row.names = FALSE)

rows_out <- paste0("structural_audit_rows", OUT_SUF, ".csv")
sum_out  <- paste0("structural_audit_summary", OUT_SUF, ".csv")
write.csv(rows_df, file.path(res, rows_out), row.names = FALSE)
write.csv(tot_df,  file.path(res, sum_out), row.names = FALSE)
message("written: results/", rows_out, ", results/", sum_out)
