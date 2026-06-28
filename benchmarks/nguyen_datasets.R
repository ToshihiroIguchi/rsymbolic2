# Nguyen symbolic regression benchmark: problem definitions and data generators.
#
# Reference: Uy et al. (2011) "Semantically-based crossover in genetic programming".
# Standard training set: n=20 uniform random samples per domain.
#
# Usage:
#   source("benchmarks/nguyen_datasets.R")
#   ds <- nguyen_dataset("N1", data_seed = 123)
#   # ds$X  : matrix of features
#   # ds$y  : numeric target vector
#   # ds$id : "N1"
#   # ds$formula : "x^3 + x^2 + x"

nguyen_problems <- list(

  N1 = list(
    id      = "N1",
    formula = "x^3 + x^2 + x",
    n_vars  = 1L,
    domains = list(c(-1, 1)),
    fn      = function(x) x[1]^3 + x[1]^2 + x[1]
  ),

  N2 = list(
    id      = "N2",
    formula = "x^4 + x^3 + x^2 + x",
    n_vars  = 1L,
    domains = list(c(-1, 1)),
    fn      = function(x) x[1]^4 + x[1]^3 + x[1]^2 + x[1]
  ),

  N3 = list(
    id      = "N3",
    formula = "x^5 + x^4 + x^3 + x^2 + x",
    n_vars  = 1L,
    domains = list(c(-1, 1)),
    fn      = function(x) x[1]^5 + x[1]^4 + x[1]^3 + x[1]^2 + x[1]
  ),

  N4 = list(
    id      = "N4",
    formula = "x^6 + x^5 + x^4 + x^3 + x^2 + x",
    n_vars  = 1L,
    domains = list(c(-1, 1)),
    fn      = function(x) x[1]^6 + x[1]^5 + x[1]^4 + x[1]^3 + x[1]^2 + x[1]
  ),

  N5 = list(
    id      = "N5",
    formula = "sin(x^2)*cos(x) - 1",
    n_vars  = 1L,
    domains = list(c(-1, 1)),
    fn      = function(x) sin(x[1]^2) * cos(x[1]) - 1
  ),

  N6 = list(
    id      = "N6",
    formula = "sin(x) + sin(x + x^2)",
    n_vars  = 1L,
    domains = list(c(-1, 1)),
    fn      = function(x) sin(x[1]) + sin(x[1] + x[1]^2)
  ),

  N7 = list(
    id      = "N7",
    formula = "log(x + 1) + log(x^2 + 1)",
    n_vars  = 1L,
    domains = list(c(0, 2)),
    fn      = function(x) log(x[1] + 1) + log(x[1]^2 + 1)
  ),

  # N8: sqrt(x). The sqrt operator IS implemented, but recovery requires "sqrt"
  # to be present in unary_ops. The default gate operator set omits it, so N8 is
  # skipped by default and enabled explicitly in the operator-extension run
  # (benchmarks/01b_nguyen_gate_sqrt.R; see docs/14).
  N8 = list(
    id      = "N8",
    formula = "sqrt(x)",
    n_vars  = 1L,
    domains = list(c(0, 4)),
    fn      = function(x) sqrt(x[1]),
    skip    = TRUE,
    skip_reason = "requires 'sqrt' in unary_ops (enabled in the operator-extension run)"
  ),

  N9 = list(
    id      = "N9",
    formula = "sin(x) + sin(y^2)",
    n_vars  = 2L,
    domains = list(c(-1, 1), c(-1, 1)),
    fn      = function(x) sin(x[1]) + sin(x[2]^2)
  ),

  N10 = list(
    id      = "N10",
    formula = "2*sin(x)*cos(y)",
    n_vars  = 2L,
    domains = list(c(-1, 1), c(-1, 1)),
    fn      = function(x) 2 * sin(x[1]) * cos(x[2])
  )
)

# Generate a training dataset for a Nguyen problem.
#
# @param problem_id  Character: "N1" through "N10".
# @param n           Number of training points (default 20; standard protocol).
# @param data_seed   RNG seed for data generation. Fixed across all benchmark runs
#                    so that only the SR seed varies between seeds 1:5.
#
# @return List: $X (n × p matrix), $y (length-n vector), $id, $formula, $skip.
nguyen_dataset <- function(problem_id, n = 20L, data_seed = 123L) {
  prob <- nguyen_problems[[problem_id]]
  if (is.null(prob)) stop("Unknown Nguyen problem: ", problem_id)

  set.seed(data_seed)
  p <- prob$n_vars
  X <- matrix(NA_real_, nrow = n, ncol = p)
  for (j in seq_len(p)) {
    lo <- prob$domains[[j]][1]
    hi <- prob$domains[[j]][2]
    X[, j] <- runif(n, lo, hi)
  }

  y <- apply(X, 1, prob$fn)

  list(
    X       = X,
    y       = y,
    id      = prob$id,
    formula = prob$formula,
    skip    = isTRUE(prob$skip),
    skip_reason = prob$skip_reason %||% ""
  )
}

# Null-coalescing helper (not in base R < 4.4).
`%||%` <- function(x, y) if (is.null(x)) y else x
