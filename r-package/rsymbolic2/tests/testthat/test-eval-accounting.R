# Tests for the evaluation-accounting result fields: n_evals (total candidate
# evaluations in max_evals units) and eval_counts (forward / lm_resid / lm_jac
# breakdown). These are behaviour-neutral instrumentation: they report how much
# work the search did, never change what it does.

line_data <- function(n = 16L) {
  X <- matrix(seq(-8, 8, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  list(X = X, y = y)
}

test_that("result carries n_evals and a consistent eval_counts breakdown", {
  d <- line_data()
  res <- symbolic_regression(
    d$X, d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 2L, generations = 40L, seed = 11L
  )

  expect_true(is.numeric(res$n_evals))
  expect_length(res$n_evals, 1L)
  expect_gt(res$n_evals, 0)

  expect_true(is.numeric(res$eval_counts))
  expect_length(res$eval_counts, 3L)
  expect_named(res$eval_counts, c("forward", "lm_resid", "lm_jac"))
  expect_true(all(res$eval_counts >= 0))
  # n_evals is the max_evals unit: forward passes + LM residual evaluations.
  # Jacobian builds are reported only and never charged to n_evals.
  expect_equal(res$n_evals,
               unname(res$eval_counts["forward"] + res$eval_counts["lm_resid"]))
})

test_that("eval accounting is deterministic for a fixed seed", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 2L, generations = 40L, seed = 7L
  )
  r1 <- do.call(symbolic_regression, common)
  r2 <- do.call(symbolic_regression, common)

  expect_identical(r1$n_evals, r2$n_evals)
  expect_identical(r1$eval_counts, r2$eval_counts)
})
