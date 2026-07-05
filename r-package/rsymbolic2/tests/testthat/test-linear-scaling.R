# Tests for the opt-in Keijzer-2003 linear scaling (linear_scaling). Unlike
# eval_cache this is a behaviour-CHANGING opt-in: with the flag on, every candidate
# is scored by its best-affine-fit SSE and the fitted a*f+b is materialised into
# the returned expressions. With the flag off (the default) the search must be
# identical to not passing the argument at all (PySR parity is untouched).

line_data <- function(n = 30L) {
  X <- matrix(seq(-5, 5, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  list(X = X, y = y)
}

test_that("linear_scaling defaults off and the off-run is unchanged", {
  expect_identical(
    formals(utils::getS3method("symbolic_regression", "default"))$linear_scaling,
    FALSE
  )
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 20L, seed = 11L,
    verbosity = 0L
  )
  r_default <- do.call(symbolic_regression, common)
  r_off     <- do.call(symbolic_regression, c(common, list(linear_scaling = FALSE)))
  expect_identical(r_off$expression, r_default$expression)
  expect_identical(r_off$loss, r_default$loss)
  expect_identical(r_off$pareto_front$expression, r_default$pareto_front$expression)
})

test_that("linear_scaling returns a finite, self-consistent materialised fit", {
  d <- line_data()
  res <- symbolic_regression(
    d$X, d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 20L, seed = 11L,
    linear_scaling = TRUE, verbosity = 0L
  )
  expect_s3_class(res, "rsymbolic2")
  expect_true(is.finite(res$loss))
  # The reported loss must be reproducible from the returned expression string:
  # predict() on the training data gives the same SSE up to the %.6g string
  # round-trip of the materialised constants (which bounds accuracy to ~1e-4
  # relative here).
  pred <- predict(res, d$X, expression = res$expression)
  sse <- sum((pred - d$y)^2)
  expect_lt(abs(sse - res$loss), 1e-4 * (1 + abs(res$loss)))
})

test_that("linear_scaling is rejected in combination with units", {
  d <- line_data()
  expect_error(
    symbolic_regression(
      d$X, d$y, X_units = "m", linear_scaling = TRUE,
      population_size = 20L, generations = 2L, seed = 1L, verbosity = 0L
    ),
    "not supported with dimensional analysis"
  )
})
