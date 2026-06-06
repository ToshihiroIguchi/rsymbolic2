test_that("sqrt operator: y = sqrt(x) is recoverable", {
  skip_on_cran()
  X <- matrix(seq(0.1, 4, length.out = 25), ncol = 1)
  y <- sqrt(X[, 1])
  res <- symbolic_regression(
    X, y,
    unary_ops       = "sqrt",
    binary_ops      = c("add", "mul"),
    population_size = 300L,
    generations     = 100L,
    seed            = 1L
  )
  expect_true(is.finite(res$loss))
  expect_false(is.nan(res$loss))
  # Loss should be tiny for clean data with the right operator available
  expect_lt(res$loss, 1e-3)
})

test_that("tanh operator does not produce NaN or Inf loss", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- tanh(X[, 1])
  res <- symbolic_regression(
    X, y,
    unary_ops       = "tanh",
    binary_ops      = c("add", "mul"),
    population_size = 200L,
    generations     = 60L,
    seed            = 2L
  )
  expect_false(is.nan(res$loss))
  expect_true(is.finite(res$loss))
})

test_that("abs operator runs and returns valid result", {
  skip_on_cran()
  X <- matrix(seq(-2, 2, length.out = 15), ncol = 1)
  y <- abs(X[, 1])
  res <- symbolic_regression(
    X, y,
    unary_ops       = "abs",
    binary_ops      = c("add", "mul"),
    population_size = 200L,
    generations     = 60L,
    seed            = 3L
  )
  expect_true(is.list(res))
  expect_false(is.nan(res$loss))
  expect_true(is.finite(res$loss))
  expect_type(res$expression, "character")
})
