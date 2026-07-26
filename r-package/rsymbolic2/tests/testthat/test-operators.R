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

test_that("square operator: y = x^2 + 1 is recoverable", {
  skip_on_cran()
  set.seed(42)
  X <- matrix(seq(-2, 2, length.out = 25), ncol = 1)
  y <- X[, 1]^2 + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = "square",
    binary_ops      = c("add", "mul"),
    population_size = 300L,
    generations     = 100L,
    seed            = 1L
  )
  expect_true(is.finite(res$loss))
  expect_false(is.nan(res$loss))
  expect_lt(res$loss, 1e-3)
})

test_that("inv operator: y = 1/x is recoverable", {
  skip_on_cran()
  # x stays away from 0: inv is unguarded (like div), so a pole would only be
  # rejected by the loss guard, not silently absorbed.
  X <- matrix(seq(0.5, 4, length.out = 25), ncol = 1)
  y <- 1 / X[, 1]
  res <- symbolic_regression(
    X, y,
    unary_ops       = "inv",
    binary_ops      = c("add", "mul"),
    population_size = 300L,
    generations     = 100L,
    seed            = 1L
  )
  expect_true(is.finite(res$loss))
  expect_false(is.nan(res$loss))
  expect_lt(res$loss, 1e-3)
})

test_that("erf operator: y = erf(x) is recoverable, and predict() agrees", {
  skip_on_cran()
  r_erf <- function(x) 2 * stats::pnorm(x * sqrt(2)) - 1
  X <- matrix(seq(-2, 2, length.out = 25), ncol = 1)
  y <- r_erf(X[, 1])
  res <- symbolic_regression(
    X, y,
    unary_ops       = "erf",
    binary_ops      = c("add", "mul"),
    population_size = 300L,
    generations     = 100L,
    seed            = 1L
  )
  expect_true(is.finite(res$loss))
  expect_false(is.nan(res$loss))
  expect_lt(res$loss, 1e-3)
  # predict() re-parses the expression string in R, where erf is not a base function:
  # the shim must reproduce the core's std::erf, not merely something erf-shaped. The
  # tolerance is loose because the printed constants carry only "%.6g" digits (docs/48
  # D2), not because the shim is approximate -- a wrong erf misses by far more.
  expect_equal(sum((y - predict(res, X))^2), res$loss, tolerance = 1e-3)
})

test_that("sinh/cosh operators: y = sinh(x) is recoverable", {
  skip_on_cran()
  X <- matrix(seq(-2, 2, length.out = 25), ncol = 1)
  y <- sinh(X[, 1])
  res <- symbolic_regression(
    X, y,
    unary_ops       = c("sinh", "cosh"),
    binary_ops      = c("add", "mul"),
    population_size = 300L,
    generations     = 100L,
    seed            = 1L
  )
  expect_true(is.finite(res$loss))
  expect_false(is.nan(res$loss))
  expect_lt(res$loss, 1e-3)
  expect_equal(sum((y - predict(res, X))^2), res$loss, tolerance = 1e-3)
})

test_that("pow operator: y = x^3 returns valid finite result", {
  skip_on_cran()
  X <- matrix(seq(0.5, 2.5, length.out = 20), ncol = 1)
  y <- X[, 1]^3
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0L),
    binary_ops      = c("add", "mul", "pow"),
    population_size = 400L,
    generations     = 150L,
    seed            = 2L
  )
  expect_false(is.nan(res$loss))
  expect_true(is.finite(res$loss))
  expect_type(res$expression, "character")
})

test_that("pow operator: negative base with integer exponent is finite", {
  skip_on_cran()
  X <- matrix(seq(-2, 2, length.out = 20), ncol = 1)
  y <- X[, 1]^2
  res <- symbolic_regression(
    X, y,
    unary_ops       = "square",
    binary_ops      = c("add", "mul", "pow"),
    population_size = 200L,
    generations     = 60L,
    seed            = 3L
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
