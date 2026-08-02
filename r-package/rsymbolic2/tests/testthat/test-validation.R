test_that("a factor y is rejected rather than silently using level codes", {
  X <- matrix(seq(-3, 3, length.out = 6), ncol = 1)
  # as.numeric() on this factor gives c(1, 2, 3, 1, 2, 3), not the numbers shown.
  y <- factor(c(100, 20, 3, 100, 20, 3))
  expect_error(symbolic_regression(X, y), "level codes")
})

test_that("a factor response is rejected by the formula method too", {
  d <- data.frame(a = seq(-3, 3, length.out = 6),
                  b = factor(c(100, 20, 3, 100, 20, 3)))
  expect_error(symbolic_regression(b ~ a, data = d), "level codes")
})

test_that("non-finite X or y is rejected", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- 2 * X[, 1] + 1

  Xbad <- X; Xbad[3, 1] <- NA
  expect_error(symbolic_regression(Xbad, y), "X must not contain")

  Xbad2 <- X; Xbad2[3, 1] <- Inf
  expect_error(symbolic_regression(Xbad2, y), "X must not contain")

  ybad <- y; ybad[4] <- NaN
  expect_error(symbolic_regression(X, ybad), "y must not contain")

  ybad2 <- y; ybad2[4] <- -Inf
  expect_error(symbolic_regression(X, ybad2), "y must not contain")
})

test_that("non-positive counts are rejected instead of aborting the session", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- 2 * X[, 1] + 1
  # population_size = -1 used to wrap to a huge size_t and terminate the R session
  # from inside the OpenMP region.
  expect_error(symbolic_regression(X, y, population_size = -1L), "population_size")
  expect_error(symbolic_regression(X, y, population_size = 0L),  "population_size")
  expect_error(symbolic_regression(X, y, generations = 0L),      "generations")
  expect_error(symbolic_regression(X, y, generations = -5L),     "generations")
  expect_error(symbolic_regression(X, y, tournament_size = 0L),  "tournament_size")
  expect_error(symbolic_regression(X, y, max_nodes = 0L),        "max_nodes")
  expect_error(symbolic_regression(X, y, max_nodes = -3L),       "max_nodes")
  expect_error(symbolic_regression(X, y, max_depth = 0L),        "max_depth")
  expect_error(symbolic_regression(X, y, n_populations = 0L),    "n_populations")
  expect_error(symbolic_regression(X, y, n_populations = -2L),   "n_populations")
})

test_that("mismatched nrow(X) and length(y) is rejected", {
  X <- matrix(1:10, ncol = 1)
  y <- 1:5
  expect_error(symbolic_regression(X, y), "nrow")
})

test_that("empty binary_ops is rejected", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- X[, 1]
  expect_error(
    symbolic_regression(X, y, binary_ops = character(0)),
    "binary_ops"
  )
})
