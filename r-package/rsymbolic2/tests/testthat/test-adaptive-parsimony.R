# Tests for frequency-based adaptive parsimony (adaptive_parsimony_scaling).
# The quantitative anti-collapse / recovery evidence lives in the sweep recorded
# in docs/24; these tests cover the correctness invariants that must always hold.

test_that("default config is PySR parity and an explicit-equivalent call reproduces it", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- X[, 1]^2 + 2 * X[, 1] + 1
  # The shipped defaults are PySR parity: parsimony = 0 and
  # adaptive_parsimony_scaling = 1040 (docs/28). Passing those explicitly must
  # reproduce the default path byte-for-byte (frequency updates consume no RNG).
  a <- symbolic_regression(X, y, unary_ops = character(0),
                           population_size = 200L, generations = 50L, seed = 1L)
  b <- symbolic_regression(X, y, unary_ops = character(0),
                           population_size = 200L, generations = 50L, seed = 1L,
                           parsimony = 0, adaptive_parsimony_scaling = 1040)
  expect_equal(a$expression, b$expression)
  expect_equal(a$loss, b$loss)
})

test_that("adaptive_parsimony_scaling=0 disables the frequency factor and recovers", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- X[, 1]^2 + 2 * X[, 1] + 1
  # scaling <= 0 turns the frequency factor off; the search must still recover.
  res <- symbolic_regression(X, y, unary_ops = character(0),
                             population_size = 200L, generations = 50L, seed = 1L,
                             parsimony = 0, adaptive_parsimony_scaling = 0)
  expect_lt(res$loss, 1e-6)
})

test_that("adaptive parsimony is reproducible for a fixed seed", {
  skip_on_cran()
  # Frequency updates consume no RNG, so a fixed seed must reproduce exactly.
  X <- matrix(seq(-2, 2, length.out = 16), ncol = 1)
  y <- X[, 1]^3
  a <- symbolic_regression(X, y, population_size = 100L, generations = 25L,
                           seed = 5L, parsimony = 0,
                           adaptive_parsimony_scaling = 20)
  b <- symbolic_regression(X, y, population_size = 100L, generations = 25L,
                           seed = 5L, parsimony = 0,
                           adaptive_parsimony_scaling = 20)
  expect_equal(a$expression, b$expression)
})

test_that("positive scaling does not break search on a simple target", {
  skip_on_cran()
  # With the frequency penalty active the search must still reach a known target.
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- X[, 1]^2 + 2 * X[, 1] + 1
  res <- symbolic_regression(X, y, unary_ops = character(0),
                             population_size = 300L, generations = 60L,
                             seed = 3L, parsimony = 0,
                             adaptive_parsimony_scaling = 20)
  expect_true(is.finite(res$loss))
  expect_lt(res$loss, 1e-4)
})

test_that("multi-island runs with adaptive parsimony are reproducible", {
  skip_on_cran()
  # Per-island statistics are share-nothing; a fixed seed must reproduce across
  # the island-model path too.
  set.seed(11)
  n <- 40
  x1 <- runif(n, 1, 3); x2 <- runif(n, 1, 3)
  X <- cbind(x1, x2)
  y <- (x1 + 2 * x2) / (x1 + x2)
  common <- list(X = X, y = y, population_size = 150L, generations = 20L,
                 n_populations = 2L, seed = 9L, parsimony = 0,
                 binary_ops = c("add", "sub", "mul", "div"),
                 adaptive_parsimony_scaling = 20)
  a <- do.call(symbolic_regression, common)
  b <- do.call(symbolic_regression, common)
  expect_equal(a$expression, b$expression)
})
