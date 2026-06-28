test_that("linear target is recovered (loss < 1e-6)", {
  skip_on_cran()
  X <- matrix(seq(-5, 5, length.out = 20), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 200L,
    generations     = 40L,
    seed            = 42L
  )
  expect_lt(res$loss, 1e-6)
})

test_that("same seed gives identical result", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 12), ncol = 1)
  y <- X[, 1]^2
  a <- symbolic_regression(X, y, population_size = 80L, generations = 15L, seed = 7L)
  b <- symbolic_regression(X, y, population_size = 80L, generations = 15L, seed = 7L)
  expect_equal(a$expression, b$expression)
})

test_that("two-feature search runs without error", {
  skip_on_cran()
  X <- as.matrix(expand.grid(seq(-2, 2, length.out = 5),
                              seq(-2, 2, length.out = 5)))
  y <- X[, 1] * X[, 2] + X[, 1]
  expect_no_error(
    symbolic_regression(
      X, y,
      unary_ops       = character(0),
      population_size = 100L,
      generations     = 20L,
      seed            = 42L
    )
  )
})
