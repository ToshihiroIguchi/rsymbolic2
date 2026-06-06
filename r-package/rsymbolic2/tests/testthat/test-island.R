test_that("n_populations=4 recovers linear target", {
  skip_on_cran()
  X <- matrix(seq(-5, 5, length.out = 20), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    n_populations   = 4L,
    population_size = 200L,
    generations     = 40L,
    seed            = 42L
  )
  expect_lt(res$loss, 1e-6)
})

test_that("n_populations=4 with same seed gives identical result", {
  skip_on_cran()
  X <- matrix(seq(-5, 5, length.out = 20), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  a <- symbolic_regression(X, y, unary_ops = character(0),
                           n_populations = 4L, population_size = 100L,
                           generations = 20L, seed = 99L)
  b <- symbolic_regression(X, y, unary_ops = character(0),
                           n_populations = 4L, population_size = 100L,
                           generations = 20L, seed = 99L)
  expect_equal(a$expression, b$expression)
})
