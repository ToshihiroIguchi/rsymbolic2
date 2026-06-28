test_that("optimize_probability = 0.2 recovers a simple linear target", {
  skip_on_cran()
  X <- matrix(seq(-5, 5, length.out = 30), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  res <- symbolic_regression(
    X, y,
    unary_ops            = character(0),
    population_size      = 300L,
    generations          = 60L,
    seed                 = 42L,
    optimize_probability = 0.2
  )
  # Looser than the p=1 test; recovery with partial optimization is expected.
  expect_lt(res$loss, 1e-2)
})

test_that("optimize_probability is reproducible for a fixed seed", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 12), ncol = 1)
  y <- X[, 1]^2
  a <- symbolic_regression(X, y, population_size = 80L, generations = 15L,
                           seed = 7L, optimize_probability = 0.3)
  b <- symbolic_regression(X, y, population_size = 80L, generations = 15L,
                           seed = 7L, optimize_probability = 0.3)
  expect_equal(a$expression, b$expression)
})

test_that("optimize_probability = 1 matches pre-B1 recovery behavior", {
  skip_on_cran()
  X <- matrix(seq(-5, 5, length.out = 20), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  res <- symbolic_regression(
    X, y,
    unary_ops            = character(0),
    population_size      = 200L,
    generations          = 40L,
    seed                 = 42L,
    optimize_probability = 1.0
  )
  expect_lt(res$loss, 1e-6)
})
