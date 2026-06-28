test_that("timeout_seconds stops the search before full budget", {
  skip_on_cran()
  X <- matrix(seq(-2, 2, length.out = 20), ncol = 1)
  y <- X[, 1]^4 + X[, 1]^3 + X[, 1]^2 + X[, 1]  # N2: needs many generations

  t0 <- proc.time()[["elapsed"]]
  res <- symbolic_regression(
    X, y,
    population_size = 500L,
    n_populations   = 4L,
    generations     = 500L,   # far more than would normally finish in time
    seed            = 1L,
    timeout_seconds = 3        # hard ceiling
  )
  elapsed <- proc.time()[["elapsed"]] - t0

  # Must return within ~2x the timeout (some OS scheduling slack).
  expect_lt(elapsed, 10)

  # Must still return a valid result structure.
  expect_type(res$expression, "character")
  expect_true(is.finite(res$loss))
  expect_true(is.data.frame(res$pareto_front))
})

test_that("timeout_seconds = 0 (default) does not change deterministic behavior", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 12), ncol = 1)
  y <- X[, 1]^2
  a <- symbolic_regression(X, y, population_size = 80L, generations = 15L, seed = 7L)
  b <- symbolic_regression(X, y, population_size = 80L, generations = 15L, seed = 7L,
                            timeout_seconds = 0)
  expect_equal(a$expression, b$expression)
})
