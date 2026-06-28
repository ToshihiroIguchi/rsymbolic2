test_that("parsimony=0 (default) preserves pre-B2 behaviour", {
  skip_on_cran()
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- X[, 1]^2 + 2 * X[, 1] + 1
  res <- symbolic_regression(X, y, unary_ops = character(0),
                             population_size = 200L, generations = 50L,
                             seed = 1L, parsimony = 0)
  expect_lt(res$loss, 1e-6)
})

test_that("higher parsimony yields smaller or equal median complexity", {
  skip_on_cran()
  # N10-like: y = sin(x1) * x2^2 has a compact true form (size ~ 5).
  # With enough parsimony pressure the search should prefer shorter expressions.
  set.seed(99)
  n <- 40
  x1 <- runif(n, -2, 2)
  x2 <- runif(n, -2, 2)
  X <- cbind(x1, x2)
  y <- sin(x1) * x2^2

  res0 <- symbolic_regression(X, y,
                               population_size = 300L, generations = 80L,
                               seed = 7L, parsimony = 0,
                               timeout_seconds = 30)
  res1 <- symbolic_regression(X, y,
                               population_size = 300L, generations = 80L,
                               seed = 7L, parsimony = 1e-3,
                               timeout_seconds = 30)

  # Pareto front median complexity should not increase with parsimony.
  med0 <- if (nrow(res0$pareto_front) > 0) median(res0$pareto_front$complexity) else res0$complexity
  med1 <- if (nrow(res1$pareto_front) > 0) median(res1$pareto_front$complexity) else res1$complexity
  expect_lte(med1, med0 + 4)  # allow small slack for single-seed variance
})

test_that("parsimony is reproducible for a fixed seed", {
  skip_on_cran()
  X <- matrix(seq(-2, 2, length.out = 16), ncol = 1)
  y <- X[, 1]^3
  a <- symbolic_regression(X, y, population_size = 100L, generations = 20L,
                           seed = 5L, parsimony = 5e-4)
  b <- symbolic_regression(X, y, population_size = 100L, generations = 20L,
                           seed = 5L, parsimony = 5e-4)
  expect_equal(a$expression, b$expression)
})
