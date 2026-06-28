test_that("formula path matches the matrix path for the same data and seed", {
  set.seed(11L)
  a <- runif(30, -2, 2)
  b <- runif(30, -2, 2)
  y <- a * b - a

  Xm <- cbind(a, b)
  res_mat <- symbolic_regression(
    Xm, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 15L,
    seed            = 7L
  )

  df <- data.frame(a = a, b = b, y = y)
  res_fml <- symbolic_regression(
    y ~ a + b, data = df,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 15L,
    seed            = 7L
  )

  # Same inputs reshaped the same way => identical search => identical result.
  expect_identical(res_fml$expression, res_mat$expression)
  expect_identical(res_fml$pareto_front, res_mat$pareto_front)
  expect_identical(res_fml$n_features, 2L)
})

test_that("y ~ . picks up all predictor columns", {
  set.seed(12L)
  df <- data.frame(
    p = runif(20, 1, 3),
    q = runif(20, 1, 3),
    y = NA_real_
  )
  df$y <- df$p + df$q

  res <- symbolic_regression(
    y ~ ., data = df,
    unary_ops       = character(0),
    population_size = 60L,
    generations     = 8L,
    seed            = 1L
  )
  expect_identical(res$n_features, 2L)
  expect_identical(res$feature_names, c("p", "q"))
})

test_that("transformations on the RHS are rejected", {
  df <- data.frame(x = seq(1, 3, length.out = 10), y = seq(1, 3, length.out = 10))
  expect_error(
    symbolic_regression(y ~ log(x), data = df),
    "bare variables"
  )
  expect_error(
    symbolic_regression(y ~ I(x^2), data = df),
    "bare variables"
  )
})

test_that("interaction terms are rejected", {
  df <- data.frame(a = 1:10, b = 1:10, y = 1:10)
  expect_error(symbolic_regression(y ~ a:b, data = df), "interaction")
  expect_error(symbolic_regression(y ~ a * b, data = df), "interaction")
})

test_that("non-numeric predictors are rejected", {
  df <- data.frame(
    g = factor(rep(c("lo", "hi"), 10)),
    x = seq(0, 1, length.out = 20),
    y = seq(0, 1, length.out = 20)
  )
  expect_error(
    symbolic_regression(y ~ g + x, data = df),
    "non-numeric"
  )
})

test_that("a missing response is rejected", {
  df <- data.frame(x = 1:10)
  expect_error(symbolic_regression(~ x, data = df), "response")
})

test_that("predict accepts a data.frame and is column-order independent", {
  set.seed(13L)
  a <- runif(30, -2, 2)
  b <- runif(30, -2, 2)
  y <- 2 * a - b
  df <- data.frame(a = a, b = b, y = y)

  res <- symbolic_regression(
    y ~ a + b, data = df,
    unary_ops       = character(0),
    population_size = 200L,
    generations     = 40L,
    target_loss     = 1e-10,
    seed            = 42L
  )

  p1 <- predict(res, df)
  # Reorder columns: selection by name must give identical predictions.
  p2 <- predict(res, df[, c("b", "y", "a")])
  expect_equal(p1, p2)

  nmse <- sum((p1 - y)^2) / ((length(y) - 1) * var(y))
  expect_lt(nmse, 1e-4)
})

test_that("dropping the intercept (-1) has no effect", {
  set.seed(14L)
  a <- runif(25, -2, 2)
  y <- 3 * a + 1
  df <- data.frame(a = a, y = y)

  common <- list(
    data = df, unary_ops = character(0),
    population_size = 80L, generations = 12L, seed = 5L
  )
  res_int <- do.call(symbolic_regression, c(list(y ~ a), common))
  res_no  <- do.call(symbolic_regression, c(list(y ~ a - 1), common))

  # No design matrix is built, so the intercept term is irrelevant.
  expect_identical(res_no$expression, res_int$expression)
})
