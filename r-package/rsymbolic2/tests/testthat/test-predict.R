test_that("predict returns numeric vector matching nrow(newdata)", {
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 20L,
    seed            = 1L
  )
  X_new <- matrix(c(-1, 0, 1), ncol = 1)
  preds <- predict(res, X_new)
  expect_type(preds, "double")
  expect_length(preds, 3L)
  expect_true(all(is.finite(preds)))
})

test_that("predict matches training predictions closely for a recovered linear equation", {
  X <- matrix(seq(-2, 2, length.out = 30), ncol = 1)
  y <- 3 * X[, 1] - 2
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 200L,
    generations     = 40L,
    target_loss     = 1e-10,
    seed            = 42L
  )
  preds <- predict(res, X)
  # Prediction should reconstruct training targets closely if equation recovered
  nmse <- sum((preds - y)^2) / ((length(y) - 1) * var(y))
  expect_lt(nmse, 1e-4)
})

test_that("predict errors when newdata has wrong number of columns", {
  X <- matrix(seq(-2, 2, length.out = 20), ncol = 2)
  y <- X[, 1] + X[, 2]
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 50L,
    generations     = 5L,
    seed            = 1L
  )
  X_wrong <- matrix(1:9, ncol = 3)
  expect_error(predict(res, X_wrong), "3 column")
})

test_that("predict handles multi-feature input", {
  set.seed(7L)
  X <- matrix(runif(60, 1, 3), nrow = 20, ncol = 3)
  y <- X[, 1] * X[, 2] + X[, 3]
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 10L,
    seed            = 3L
  )
  X_new <- matrix(c(2, 2, 1), nrow = 1)
  preds <- predict(res, X_new)
  expect_length(preds, 1L)
  expect_true(is.finite(preds))
})

test_that("predict handles the inv operator in expression", {
  # y = 2/x forces inv into the expression; predict() re-parses the infix string,
  # so this is what catches a missing inv() shim in the evaluation environment.
  X <- matrix(seq(0.5, 4, length.out = 20), ncol = 1)
  y <- 2 / X[, 1]
  res <- symbolic_regression(
    X, y,
    unary_ops       = "inv",
    binary_ops      = c("add", "mul"),
    population_size = 200L,
    generations     = 60L,
    target_loss     = 1e-10,
    seed            = 3L
  )
  preds <- predict(res, X, expression = res$expression)
  expect_true(all(is.finite(preds)))
  # The re-parsed expression must reproduce the fitted values.
  expect_lt(sum((preds - y)^2), res$loss + 1e-8)
})

test_that("predict recycles a constant expression to nrow(newdata)", {
  # The simplest Pareto member is usually a bare constant, and `expression=` is
  # documented to accept any member, so a data-free expression must still return one
  # prediction per row rather than the single value R's evaluator produces.
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 40L,
    generations     = 20L,
    n_populations   = 2L,
    seed            = 1L
  )
  X_new <- matrix(c(-1, 0, 1), ncol = 1)
  preds <- predict(res, X_new, expression = "4.2")
  expect_length(preds, 3L)
  expect_equal(preds, rep(4.2, 3L))

  # The simplest archived member, whatever it turned out to be, obeys the same rule.
  simplest <- predict(res, X_new, expression = res$pareto_front$expression[1L])
  expect_length(simplest, 3L)
})

test_that("predict evaluates the inf/nan constant tokens the core can emit", {
  # "%.6g" renders a non-finite constant as the bare token `inf`/`nan`, which R parses
  # as a name. The package's tree renderer already recognises them; predict() must too.
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 40L,
    generations     = 10L,
    n_populations   = 2L,
    seed            = 1L
  )
  X_new <- matrix(c(1, 2), ncol = 1)
  expect_equal(predict(res, X_new, expression = "(x0 * inf)"), c(Inf, Inf))
  expect_true(all(is.nan(predict(res, X_new, expression = "(x0 * nan)"))))
})

test_that("a formula-fitted model requires a data.frame newdata", {
  # A matrix would be matched positionally with only an ncol check, so a caller who
  # fitted by name could get silently wrong numbers from a reordered matrix.
  set.seed(11L)
  df <- data.frame(a = runif(20, -2, 2), b = runif(20, -2, 2))
  df$y <- 2 * df$a - df$b
  res <- symbolic_regression(
    y ~ a + b, data = df,
    unary_ops       = character(0),
    population_size = 40L,
    generations     = 10L,
    n_populations   = 2L,
    seed            = 1L
  )
  expect_error(predict(res, as.matrix(df[1:3, c("b", "a")])), "must be a data.frame")
  # The data.frame path is unaffected, in any column order.
  expect_equal(
    predict(res, df[1:3, c("b", "a")]),
    predict(res, df[1:3, c("a", "b")])
  )
})

test_that("predict rejects a non-numeric newdata column with a clear message", {
  X <- cbind(seq(-2, 2, length.out = 20), seq(1, 3, length.out = 20))
  y <- X[, 1] + X[, 2]
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 40L,
    generations     = 5L,
    n_populations   = 2L,
    seed            = 1L
  )
  bad <- data.frame(a = c(1, 2, 3), b = factor(c("p", "q", "p")))
  expect_error(predict(res, bad), "must be numeric")
})

test_that("predict handles neg and square operators in expression", {
  # Construct a fit that forces neg and square into the expression by fitting
  # y = -x^2 (neg + square both needed)
  X <- matrix(seq(-2, 2, length.out = 20), ncol = 1)
  y <- -(X[, 1]^2)
  res <- symbolic_regression(
    X, y,
    unary_ops       = c("neg", "square"),
    binary_ops      = c("add", "sub", "mul"),
    population_size = 200L,
    generations     = 40L,
    target_loss     = 1e-10,
    seed            = 5L
  )
  preds <- predict(res, X)
  expect_true(all(is.finite(preds)))
})
