test_that("macro_ops default NULL leaves the result unchanged", {
  X <- matrix(seq(-2, 2, length.out = 20), ncol = 1)
  y <- 1.5 * X[, 1] + 0.5
  common <- list(
    X = X, y = y, unary_ops = c("exp"), binary_ops = c("add", "mul"),
    population_size = 60L, generations = 30L, seed = 11L
  )
  res_off  <- do.call(symbolic_regression, common)
  res_null <- do.call(symbolic_regression, c(common, list(macro_ops = NULL)))
  # macro_ops = NULL must be the same run, not merely a comparable one.
  expect_identical(res_off$expression, res_null$expression)
  expect_identical(res_off$loss, res_null$loss)
})

test_that("a macro operator makes an otherwise unreachable target reachable", {
  skip_on_cran()
  # y = 3 * exp(-x^2). With no unary operators enabled the search cannot build the
  # Gaussian at all; the macro supplies it as one template.
  X <- matrix(seq(-2, 2, length.out = 30), ncol = 1)
  y <- 3 * exp(-(X[, 1]^2))
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0L),
    binary_ops      = c("add", "mul"),
    macro_ops       = c(gauss = "exp(neg(square(x)))"),
    population_size = 200L,
    generations     = 80L,
    seed            = 4L
  )
  expect_true(is.finite(res$loss))
  expect_lt(res$loss, 1e-3)
  # The macro is expanded, so the reported expression is in primitive form.
  expect_true(grepl("exp", res$expression, fixed = TRUE))
  expect_false(grepl("gauss", res$expression, fixed = TRUE))
  # predict() re-parses that expression: the round trip must still work.
  expect_true(all(is.finite(predict(res, X, expression = res$expression))))
})

test_that("invalid macro bodies are rejected with a useful message", {
  X <- matrix(seq(1, 2, length.out = 8), ncol = 1)
  y <- X[, 1]
  run <- function(macros) {
    symbolic_regression(X, y, binary_ops = c("add", "mul"), macro_ops = macros,
                        population_size = 10L, generations = 1L)
  }
  expect_error(run(c(f = "exp(x) + x")), "exactly once")
  expect_error(run(c(f = "exp(1)")), "exactly once")
  expect_error(run(c(f = "gauss(x)")), "unknown function")
  expect_error(run(c(f = "exp(x")), "expected ')'")
  expect_error(run(c(f = "x0 + x")), "unknown identifier")
  expect_error(run(c(square = "exp(x)")), "shadows")
  expect_error(run("exp(neg(x))"), "named character vector")
})
