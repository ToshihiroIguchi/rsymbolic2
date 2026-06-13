test_that("result has the documented five-element structure", {
  X <- matrix(seq(-3, 3, length.out = 12), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops      = character(0),
    population_size = 50L,
    generations    = 10L,
    seed           = 1L
  )
  expect_named(res, c("expression", "loss", "complexity", "pareto_front", "n_features"))
  expect_type(res$expression, "character")
  expect_true(nchar(res$expression) > 0)
  expect_true(is.finite(res$loss))
  expect_true(res$complexity > 0L)
  expect_s3_class(res$pareto_front, "data.frame")
  expect_named(res$pareto_front, c("complexity", "loss", "expression"))
  expect_true(nrow(res$pareto_front) >= 1L)
  expect_equal(res$n_features, 1L)
})
