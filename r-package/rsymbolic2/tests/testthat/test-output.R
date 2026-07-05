test_that("result has the documented structure", {
  X <- matrix(seq(-3, 3, length.out = 12), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops      = character(0),
    population_size = 50L,
    generations    = 10L,
    seed           = 1L
  )
  expect_named(res, c("expression", "loss", "complexity", "recommended",
                      "best_index", "pareto_front", "n_obs", "sst",
                      "n_evals", "eval_counts", "n_features"))
  expect_identical(res$n_obs, 12L)
  expect_true(is.finite(res$sst) && res$sst > 0)
  expect_type(res$expression, "character")
  expect_true(nchar(res$expression) > 0)
  expect_true(is.finite(res$loss))
  expect_true(res$complexity > 0L)
  expect_type(res$recommended, "character")
  expect_true(nchar(res$recommended) > 0)
  expect_true(res$best_index >= 1L && res$best_index <= nrow(res$pareto_front))
  expect_s3_class(res$pareto_front, "data.frame")
  expect_named(res$pareto_front,
               c("complexity", "loss", "score", "expression", "latex"))
  expect_type(res$pareto_front$score, "double")
  expect_identical(res$pareto_front$score[1L], 0)
  expect_true(nrow(res$pareto_front) >= 1L)
  # The recommended expression must be the one best_index points to in the front.
  expect_equal(res$recommended, res$pareto_front$expression[res$best_index])
  expect_equal(res$n_features, 1L)
})
