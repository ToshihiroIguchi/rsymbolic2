# Tests for the opt-in search-time strong simplification (strong_simplify). Like
# linear_scaling this is a documented opt-in (docs/55); unlike eval_cache it is not
# claimed to be bit-identical on/off (see docs/54's display simplifier). With the
# flag off (the default) the search must be identical to not passing the argument
# at all (PySR parity is untouched).

line_data <- function(n = 30L) {
  X <- matrix(seq(-5, 5, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  list(X = X, y = y)
}

test_that("strong_simplify defaults off and the off-run is unchanged", {
  expect_identical(
    formals(utils::getS3method("symbolic_regression", "default"))$strong_simplify,
    FALSE
  )
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 20L, seed = 11L,
    n_threads = 1L, verbosity = 0L
  )
  r_default <- do.call(symbolic_regression, common)
  r_off     <- do.call(symbolic_regression, c(common, list(strong_simplify = FALSE)))
  expect_identical(r_off$expression, r_default$expression)
  expect_identical(r_off$loss, r_default$loss)
  expect_identical(r_off$pareto_front$expression, r_default$pareto_front$expression)
})

test_that("strong_simplify runs without error and produces a finite loss", {
  d <- line_data()
  res <- symbolic_regression(
    d$X, d$y,
    unary_ops = c("neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"),
    binary_ops = c("add", "sub", "mul", "div", "pow"),
    population_size = 60L, n_populations = 1L, generations = 20L, seed = 11L,
    n_threads = 1L, strong_simplify = TRUE, verbosity = 0L
  )
  expect_s3_class(res, "rsymbolic2")
  expect_true(is.finite(res$loss))
  expect_gte(unname(res$eval_counts[["strong_simplify_attempts"]]),
             unname(res$eval_counts[["strong_simplify_adopted"]]))
  # A generous operator set gives the simplifier plenty of opportunities to fire.
  expect_gt(unname(res$eval_counts[["strong_simplify_attempts"]]), 0)
})

test_that("eval_counts carries strong_simplify counters: populated on, zero off", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y,
    unary_ops = c("neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"),
    binary_ops = c("add", "sub", "mul", "div", "pow"),
    population_size = 60L, n_populations = 1L, generations = 20L, seed = 11L,
    n_threads = 1L, verbosity = 0L
  )
  r_off <- do.call(symbolic_regression, common)
  r_on  <- do.call(symbolic_regression, c(common, list(strong_simplify = TRUE)))

  expect_length(r_off$eval_counts, 7L)
  expect_named(r_off$eval_counts,
               c("forward", "lm_resid", "lm_jac", "cache_hits", "cache_misses",
                 "strong_simplify_attempts", "strong_simplify_adopted"))
  expect_identical(unname(r_off$eval_counts["strong_simplify_attempts"]), 0)
  expect_identical(unname(r_off$eval_counts["strong_simplify_adopted"]), 0)

  expect_gt(unname(r_on$eval_counts["strong_simplify_attempts"]), 0)
  expect_gte(unname(r_on$eval_counts["strong_simplify_attempts"]),
             unname(r_on$eval_counts["strong_simplify_adopted"]))
})
