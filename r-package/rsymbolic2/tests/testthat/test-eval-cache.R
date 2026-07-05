# Tests for the opt-in duplicate-evaluation cache (eval_cache). It is an
# implementation-only memoisation: with the cache on, results must be bit-identical
# to the cache-off run — only the cache_hits/cache_misses counters may differ.

line_data <- function(n = 16L) {
  X <- matrix(seq(-8, 8, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  list(X = X, y = y)
}

test_that("eval_cache defaults off and is accepted as an argument", {
  expect_identical(
    formals(utils::getS3method("symbolic_regression", "default"))$eval_cache,
    FALSE
  )
  d <- line_data()
  res <- symbolic_regression(
    d$X, d$y, unary_ops = character(0),
    population_size = 40L, n_populations = 1L, generations = 10L, seed = 3L,
    eval_cache = TRUE
  )
  expect_s3_class(res, "rsymbolic2")
})

test_that("eval_cache on/off is bit-identical for a fixed seed", {
  d <- line_data()
  # n_threads = 1 pins out a pre-existing, eval_cache-independent flake seen in the
  # Python host process (multi-threaded runs occasionally flip between two
  # trajectories even with eval_cache off; single-threaded runs never do). The
  # multi-thread identity is covered by the standalone C++ gates in a clean process.
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 2L, generations = 40L, seed = 11L,
    n_threads = 1L
  )
  r_off <- do.call(symbolic_regression, common)
  r_on  <- do.call(symbolic_regression, c(common, list(eval_cache = TRUE)))

  expect_identical(r_on$expression, r_off$expression)
  expect_identical(r_on$loss, r_off$loss)
  expect_identical(r_on$pareto_front$loss, r_off$pareto_front$loss)
  expect_identical(r_on$pareto_front$expression, r_off$pareto_front$expression)
  # A cache hit is charged like a real evaluation, so the accounting matches too.
  expect_identical(r_on$n_evals, r_off$n_evals)
  expect_identical(unname(r_on$eval_counts[c("forward", "lm_resid", "lm_jac")]),
                   unname(r_off$eval_counts[c("forward", "lm_resid", "lm_jac")]))
})

test_that("eval_counts carries cache counters: populated on, zero off", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 2L, generations = 40L, seed = 11L,
    n_threads = 1L
  )
  r_off <- do.call(symbolic_regression, common)
  r_on  <- do.call(symbolic_regression, c(common, list(eval_cache = TRUE)))

  expect_length(r_off$eval_counts, 5L)
  expect_named(r_off$eval_counts,
               c("forward", "lm_resid", "lm_jac", "cache_hits", "cache_misses"))
  expect_identical(unname(r_off$eval_counts["cache_hits"]), 0)
  expect_identical(unname(r_off$eval_counts["cache_misses"]), 0)
  # The tiny operator set guarantees duplicate candidates, so the cache hits.
  expect_gt(unname(r_on$eval_counts["cache_hits"]), 0)
  expect_gt(unname(r_on$eval_counts["cache_misses"]), 0)
})
