# Tests for the optional PySR-parity features added on top of the default search:
# model_selection, weights (weighted least squares), max_evals, and early_stop_condition.

line_data <- function(n = 16L) {
  X <- matrix(seq(-8, 8, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  list(X = X, y = y)
}

test_that("model_selection accepts the three modes and 'accuracy' picks the lowest loss", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 4L, generations = 40L, seed = 11L
  )
  res_best  <- do.call(symbolic_regression, c(common, model_selection = "best"))
  res_acc   <- do.call(symbolic_regression, c(common, model_selection = "accuracy"))
  res_score <- do.call(symbolic_regression, c(common, model_selection = "score"))

  for (res in list(res_best, res_acc, res_score)) {
    expect_true(res$best_index >= 1L && res$best_index <= nrow(res$pareto_front))
    expect_equal(res$recommended, res$pareto_front$expression[res$best_index])
  }
  # "accuracy" is the lowest-loss member; the front is sorted ascending in complexity with
  # strictly decreasing loss, so that is the last row.
  expect_equal(res_acc$best_index, nrow(res_acc$pareto_front))
  expect_equal(res_acc$pareto_front$loss[res_acc$best_index], min(res_acc$pareto_front$loss))
})

test_that("model_selection rejects unknown modes", {
  d <- line_data()
  expect_error(
    symbolic_regression(d$X, d$y, model_selection = "nonsense"),
    "should be one of"  # match.arg message
  )
})

test_that("weights of the wrong length or with negative values are rejected", {
  d <- line_data()
  expect_error(symbolic_regression(d$X, d$y, weights = c(1, 2, 3)),
               "same length as y")
  bad <- rep(1, length(d$y)); bad[1] <- -1
  expect_error(symbolic_regression(d$X, d$y, weights = bad),
               "non-negative")
})

test_that("a zero-weighted outlier does not derail the weighted fit", {
  d <- line_data()
  y <- d$y
  w <- rep(1, length(y))
  y[5] <- y[5] + 200    # gross outlier
  w[5] <- 0             # excluded from the weighted loss

  res <- symbolic_regression(
    d$X, y, weights = w, unary_ops = character(0),
    population_size = 80L, n_populations = 4L, generations = 80L, seed = 3L
  )
  # The weighted SSE of the recovered clean line ignores the zero-weighted point.
  expect_true(res$loss < 1e-4)
})

test_that("max_evals is deterministic and bounds the search", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 60L, seed = 7L
  )
  full <- do.call(symbolic_regression, c(common, max_evals = 0))
  cap1 <- do.call(symbolic_regression, c(common, max_evals = 2000))
  cap2 <- do.call(symbolic_regression, c(common, max_evals = 2000))

  # Same seed + budget => identical result (deterministic, thread-count independent).
  expect_equal(cap1$expression, cap2$expression)
  expect_equal(cap1$loss, cap2$loss)
  expect_true(nrow(cap1$pareto_front) >= 1L)
  # With a single island the capped run is a prefix of the full run and the hall of fame
  # is monotone, so more budget is never worse.
  expect_true(full$loss <= cap1$loss + 1e-12)
})

test_that("verbosity does not affect the search result (display-only setting)", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 40L, seed = 5L
  )
  # verbosity only controls the per-epoch stderr log; for a fixed seed the
  # discovered expression, loss, and Pareto front must be identical at 0 and 1.
  res_silent  <- do.call(symbolic_regression, c(common, verbosity = 0L))
  res_verbose <- do.call(symbolic_regression, c(common, verbosity = 1L))

  expect_equal(res_silent$expression, res_verbose$expression)
  expect_equal(res_silent$loss, res_verbose$loss)
  expect_equal(res_silent$pareto_front, res_verbose$pareto_front)
})

test_that("early_stop_condition halts once the loss crosses the threshold", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 60L, seed = 7L
  )
  full  <- do.call(symbolic_regression, c(common, early_stop_condition = 0))
  early <- do.call(symbolic_regression, c(common, early_stop_condition = 0.5))

  expect_true(full$loss < 1e-6)              # control converges far below the threshold
  expect_true(early$loss < 0.5)              # the threshold was crossed before returning
  expect_true(full$loss <= early$loss + 1e-12)  # early run stopped no later than full
})

# PySR batching: with many rows and a small per-iteration subsample, the search still
# recovers the line and — because the hall of fame and final result are recomputed on the
# full dataset — the reported loss reflects all rows, not a lucky batch.
test_that("batching recovers the line and reports a full-data loss", {
  n <- 300L
  X <- matrix(seq(-10, 10, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  res <- symbolic_regression(
    X, y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 80L, seed = 13L,
    batching = TRUE, batch_size = 16L
  )
  expect_true(res$loss < 1e-6)               # exact line => full-data SSE ~ 0
  expect_true(nrow(res$pareto_front) >= 1L)
})

test_that("batching is deterministic for a fixed seed", {
  n <- 200L
  X <- matrix(seq(-10, 10, length.out = n), ncol = 1)
  y <- 2.5 * X[, 1] + 1.7
  common <- list(
    X = X, y = y, unary_ops = character(0),
    population_size = 50L, n_populations = 1L, generations = 40L, seed = 21L,
    batching = TRUE, batch_size = 20L
  )
  r1 <- do.call(symbolic_regression, common)
  r2 <- do.call(symbolic_regression, common)
  expect_equal(r1$expression, r2$expression)
  expect_equal(r1$loss, r2$loss)
})

test_that("batch_size must be a positive integer", {
  d <- line_data()
  expect_error(symbolic_regression(d$X, d$y, batching = TRUE, batch_size = 0L),
               "batch_size must be a positive integer")
})

# PySR warmup_maxsize_by: the size cap ramps from 3 up to max_nodes over the warmup
# fraction. The default 0 leaves the search unchanged; a non-zero value must still
# converge on an easy target and stay deterministic.
test_that("warmup_maxsize_by = 0 (default) leaves the search unchanged", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 40L, seed = 9L
  )
  base <- do.call(symbolic_regression, common)
  off  <- do.call(symbolic_regression, c(common, warmup_maxsize_by = 0))
  expect_equal(base$expression, off$expression)
  expect_equal(base$loss, off$loss)
})

test_that("warmup_maxsize_by recovers the line and is deterministic", {
  d <- line_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 1L, generations = 80L, seed = 9L,
    warmup_maxsize_by = 0.5
  )
  r1 <- do.call(symbolic_regression, common)
  r2 <- do.call(symbolic_regression, common)
  expect_equal(r1$expression, r2$expression)
  expect_equal(r1$loss, r2$loss)
  expect_true(r1$loss < 1e-6)
  expect_true(all(r1$pareto_front$complexity <= 30L))  # never exceeds max_nodes
})

test_that("warmup_maxsize_by rejects negative or non-finite values", {
  d <- line_data()
  expect_error(symbolic_regression(d$X, d$y, warmup_maxsize_by = -0.1),
               "warmup_maxsize_by")
  expect_error(symbolic_regression(d$X, d$y, warmup_maxsize_by = NA_real_),
               "warmup_maxsize_by")
})
