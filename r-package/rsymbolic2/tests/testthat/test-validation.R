test_that("a factor y is rejected rather than silently using level codes", {
  X <- matrix(seq(-3, 3, length.out = 6), ncol = 1)
  # as.numeric() on this factor gives c(1, 2, 3, 1, 2, 3), not the numbers shown.
  y <- factor(c(100, 20, 3, 100, 20, 3))
  expect_error(symbolic_regression(X, y), "level codes")
})

test_that("a factor response is rejected by the formula method too", {
  d <- data.frame(a = seq(-3, 3, length.out = 6),
                  b = factor(c(100, 20, 3, 100, 20, 3)))
  expect_error(symbolic_regression(b ~ a, data = d), "level codes")
})

test_that("non-finite X or y is rejected", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- 2 * X[, 1] + 1

  Xbad <- X; Xbad[3, 1] <- NA
  expect_error(symbolic_regression(Xbad, y), "X must not contain")

  Xbad2 <- X; Xbad2[3, 1] <- Inf
  expect_error(symbolic_regression(Xbad2, y), "X must not contain")

  ybad <- y; ybad[4] <- NaN
  expect_error(symbolic_regression(X, ybad), "y must not contain")

  ybad2 <- y; ybad2[4] <- -Inf
  expect_error(symbolic_regression(X, ybad2), "y must not contain")
})

test_that("non-positive counts are rejected instead of aborting the session", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- 2 * X[, 1] + 1
  # population_size = -1 used to wrap to a huge size_t and terminate the R session
  # from inside the OpenMP region.
  expect_error(symbolic_regression(X, y, population_size = -1L), "population_size")
  expect_error(symbolic_regression(X, y, population_size = 0L),  "population_size")
  expect_error(symbolic_regression(X, y, generations = 0L),      "generations")
  expect_error(symbolic_regression(X, y, generations = -5L),     "generations")
  expect_error(symbolic_regression(X, y, tournament_size = 0L),  "tournament_size")
  expect_error(symbolic_regression(X, y, max_nodes = 0L),        "max_nodes")
  expect_error(symbolic_regression(X, y, max_nodes = -3L),       "max_nodes")
  expect_error(symbolic_regression(X, y, max_depth = 0L),        "max_depth")
  expect_error(symbolic_regression(X, y, n_populations = 0L),    "n_populations")
  expect_error(symbolic_regression(X, y, n_populations = -2L),   "n_populations")
})

test_that("mismatched nrow(X) and length(y) is rejected", {
  X <- matrix(1:10, ncol = 1)
  y <- 1:5
  expect_error(symbolic_regression(X, y), "nrow")
})

test_that("empty binary_ops is rejected", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- X[, 1]
  expect_error(
    symbolic_regression(X, y, binary_ops = character(0)),
    "binary_ops"
  )
})

# --- docs/80: unusable data is refused, degenerate data warns ------------------------

test_that("X with no columns is rejected", {
    # Silently returned the constant expression "1": with no features there is no function
    # of X to discover, so any result is arbitrary.
    y <- seq_len(10)
    expect_error(symbolic_regression(matrix(numeric(0), nrow = 10, ncol = 0), y),
                 "at least one column")
})

test_that("all-zero weights are rejected", {
    X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
    y <- 2 * X[, 1] + 1
    # Non-negative and finite, but the weighted SSE is then identically 0, so every
    # candidate ties at a perfect loss and the winner is whatever the tournament held.
    expect_error(symbolic_regression(X, y, weights = rep(0, 10)), "all zero")
    # Only the sum has to be positive; the rest may be zero.
    expect_error(
        symbolic_regression(X, y, weights = c(1, 1, rep(0, 8)), generations = 5L,
                            n_populations = 2L, population_size = 6L, verbosity = 0L),
        NA)
    # ...and the degeneracy check reads the same weights the core does, so weighting all
    # but one point out of existence is a zero-variance target and says so.
    expect_warning(
        symbolic_regression(X, y, weights = c(1, rep(0, 9)), generations = 5L,
                            n_populations = 2L, population_size = 6L, verbosity = 0L),
        "constant (zero variance)", fixed = TRUE)
})

test_that("a logical X is accepted, matching Python and the mixed-column case", {
    # is.numeric() is FALSE for a logical matrix, so the character guard rejected 0/1
    # indicator columns too -- but only when EVERY column was logical, since as.matrix()
    # on a mixed data frame promotes them.
    Xl <- matrix(rep(c(TRUE, FALSE), 5), ncol = 1)
    y  <- as.numeric(rep(c(2, 5), 5))
    res <- symbolic_regression(Xl, y, generations = 5L, n_populations = 2L,
                               population_size = 6L, verbosity = 0L)
    expect_s3_class(res, "rsymbolic2")
    expect_equal(res$n_features, 1L)
})

test_that("a constant target warns but still runs", {
    X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
    expect_warning(
        res <- symbolic_regression(X, rep(5, 10), generations = 5L, n_populations = 2L,
                                   population_size = 6L, verbosity = 0L),
        "constant (zero variance)", fixed = TRUE)
    expect_s3_class(res, "rsymbolic2")
    # The warning explains exactly the case that already made R^2 undefined.
    expect_equal(res$sst, 0)
    expect_true(is.na(summary(res)$r_squared))
})

test_that("a constant feature column warns and is named 0-based", {
    X <- cbind(a = rep(2, 10), b = seq(-3, 3, length.out = 10))
    y <- X[, 2]
    expect_warning(
        symbolic_regression(X, y, generations = 5L, n_populations = 2L,
                            population_size = 6L, verbosity = 0L),
        "x0 (a) is constant", fixed = TRUE)
})

test_that("a y scale that overflows the sum of squares warns", {
    X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
    expect_warning(
        symbolic_regression(X, (2 * X[, 1] + 1) * 1e200, generations = 5L,
                            n_populations = 2L, population_size = 6L, verbosity = 0L),
        "overflows the sum-of-squares loss")
})

test_that("ordinary data raises none of the degeneracy warnings", {
    X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
    expect_warning(
        symbolic_regression(X, 2 * X[, 1] + 1, generations = 5L, n_populations = 2L,
                            population_size = 6L, verbosity = 0L),
        NA)
})
