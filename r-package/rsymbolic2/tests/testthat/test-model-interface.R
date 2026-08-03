# The R regression-object vocabulary: fitted(), residuals(), predict() with no
# newdata, and plot() against the stored training data (docs/81 P2), plus the
# refusal messages that now carry a remedy (docs/81 P3).

fit_small <- function(...) {
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1] + 1
  list(X = X, y = y,
       res = symbolic_regression(X, y, unary_ops = character(0),
                                 population_size = 100L, generations = 30L,
                                 seed = 1L, verbosity = 0L, ...))
}

test_that("keep_data = TRUE (the default) stores the training data", {
  f <- fit_small()
  expect_equal(f$res$X, f$X)
  expect_equal(f$res$y, f$y)
})

test_that("fitted() returns one value per training observation", {
  f <- fit_small()
  fv <- fitted(f$res)
  expect_type(fv, "double")
  expect_length(fv, nrow(f$X))
  expect_true(all(is.finite(fv)))
})

test_that("residuals() is exactly y minus fitted()", {
  f <- fit_small()
  expect_equal(residuals(f$res), f$y - fitted(f$res))
})

test_that("predict() with no newdata returns the fitted values", {
  f <- fit_small()
  expect_equal(predict(f$res), fitted(f$res))
  # ... and agrees with passing the training data back in explicitly.
  expect_equal(predict(f$res, f$X), fitted(f$res))
})

test_that("fitted() and residuals() accept an explicit expression", {
  f <- fit_small()
  simplest <- f$res$pareto_front$expression[1]
  expect_length(fitted(f$res, expression = simplest), nrow(f$X))
  expect_equal(residuals(f$res, expression = simplest),
               f$y - fitted(f$res, expression = simplest))
})

test_that("a formula fit supports fitted()/residuals() too", {
  set.seed(1)
  d <- data.frame(a = seq(-3, 3, length.out = 20), b = rnorm(20))
  d$yy <- 2 * d$a + 1
  res <- symbolic_regression(yy ~ a + b, data = d, unary_ops = character(0),
                             population_size = 100L, generations = 30L,
                             seed = 1L, verbosity = 0L)
  # fitted() must not route through design_matrix(), which demands a data.frame
  # for formula fits -- the stored X is a matrix (docs/81 P2).
  expect_length(fitted(res), nrow(d))
  expect_equal(residuals(res), d$yy - fitted(res))
  # Name-based prediction still holds: a reordered data.frame gives the same answer.
  expect_equal(predict(res, d[, c("b", "a")]), fitted(res))
})

test_that("keep_data = FALSE omits the data and says what to do instead", {
  f <- fit_small(keep_data = FALSE)
  expect_null(f$res$X)
  expect_null(f$res$y)
  expect_error(fitted(f$res), "keep_data = TRUE")
  expect_error(residuals(f$res), "keep_data = TRUE")
  expect_error(predict(f$res), "keep_data = TRUE")
  # An explicit newdata still works: only the stored-data entry points are affected.
  expect_length(predict(f$res, f$X), nrow(f$X))
})

test_that('plot(type = "fit") uses the stored data, and refuses one half only', {
  skip_if_not_installed("ggplot2")
  f <- fit_small()
  expect_s3_class(plot(f$res, type = "fit"), "ggplot")
  expect_s3_class(plot(f$res, type = "fit", newdata = f$X, y = f$y), "ggplot")
  expect_error(plot(f$res, type = "fit", newdata = f$X), "both halves")
  expect_error(plot(fit_small(keep_data = FALSE)$res, type = "fit"),
               "keep_data = FALSE")
})

test_that("a non-numeric data frame column is named in the error", {
  d <- data.frame(a = c(1, 2, 3, 4, 5), f = factor(c("x", "y", "x", "y", "x")))
  y <- c(1, 2, 3, 4, 5)
  # The blanket "X must be numeric" could not say which column was at fault,
  # because as.matrix() had already made every column character (docs/81 P3).
  expect_error(symbolic_regression(d, y, generations = 5L, verbosity = 0L),
               "non-numeric column\\(s\\): f")
  expect_error(symbolic_regression(d, y, generations = 5L, verbosity = 0L),
               "model.matrix")
})

test_that("non-finite data errors say how to drop the rows", {
  X <- matrix(seq(-3, 3, length.out = 10), ncol = 1)
  y <- 2 * X[, 1] + 1
  Xbad <- X; Xbad[3, 1] <- NA
  expect_error(symbolic_regression(Xbad, y, verbosity = 0L), "complete.cases")
  ybad <- y; ybad[4] <- NaN
  expect_error(symbolic_regression(X, ybad, verbosity = 0L), "is.finite\\(y\\)")
})
