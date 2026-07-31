# to_sympy(): accessor + variable-name substitution on a handcrafted object (the
# serializer itself is covered by the standalone C++ test test_to_sympy.cpp, and its
# semantics — that SymPy parses it to the same function — by
# python/tests/test_sympy_export.py, which round-trips a real front against predict()).

make_sympy_fixture <- function() {
  structure(list(
    expression  = "x0",
    loss        = 0.1,
    complexity  = 1L,
    recommended = "x0",
    best_index  = 2L,
    pareto_front = data.frame(
      complexity = c(1L, 3L),
      loss       = c(1.0, 0.1),
      score      = c(0.0, 1.15),
      expression = c("1", "(neg(x0) + square(x10))"),
      sympy      = c("1", "-x0 + x10**2"),
      stringsAsFactors = FALSE
    ),
    n_features    = 11L,
    feature_names = NULL
  ), class = "rsymbolic2")
}

test_that("to_sympy defaults to the recommended member", {
  res <- make_sympy_fixture()
  expect_identical(to_sympy(res), "-x0 + x10**2")
  expect_identical(to_sympy(res, index = 1L), "1")
  expect_identical(to_sympy(res, index = c(2L, 1L)), c("-x0 + x10**2", "1"))
})

test_that("to_sympy keeps x0/x1 unless names are asked for", {
  res <- make_sympy_fixture()
  # Unlike to_latex(), feature_names are NOT applied by default: a column name is free
  # text, and "flow rate" would produce a string SymPy cannot parse.
  res$feature_names <- c("flow rate", paste0("v", 1:10))
  expect_identical(to_sympy(res), "-x0 + x10**2")
})

test_that("to_sympy substitutes names without re-substituting", {
  res <- make_sympy_fixture()
  nms <- paste0("v", 0:10)
  expect_identical(to_sympy(res, variable_names = nms), "-v0 + v10**2")
  # x10 must not be hit by the x1 rule (word boundaries), and a rename whose targets are
  # themselves tokens must not chain: swapping x0 and x10 gives exactly the swap.
  swap <- nms
  swap[1] <- "x10"
  swap[11] <- "x0"
  expect_identical(to_sympy(res, variable_names = swap), "-x10 + x0**2")
})

test_that("to_sympy rejects names that are not Python identifiers", {
  res <- make_sympy_fixture()
  nms <- paste0("v", 0:10)
  nms[1] <- "flow rate"
  expect_error(to_sympy(res, variable_names = nms), "Python identifiers")
  nms[1] <- "2x"
  expect_error(to_sympy(res, variable_names = nms), "Python identifiers")
  expect_error(to_sympy(res, variable_names = "only_one"), "variable_names")
  expect_error(to_sympy(res, index = 0L), "index")
  expect_error(to_sympy(res, index = 3L), "index")
})

test_that("to_sympy errors clearly on objects fitted before the sympy column", {
  res <- make_sympy_fixture()
  res$pareto_front$sympy <- NULL
  expect_error(to_sympy(res), "sympy")
})

test_that("a real fit exports no square()/inv()/neg()/^ in any sympy rendering", {
  X <- matrix(seq(0.5, 4, length.out = 30), ncol = 1)
  y <- 2 * X[, 1]^2 + 1 / X[, 1]
  res <- symbolic_regression(
    X, y,
    unary_ops       = c("square", "inv", "neg"),
    binary_ops      = c("add", "sub", "mul", "div", "pow"),
    population_size = 40L,
    n_populations   = 6L,
    generations     = 80L,
    seed            = 3L
  )
  df <- res$pareto_front
  expect_true(all(nzchar(df$sympy)))
  expect_true(all(nzchar(df$sympy_simplified)))
  not_python <- "\\b(square|inv|neg)\\(|\\^"
  expect_false(any(grepl(not_python, df$sympy)))
  expect_false(any(grepl(not_python, df$sympy_simplified)))
  # The point of the export: the frozen `expression` strings DO carry those tokens, so
  # the two renderings cannot be the same string.
  expect_true(any(grepl(not_python, df$expression)))
  expect_type(to_sympy(res), "character")
})

test_that("a large unbatched fit emits an advisory message and nothing else changes", {
  X <- matrix(seq(0, 1, length.out = 10001), ncol = 1)
  y <- 2 * X[, 1]
  args <- list(X = X, y = y, generations = 1L, n_populations = 1L,
               population_size = 4L, seed = 5L)
  expect_message(loud <- do.call(symbolic_regression, args),
                 "Every candidate evaluation is O\\(rows\\)")
  # Advisory only: the message does not touch the search.
  quiet <- suppressMessages(do.call(symbolic_regression, args))
  expect_identical(loud$expression, quiet$expression)
  expect_identical(loud$loss, quiet$loss)
  # Batching is the thing it points at, so turning it on silences it.
  expect_no_message(do.call(symbolic_regression, c(args, list(batching = TRUE))))
  # Below the threshold nothing is emitted.
  small <- args
  small$X <- X[1:100, , drop = FALSE]
  small$y <- y[1:100]
  expect_no_message(do.call(symbolic_regression, small))
})
