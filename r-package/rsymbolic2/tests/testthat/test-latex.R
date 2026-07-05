# to_latex(): accessor + variable-name substitution on a handcrafted object (the
# serializer itself is covered by the standalone C++ test test_to_latex.cpp).

make_latex_fixture <- function() {
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
      expression = c("1", "x0"),
      latex      = c("1", "x_{1} + x_{10} \\cdot x_{0}"),
      stringsAsFactors = FALSE
    ),
    n_features    = 11L,
    feature_names = NULL
  ), class = "rsymbolic2")
}

test_that("to_latex defaults to the recommended member", {
  res <- make_latex_fixture()
  expect_identical(to_latex(res), "x_{1} + x_{10} \\cdot x_{0}")
  expect_identical(to_latex(res, index = 1L), "1")
  # A vector index returns the members in order.
  expect_identical(to_latex(res, index = c(2L, 1L)),
                   c("x_{1} + x_{10} \\cdot x_{0}", "1"))
})

test_that("to_latex substitutes variable names with underscore escaping", {
  res <- make_latex_fixture()
  nms <- paste0("v", 0:10)
  nms[11] <- "big_name"  # x_{10}; underscore must be escaped for LaTeX
  expect_identical(to_latex(res, variable_names = nms),
                   "v1 + big\\_name \\cdot v0")
  # feature_names are the default substitution source.
  res$feature_names <- nms
  expect_identical(to_latex(res), "v1 + big\\_name \\cdot v0")
  # character(0) forces the raw x_{i} form.
  expect_identical(to_latex(res, variable_names = character(0)),
                   "x_{1} + x_{10} \\cdot x_{0}")
})

test_that("to_latex validates index and variable_names", {
  res <- make_latex_fixture()
  expect_error(to_latex(res, index = 0L), "index")
  expect_error(to_latex(res, index = 3L), "index")
  expect_error(to_latex(res, variable_names = "only_one"), "variable_names")
})

test_that("to_latex errors clearly on objects fitted before the latex column", {
  res <- make_latex_fixture()
  res$pareto_front$latex <- NULL
  expect_error(to_latex(res), "latex")
})

test_that("to_latex works on a real fit", {
  X <- matrix(seq(-2, 2, length.out = 25), ncol = 1)
  y <- 2 * X[, 1]
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 30L,
    seed            = 7L
  )
  s <- to_latex(res)
  expect_type(s, "character")
  expect_true(nchar(s) > 0)
  expect_match(s, "x_\\{0\\}", fixed = FALSE)
})
