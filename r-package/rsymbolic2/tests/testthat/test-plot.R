# The plots are drawn by ggplot2, which builds lazily: constructing the object proves
# nothing, so the tests that check layer contents force the build (ggplot_build) -- that
# is what catches a bad aesthetic mapping or a layer fed the wrong data.

# plot() prints as a side effect; route that to a null device so the tests leave no
# Rplots.pdf behind.
draw <- function(...) {
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  plot(...)
}

fit_1d <- function() {
  X <- matrix(seq(-2, 2, length.out = 30), ncol = 1)
  y <- 3 * X[, 1] - 2
  list(
    res = symbolic_regression(
      X, y,
      unary_ops       = character(0),
      population_size = 100L,
      generations     = 20L,
      seed            = 1L
    ),
    X = X, y = y
  )
}

test_that("the default plot is still the Pareto front", {
  skip_if_not_installed("ggplot2")
  f <- fit_1d()
  p <- draw(f$res)
  expect_s3_class(p, "ggplot")
  expect_identical(p$labels$title, "rsymbolic2 Pareto front")
})

test_that("type = 'fit' overlays the fitted curve on a single feature", {
  skip_if_not_installed("ggplot2")
  f <- fit_1d()
  p <- draw(f$res, type = "fit", newdata = f$X, y = f$y)
  expect_s3_class(p, "ggplot")
  expect_identical(p$labels$title, "rsymbolic2 fit")
  # Two layers: the observed scatter and the fitted line.
  expect_length(p$layers, 2L)
  built <- ggplot2::ggplot_build(p)
  expect_identical(nrow(built$data[[1]]), nrow(f$X))
  # The line layer is drawn in x order, so its y values are the predictions of the
  # recommended expression, not the observations.
  expect_equal(
    built$data[[2]]$y,
    unname(predict(f$res, f$X)[order(f$X[, 1])]),
    tolerance = 1e-8
  )
})

test_that("type = 'fit' falls back to predicted vs observed for several features", {
  skip_if_not_installed("ggplot2")
  set.seed(2)
  X <- cbind(runif(40, 1, 3), runif(40, 1, 3))
  y <- X[, 1] + X[, 2]
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 20L,
    seed            = 1L
  )
  p <- draw(res, type = "fit", newdata = X, y = y)
  expect_identical(p$labels$title, "rsymbolic2 fit: predicted vs observed")
  built <- ggplot2::ggplot_build(p)
  # Layer 1 is the dashed y = x reference; layer 2 the points.
  expect_length(p$layers, 2L)
  expect_identical(nrow(built$data[[2]]), nrow(X))
})

test_that("supplying data selects the fit plot; an explicit type wins", {
  skip_if_not_installed("ggplot2")
  f <- fit_1d()
  p <- draw(f$res, newdata = f$X, y = f$y)
  expect_identical(p$labels$title, "rsymbolic2 fit")
  p2 <- draw(f$res, type = "pareto", newdata = f$X, y = f$y)
  expect_identical(p2$labels$title, "rsymbolic2 Pareto front")
})

test_that("the fit plot uses the formula interface's column names", {
  skip_if_not_installed("ggplot2")
  df <- data.frame(t = seq(-2, 2, length.out = 30))
  df$z <- 3 * df$t - 2
  res <- symbolic_regression(
    z ~ t, data = df,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 20L,
    seed            = 1L
  )
  p <- draw(res, type = "fit", newdata = df, y = df$z)
  expect_identical(p$labels$x, "t")
})

# --- Equation tree (docs/48 D6) ------------------------------------------------------
# The layout is a pure function, so it is asserted directly (no ggplot build needed); the
# same expression must lay out identically here, in the Python package and in the web GUI.
REFERENCE_TREE <- "(2.2 - (x0 / 11)) + (7 * cos(x1))"

test_that("the tree layout of the reference expression", {
  n <- expr_tree_layout(REFERENCE_TREE)
  expect_identical(nrow(n), 10L)
  expect_identical(max(n$depth), 3L)
  expect_identical(n$label,
                   c("+", "-", "2.2", DIVIDE_GLYPH, "x0", "11", TIMES_GLYPH, "7", "cos",
                     "x1"))
  expect_true(is.na(n$parent[1]))
  # Kinds drive the node fills: operators, data columns, fitted constants.
  expect_identical(n$kind[n$kind != "operator"],
                   c("constant", "variable", "constant", "constant", "variable"))
  # A unary node sits directly above its only child.
  cos_id <- n$id[n$label == "cos"]
  expect_identical(n$x[n$id == cos_id], n$x[n$parent %in% cos_id])
})

test_that("R's parentheses are stripped, unlike Python's ast and the browser parser", {
  # R keeps `(` as a call node; without the strip this tree would be larger than the
  # other two surfaces' for the same equation.
  expect_identical(nrow(expr_tree_layout("(x0 + 1)")), 3L)
  expect_identical(nrow(expr_tree_layout("(((x0)) + ((1)))")), 3L)
})

test_that("a negated literal is one node, and variables take the fitted names", {
  # "%.6g" prints "-1.3"; R reads unary minus over 1.3, which must not be two nodes.
  expect_identical(expr_tree_layout("(x0 + -1.3)")$label, c("+", "x0", "-1.3"))
  expect_identical(expr_tree_layout("(x0 * x1)", c("t", "flow rate"))$label,
                   c(TIMES_GLYPH, "t", "flow rate"))
  # inf / nan parse as names but are values, not data columns.
  expect_identical(expr_tree_layout("(x0 + inf)")$kind,
                   c("operator", "variable", "constant"))
})

test_that("the tree layout rejects what it cannot draw", {
  expect_error(expr_tree_layout("nope("), "could not parse")
  expect_error(expr_tree_layout("foo(x0)"), "unsupported operator")
})

test_that("type = 'tree' draws one node per element and needs no data", {
  skip_if_not_installed("ggplot2")
  f <- fit_1d()
  p <- draw(f$res, type = "tree", expression = REFERENCE_TREE)
  expect_s3_class(p, "ggplot")
  expect_identical(p$labels$title, "rsymbolic2 equation tree")
  built <- ggplot2::ggplot_build(p)
  expect_identical(nrow(built$data[[1]]), 9L)   # one edge per non-root node
  expect_identical(nrow(built$data[[2]]), 10L)  # one capsule per node
  # Three fills, one per kind.
  expect_setequal(unique(built$data[[2]]$fill), unname(TREE_FILL))
  # The default draws the recommended expression, so it works with no arguments at all.
  expect_s3_class(draw(f$res, type = "tree"), "ggplot")
})

test_that("the tree plot checks variable_names against the fitted feature count", {
  skip_if_not_installed("ggplot2")
  f <- fit_1d()
  expect_error(plot(f$res, type = "tree", variable_names = c("a", "b")),
               "was fitted on")
  p <- draw(f$res, type = "tree", expression = "(x0 + 1)", variable_names = "speed")
  labels <- trimws(ggplot2::ggplot_build(p)$data[[2]]$label)
  expect_true("speed" %in% labels)
})

test_that("the fit plot rejects missing or mismatched data", {
  skip_if_not_installed("ggplot2")
  f <- fit_1d()
  expect_error(plot(f$res, type = "fit"), "needs the data")
  expect_error(plot(f$res, type = "fit", newdata = f$X), "needs the data")
  expect_error(
    plot(f$res, type = "fit", newdata = f$X, y = f$y[1:5]),
    "value\\(s\\) but newdata has"
  )
  expect_error(plot(f$res, type = "nope"), "'arg' should be one of")
})
