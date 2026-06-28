test_that("print.rsymbolic2 returns invisibly and shows key tokens", {
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 15L,
    seed            = 1L
  )
  expect_output(print(res), "rsymbolic2")
  expect_output(print(res), "recommended")
  expect_output(print(res), "Pareto front")
  expect_invisible(print(res))
  expect_identical(withVisible(print(res))$value, res)
})

test_that("summary.rsymbolic2 returns a structured object with a score column", {
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 15L,
    seed            = 2L
  )
  s <- summary(res)
  expect_s3_class(s, "summary.rsymbolic2")
  expect_s3_class(s$pareto, "data.frame")
  expect_named(s$pareto,
               c("complexity", "loss", "score", "recommended", "expression"))
  # Exactly one member is flagged as recommended (when the front is non-empty).
  expect_equal(sum(s$pareto$recommended), 1L)
  # The flagged row's expression equals the reported recommendation.
  rec <- s$pareto$expression[s$pareto$recommended]
  expect_equal(rec, res$recommended)
  # The simplest member has no predecessor, hence NA score.
  expect_true(is.na(s$pareto$score[1L]))
  expect_output(print(s), "Pareto front")
})

test_that("R score reproduces the C++ best_index for model_selection='score'", {
  # The R-side pareto_score() must match the engine's select_best() so summary()
  # reports the same scores the engine used. For model_selection='score' the
  # recommended member is the global argmax of that score over the whole front.
  X <- matrix(seq(-3, 3, length.out = 40), ncol = 1)
  y <- sin(X[, 1]) + X[, 1]^2
  res <- symbolic_regression(
    X, y,
    population_size = 200L,
    generations     = 40L,
    model_selection = "score",
    seed            = 3L
  )
  score <- rsymbolic2:::pareto_score(res$pareto_front$complexity,
                                     res$pareto_front$loss)
  if (sum(!is.na(score)) > 0L) {
    expect_equal(which.max(score), res$best_index)
  } else {
    succeed("degenerate single-member front; nothing to compare")
  }
})

test_that("as.data.frame.rsymbolic2 returns the front with a single recommended row", {
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1]^2 - 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = c("square"),
    population_size = 150L,
    generations     = 20L,
    seed            = 4L
  )
  df <- as.data.frame(res)
  expect_s3_class(df, "data.frame")
  expect_named(df, c("complexity", "loss", "score", "recommended", "expression"))
  expect_equal(nrow(df), nrow(res$pareto_front))
  expect_equal(sum(df$recommended), 1L)
  # score = FALSE drops the column.
  df2 <- as.data.frame(res, score = FALSE)
  expect_false("score" %in% names(df2))
})

test_that("feature names are kept and shown as a legend by print/summary", {
  X <- cbind(seq(-3, 3, length.out = 20), seq(0, 1, length.out = 20))
  colnames(X) <- c("speed", "mass")
  y <- 2 * X[, 1] + X[, 2]
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 15L,
    seed            = 6L
  )
  expect_identical(res$feature_names, c("speed", "mass"))
  # Legend maps the 0-based variables to the column names.
  expect_output(print(res), "x0 = speed, x1 = mass")
  expect_output(print(summary(res)), "x0 = speed, x1 = mass")
  # The fitted expression strings remain 0-based; names are display-only.
  expect_false(any(grepl("speed", res$pareto_front$expression)))
})

test_that("no legend is printed when X carries no column names", {
  X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
  y <- 2 * X[, 1] + 1
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 100L,
    generations     = 15L,
    seed            = 7L
  )
  expect_null(res$feature_names)
  out <- paste(capture.output(print(res)), collapse = "\n")
  expect_false(grepl("variables:", out))
})

test_that("predict defaults to recommended and honours expression=", {
  X <- matrix(seq(-2, 2, length.out = 30), ncol = 1)
  y <- 3 * X[, 1] - 2
  res <- symbolic_regression(
    X, y,
    unary_ops       = character(0),
    population_size = 200L,
    generations     = 30L,
    seed            = 5L
  )
  X_new <- matrix(c(-1, 0, 1), ncol = 1)
  p_rec  <- predict(res, X_new)                              # default: recommended
  p_best <- predict(res, X_new, expression = res$expression) # lowest-loss
  expect_length(p_rec, 3L)
  expect_true(all(is.finite(p_rec)))
  expect_true(all(is.finite(p_best)))
  # Any Pareto-front member can be selected by its expression string.
  p_member <- predict(res, X_new, expression = res$pareto_front$expression[1L])
  expect_length(p_member, 3L)
})
