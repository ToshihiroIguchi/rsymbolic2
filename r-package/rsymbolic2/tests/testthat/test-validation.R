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
