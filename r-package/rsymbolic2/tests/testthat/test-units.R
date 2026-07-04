# Tests for the opt-in dimensional-analysis feature (PySR X_units / y_units /
# dimensional_constraint_penalty / dimensionless_constants_only; docs/46).

# Force = mass * acceleration: a dimensionally consistent 2-feature target.
# x0 = mass (kg), x1 = acceleration (m/s^2), y = force (N = kg*m/s^2).
force_data <- function(n = 24L) {
  set.seed(7)
  m <- runif(n, 1, 5)
  a <- runif(n, 1, 5)
  list(X = cbind(m, a), y = m * a)
}

test_that("X_units of the wrong length is rejected", {
  d <- force_data()
  expect_error(
    symbolic_regression(d$X, d$y, X_units = c("kg")),  # need 2
    "X_units must have length"
  )
})

test_that("an invalid unit string is rejected with the offending token", {
  d <- force_data()
  expect_error(
    symbolic_regression(d$X, d$y, X_units = c("kg", "flibble")),
    "flibble"
  )
})

test_that("y_units must be scalar and requires X_units", {
  d <- force_data()
  expect_error(
    symbolic_regression(d$X, d$y, X_units = c("kg", "m/s^2"), y_units = c("N", "N")),
    "single unit string"
  )
  expect_error(
    symbolic_regression(d$X, d$y, y_units = "N"),  # no X_units
    "requires X_units"
  )
})

test_that("dimensional_constraint_penalty must be a non-negative finite scalar", {
  d <- force_data()
  expect_error(
    symbolic_regression(d$X, d$y, X_units = c("kg", "m/s^2"),
                        dimensional_constraint_penalty = -5),
    "dimensional_constraint_penalty"
  )
})

test_that("a units-enabled run completes and returns a valid model", {
  d <- force_data()
  res <- symbolic_regression(
    d$X, d$y, unary_ops = character(0),
    X_units = c("kg", "m/s^2"), y_units = "N",
    population_size = 60L, n_populations = 4L, generations = 150L, seed = 1L
  )
  expect_s3_class(res, "rsymbolic2")
  expect_true(is.finite(res$loss))
  expect_true(nrow(res$pareto_front) >= 1L)
})

test_that("mismatched y_units penalises the fit relative to correct units", {
  d <- force_data()
  # dimensionless_constants_only closes the "multiply by a free (wildcard) constant" escape:
  # otherwise a trailing *c makes the root a wildcard that satisfies any y_units (faithful to
  # SR.jl). With dimensionless constants, x0*x1 has a fixed output dimension (N), so a
  # mismatched y_units="s" is a genuine, deterministic violation.
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    X_units = c("kg", "m/s^2"), dimensionless_constants_only = TRUE,
    population_size = 60L, n_populations = 4L, generations = 200L, seed = 1L
  )
  # Correct output units (N): the true model x0*x1 is consistent and recovers cleanly.
  res_ok  <- do.call(symbolic_regression, c(common, y_units = "N"))
  # Wrong output units (s): every well-fitting model (dimension N) violates and is
  # penalised, so the recommended model's loss is far higher.
  res_bad <- do.call(symbolic_regression, c(common, y_units = "s"))

  expect_true(res_ok$loss < 1e-6)        # x0*x1 recovered under correct units
  expect_true(res_bad$loss > res_ok$loss)
})

test_that("correct units do not degrade recovery of a consistent target", {
  d <- force_data()
  common <- list(
    X = d$X, y = d$y, unary_ops = character(0),
    population_size = 60L, n_populations = 4L, generations = 150L, seed = 3L
  )
  res_off <- do.call(symbolic_regression, common)  # units off (default)
  res_on  <- do.call(symbolic_regression,
                     c(common, X_units = list(c("kg", "m/s^2")), y_units = "N"))
  expect_true(res_off$loss < 1e-6)
  expect_true(res_on$loss < 1e-6)
})
