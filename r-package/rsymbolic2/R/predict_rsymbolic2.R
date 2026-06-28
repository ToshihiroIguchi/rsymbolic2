# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Predict from a symbolic regression fit
#'
#' Evaluate the best expression returned by \code{\link{symbolic_regression}}
#' on new input data.
#'
#' The expression is evaluated using R's standard arithmetic operators and math
#' functions.  \code{pow(x, y)} nodes are rendered as \code{x ^ y} and use R's
#' \code{^} operator, which returns \code{NaN} for negative bases with
#' non-integer exponents.  This differs slightly from the safe-pow semantics
#' used during training (which returns 0 in those cases).  If the training
#' variable domains excluded negative inputs (the Feynman / Nguyen defaults),
#' this distinction does not affect predictions.
#'
#' @param object An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param newdata Numeric matrix (or coercible to one) of new input features.
#'   Must have \code{object$n_features} columns in the same order as the
#'   training matrix \code{X}.
#' @param ... Ignored (present for S3 compatibility).
#'
#' @return Numeric vector of predicted values, one per row of \code{newdata}.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2 * X[, 1]^2 - 1
#' res <- symbolic_regression(X, y, population_size = 200L, generations = 40L,
#'                            seed = 1L)
#' X_new <- matrix(seq(-1, 1, length.out = 5), ncol = 1)
#' predict(res, X_new)
#' }
#'
#' @export
predict.rsymbolic2 <- function(object, newdata, ...) {
    X <- as.matrix(newdata)
    p <- object$n_features
    if (is.null(p)) {
        stop("object$n_features is missing. Re-fit using the current version of symbolic_regression().")
    }
    if (ncol(X) != p) {
        stop(sprintf(
            "newdata has %d column(s) but the model was fitted on %d feature(s).",
            ncol(X), p
        ))
    }

    # Variable names in the expression string are 0-based (x0, x1, x2, ...),
    # matching the C++ to_string() renderer which uses node.index (0-based).
    env <- new.env(parent = baseenv())
    for (j in seq_len(p)) {
        assign(paste0("x", j - 1L), X[, j], envir = env)
    }
    # Operators used in the expression string that are not in R's base:
    #   neg(x)    -> unary minus
    #   square(x) -> x^2
    env$neg    <- function(x) -x
    env$square <- function(x) x * x

    as.numeric(eval(parse(text = object$expression), envir = env))
}
