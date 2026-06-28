# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Predict from a symbolic regression fit
#'
#' Evaluate a fitted expression returned by \code{\link{symbolic_regression}}
#' on new input data. By default the recommended (Pareto "best") expression is
#' used; see the \code{which} argument to evaluate the lowest-loss expression or
#' any other member of the Pareto front.
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
#' @param newdata New input features. For a model fitted via the matrix
#'   interface, a numeric matrix (or coercible to one) with \code{object$n_features}
#'   columns in the same order as the training matrix \code{X}. For a model fitted
#'   via the \code{\link{symbolic_regression}} formula interface, a
#'   \code{data.frame} containing the predictor columns named in the formula; they
#'   are selected by name in the fitted order, so column order in \code{newdata}
#'   does not matter.
#' @param expression Which fitted expression to evaluate. \code{NULL} (default)
#'   uses \code{object$recommended} (the Pareto "best" accuracy/complexity
#'   trade-off chosen by \code{model_selection}). Otherwise pass an expression
#'   string, e.g. \code{object$expression} for the lowest-loss model or any row of
#'   \code{object$pareto_front$expression} for a specific Pareto member. This
#'   mirrors the Python interface's \code{predict(..., expression=)} and PySR's
#'   \code{.predict()}, which evaluate the selected model by default.
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
#' predict(res, X_new)                                # recommended (default)
#' predict(res, X_new, expression = res$expression)  # lowest-loss expression
#' }
#'
#' @export
predict.rsymbolic2 <- function(object, newdata, expression = NULL, ...) {
    # A formula-fitted model carries `terms`: use them to pull the predictor
    # columns out of a data.frame newdata by name, in the fitted order (so the
    # caller's column order is irrelevant). Otherwise treat newdata as a matrix in
    # column order, preserving the matrix-interface behaviour.
    if (is.data.frame(newdata) && !is.null(object$terms)) {
        X <- as.matrix(stats::model.frame(object$terms, newdata))
    } else {
        X <- as.matrix(newdata)
    }
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

    # NULL evaluates the recommended (Pareto "best") model, matching PySR / the
    # Python interface; any other value is a literal expression string, so callers
    # can evaluate the lowest-loss model or any other Pareto-front member.
    expr <- if (is.null(expression)) object$recommended else expression

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

    as.numeric(eval(parse(text = expr), envir = env))
}
