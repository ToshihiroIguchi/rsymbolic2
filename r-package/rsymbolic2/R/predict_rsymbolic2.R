# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Predict from a symbolic regression fit
#'
#' Evaluate a fitted expression returned by \code{\link{symbolic_regression}}
#' on new input data. By default the recommended (Pareto "best") expression is
#' used; see the \code{expression} argument to evaluate the lowest-loss
#' expression or any other member of the Pareto front. Called without
#' \code{newdata} it returns the fitted values, as \code{predict()} does for
#' every other R model object.
#'
#' The expression is evaluated using R's standard arithmetic operators and math
#' functions.  These agree with the guarded operators used during training on the
#' ordinary out-of-domain cases: \code{sqrt} of a negative number and \code{x ^ y}
#' for a negative base under a fractional exponent are \code{NaN} both here and in
#' the engine (see \code{docs/69}).  They part only at the edges the engine guards
#' explicitly, where R follows IEEE: \code{0 ^ -1} and \code{(-Inf) ^ 0.5} are
#' \code{Inf} in R and \code{NaN} in the engine, and \code{log(0)} is \code{-Inf}
#' in R and \code{NaN} in the engine (the \code{safe_log} guard, \code{docs/77}).
#' It matters only if the prediction inputs reach them.
#'
#' An expression that uses no data column -- a bare constant, which the simplest
#' Pareto-front member usually is -- yields the same value for every row; it is
#' recycled to \code{nrow(newdata)} so the returned length is always one per row.
#'
#' @param object An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param newdata New input features. Omitted, the training data stored on the
#'   object is used and the result is the vector of fitted values, exactly as
#'   \code{\link[stats]{fitted}} returns (this needs \code{keep_data = TRUE}, the
#'   default, at fit time). For a model fitted via the matrix
#'   interface, a numeric matrix (or coercible to one) with \code{object$n_features}
#'   columns in the same order as the training matrix \code{X}. For a model fitted
#'   via the \code{\link{symbolic_regression}} formula interface, a
#'   \code{data.frame} containing the predictor columns named in the formula; they
#'   are selected by name in the fitted order, so column order in \code{newdata}
#'   does not matter. A formula-fitted model \strong{requires} a \code{data.frame}:
#'   a matrix carries no names to match and would be read positionally, which for a
#'   fit made by name is a silently wrong answer rather than an error.
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
#' @seealso \code{\link{fitted.rsymbolic2}} and
#'   \code{\link{residuals.rsymbolic2}} for the training data,
#'   \code{\link{plot.rsymbolic2}} (\code{type = "fit"}) to see the same
#'   predictions against the observations, and \code{\link{symbolic_regression}}
#'   for the fit itself.
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
    # No newdata: predict on the data the model was fitted to, which is what
    # predict(fit) means for every other R regression object. Routed through
    # fitted() so the two cannot disagree.
    if (missing(newdata) || is.null(newdata))
        return(stats::fitted(object, expression = expression))
    eval_on_matrix(design_matrix(object, newdata), object, expression)
}

#' Fitted values from a symbolic regression fit
#'
#' Evaluate a fitted expression on the training data stored on the object, which
#' requires the fit to have been made with \code{keep_data = TRUE} (the default).
#'
#' @param object An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param expression Which fitted expression to evaluate. \code{NULL} (default)
#'   uses \code{object$recommended}; otherwise an expression string, e.g.
#'   \code{object$expression} for the lowest-loss model or any row of
#'   \code{object$pareto_front$expression}.
#' @param ... Ignored (present for S3 compatibility).
#'
#' @return Numeric vector of fitted values, one per training observation.
#'
#' @seealso \code{\link{residuals.rsymbolic2}} for the same values as deviations
#'   from \code{y}, and \code{\link{predict.rsymbolic2}} to evaluate the fit on
#'   new data.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2 * X[, 1] + 1
#' res <- symbolic_regression(X, y, population_size = 200L, generations = 40L,
#'                            seed = 1L)
#' head(fitted(res))
#' head(residuals(res))
#' }
#'
#' @export
fitted.rsymbolic2 <- function(object, expression = NULL, ...) {
    eval_on_matrix(training_matrix(object, "fitted"), object, expression)
}

#' Residuals from a symbolic regression fit
#'
#' The training response minus the fitted values, which requires the fit to have
#' been made with \code{keep_data = TRUE} (the default).
#'
#' @inheritParams fitted.rsymbolic2
#'
#' @return Numeric vector of residuals, one per training observation.
#'
#' @seealso \code{\link{fitted.rsymbolic2}} for the fitted values themselves,
#'   and \code{\link{predict.rsymbolic2}} to evaluate the fit on new data.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2 * X[, 1] + 1
#' res <- symbolic_regression(X, y, population_size = 200L, generations = 40L,
#'                            seed = 1L)
#' summary(residuals(res))
#' }
#'
#' @export
residuals.rsymbolic2 <- function(object, expression = NULL, ...) {
    y <- object$y
    if (is.null(y))
        stop("this model was fitted with keep_data = FALSE, so it carries no training ",
             "response to take residuals against. Re-fit with keep_data = TRUE, or ",
             "compute y - predict(object, newdata) yourself.", call. = FALSE)
    as.numeric(y) - eval_on_matrix(training_matrix(object, "residuals"), object, expression)
}

# The stored training features, or an error that says how to get them. Shared by
# fitted(), residuals(), predict() with no newdata, and plot(type = "fit") so the
# keep_data contract is stated once.
training_matrix <- function(object, what) {
    X <- object$X
    if (is.null(X))
        stop("this model was fitted with keep_data = FALSE, so it carries no training ",
             "data for ", what, "() to evaluate. Re-fit with keep_data = TRUE (the ",
             "default), or call predict(object, newdata = <features>) instead.",
             call. = FALSE)
    as.matrix(X)
}

# Evaluate one fitted expression on an already-built numeric design matrix.
#
# Deliberately takes a matrix rather than user `newdata`: design_matrix() requires a
# data.frame for formula-fitted models (docs/61 C2), which is right for predict() but
# wrong for fitted(), where the stored `object$X` is a matrix and no name matching is
# needed or possible. Keeping the evaluation here means predict() and fitted() cannot
# drift apart.
eval_on_matrix <- function(X, object, expression) {
    p <- ncol(X)

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
    #   erf(x)    -> 2 * pnorm(x * sqrt(2)) - 1
    #   neg(x)    -> unary minus     |  no longer emitted: the core renders these
    #   square(x) -> x^2             |  three as `(-x)`, `(x ^ 2)` and `(1 / x)`
    #   inv(x)    -> 1/x             |  (docs/71). Kept bound so an expression string
    #                                   saved by an earlier version still evaluates.
    # R has sinh() and cosh() in base, but no erf(); the pnorm identity is exact to
    # double precision (pnorm uses Cody's algorithm), so predict() reproduces the
    # core's std::erf rather than approximating it.
    env$neg    <- function(x) -x
    env$square <- function(x) x * x
    env$inv    <- function(x) 1 / x
    env$erf    <- function(x) 2 * stats::pnorm(x * sqrt(2)) - 1
    # The core renders constants with "%.6g", which emits the bare tokens `inf` and
    # `nan` for a non-finite one. R's parser reads those as names, not numbers, so they
    # need binding here -- the package's tree renderer (tree_plot.R) and the Python
    # package already recognise them.
    env$inf    <- Inf
    env$nan    <- NaN

    out <- as.numeric(eval(parse(text = expr), envir = env))
    # An expression that uses no data column (a bare constant -- the simplest Pareto
    # member usually is one) evaluates to a single value. The documented contract is one
    # prediction per row, so recycle it; any other length means the string was not one
    # this engine produced.
    if (length(out) == 1L) out <- rep(out, nrow(X))
    if (length(out) != nrow(X))
        stop(sprintf("the expression evaluated to %d value(s) for %d row(s) of newdata.",
                     length(out), nrow(X)))
    out
}

# Turn user-supplied `newdata` into the numeric design matrix a fitted expression
# expects, checked against the fitted feature count. Shared by predict() and
# plot(type = "fit") so the formula/matrix rule is stated once.
#
# A formula-fitted model carries `terms`: the predictor columns are pulled out of a
# data.frame newdata by name, in the fitted order, so the caller's column order is
# irrelevant. That guarantee is only available from a named object, so such a model
# REQUIRES a data.frame -- a matrix would silently be matched positionally, and a user
# who fitted by name has no reason to expect their column order to matter. A
# matrix-fitted model has no names to match and stays positional.
design_matrix <- function(object, newdata) {
    if (!is.null(object$terms)) {
        if (!is.data.frame(newdata)) {
            stop("this model was fitted with the formula interface, so newdata must be ",
                 "a data.frame holding the predictor column(s): ",
                 paste(all.vars(object$terms), collapse = ", "),
                 ". Columns are matched by name, which a matrix cannot provide.")
        }
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
    # as.matrix() on a data.frame carrying a factor or character column yields a
    # character matrix; without this the failure surfaces from inside eval() as
    # "non-numeric argument to binary operator", naming nothing useful.
    if (!is.numeric(X)) stop("newdata must be numeric; convert factor/character ",
                             "columns to numeric ones first.")
    X
}
