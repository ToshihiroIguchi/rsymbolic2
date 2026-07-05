# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Render fitted expressions as LaTeX
#'
#' Generic for extracting a LaTeX rendering of a fitted model. See
#' \code{\link{to_latex.rsymbolic2}} for the method on symbolic regression fits.
#'
#' @param x A fitted model object.
#' @param ... Passed to methods.
#'
#' @return A character vector of LaTeX math strings (no surrounding \code{$}).
#'
#' @export
to_latex <- function(x, ...) UseMethod("to_latex")

#' @describeIn to_latex LaTeX for one or more Pareto-front members of an
#'   \code{\link{symbolic_regression}} fit. Rendered by the C++ core with
#'   minimal parentheses (\code{\\frac} for division, \code{\\cdot},
#'   \code{\\sqrt}, ...). Display-only: \code{\link{predict.rsymbolic2}} keeps
#'   using the plain \code{expression} strings.
#'
#' @param index Integer vector of rows of \code{x$pareto_front} to render.
#'   Defaults to \code{x$best_index}, the member chosen by
#'   \code{model_selection}.
#' @param variable_names Character vector of names substituted for the
#'   \code{x_{i}} tokens (underscores are escaped for LaTeX). Defaults to
#'   \code{x$feature_names} when set; pass an explicit vector to override, or
#'   \code{character(0)} to force the \code{x_{i}} form.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2 * X[, 1]^2 - 1
#' res <- symbolic_regression(X, y, unary_ops = c("square"),
#'                            population_size = 200L, generations = 40L,
#'                            seed = 1L)
#' to_latex(res)                                  # recommended member
#' to_latex(res, index = seq_len(nrow(res$pareto_front)))  # whole front
#' }
#'
#' @export
to_latex.rsymbolic2 <- function(x, index = NULL, variable_names = NULL, ...) {
    df <- x$pareto_front
    if (is.null(df$latex))
        stop("this fit has no 'latex' column; re-fit with the current ",
             "rsymbolic2 version to use to_latex().")
    if (is.null(index)) index <- x$best_index
    if (length(index) == 0L || anyNA(index) ||
        any(index < 1L | index > nrow(df)))
        stop("index must select rows of x$pareto_front (1..", nrow(df), ").")
    out <- df$latex[index]

    names <- if (is.null(variable_names)) x$feature_names else variable_names
    if (length(names) > 0L) {
        if (length(names) != x$n_features)
            stop("variable_names has ", length(names), " name(s) but the model ",
                 "was fitted on ", x$n_features, " feature(s).")
        names <- gsub("_", "\\_", names, fixed = TRUE)
        for (j in seq_along(names)) {
            out <- gsub(paste0("x_{", j - 1L, "}"), names[[j]], out, fixed = TRUE)
        }
    }
    out
}
