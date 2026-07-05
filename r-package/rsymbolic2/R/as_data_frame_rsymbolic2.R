# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Coerce a symbolic regression fit to a data frame
#'
#' Returns the Pareto front of an \code{\link{symbolic_regression}} result as a
#' tidy data frame, the R counterpart of the Python interface's
#' \code{.to_pandas()}. Useful for further analysis, sorting, or export.
#'
#' @param x An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param ... Ignored (present for S3 compatibility).
#' @param score Logical; if \code{TRUE} (default) include a \code{score} column
#'   (loss-drop-per-complexity, computed by the C++ core; \code{0} for the
#'   simplest member).
#'
#' @return A data frame with columns \code{complexity}, \code{loss},
#'   (optionally \code{score}), \code{recommended} (logical; \code{TRUE} for the
#'   member returned as \code{recommended}), and \code{expression}, one row per
#'   Pareto-front member in increasing complexity.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2 * X[, 1]^2 - 1
#' res <- symbolic_regression(X, y, population_size = 200L,
#'                            generations = 40L, seed = 1L)
#' as.data.frame(res)
#' }
#'
#' @export
as.data.frame.rsymbolic2 <- function(x, ..., score = TRUE) {
    df  <- x$pareto_front
    out <- data.frame(
        complexity  = df$complexity,
        loss        = df$loss,
        stringsAsFactors = FALSE
    )
    if (isTRUE(score))
        out$score <- df$score
    out$recommended <- seq_len(nrow(df)) == x$best_index
    out$expression  <- df$expression
    out
}
