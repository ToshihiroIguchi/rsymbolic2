# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

# Format a numeric vector for compact display (PySR uses %.6g; formatC with
# format="g", digits=6 is the R equivalent).
fmt_g <- function(x) formatC(x, format = "g", digits = 6L)

# Render the Pareto front as aligned "complexity | loss | expression" lines, with
# a ">" marker on the recommended row. When the front is longer than `max_rows`,
# the head and tail are shown with an elided middle. Shared by print() and the
# package's other display methods.
format_pareto_lines <- function(df, best_index, indent = "  ", max_rows = 20L) {
    n <- nrow(df)
    if (n == 0L) return(paste0(indent, "(empty Pareto front)"))

    marker <- ifelse(seq_len(n) == best_index, ">", " ")
    comp   <- formatC(df$complexity, width = max(nchar(df$complexity)), flag = " ")
    loss   <- formatC(fmt_g(df$loss), width = max(nchar(fmt_g(df$loss))), flag = "-")
    rows   <- paste0(indent, marker, " ", comp, " | ", loss, " | ", df$expression)

    if (n <= max_rows) return(rows)
    head_n <- max_rows %/% 2L
    tail_n <- max_rows - head_n
    c(rows[seq_len(head_n)],
      paste0(indent, "  ... (", n - max_rows, " more) ..."),
      rows[(n - tail_n + 1L):n])
}

# Build "x0 = name0, x1 = name1" legend strings mapping the 0-based variables used
# in the expression strings to the column names of X. Returns NULL when no names are
# available (or they don't match n_features), in which case callers print no legend
# and the expressions read in their x0-based form. Shared by print() and summary().
format_feature_legend <- function(feature_names, n_features = NULL) {
    if (is.null(feature_names) || length(feature_names) == 0L) return(NULL)
    if (!is.null(n_features) && length(feature_names) != n_features) return(NULL)
    paste0("x", seq_along(feature_names) - 1L, " = ", feature_names)
}

#' Print a symbolic regression fit
#'
#' Compactly summarises an \code{\link{symbolic_regression}} result: the
#' recommended and lowest-loss expressions and the Pareto front (complexity vs.
#' loss). For the full front with per-member scores use
#' \code{\link{summary.rsymbolic2}}.
#'
#' @param x An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param ... Ignored (present for S3 compatibility).
#'
#' @return \code{x}, invisibly.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2.5 * X[, 1] + 1.7
#' res <- symbolic_regression(X, y, unary_ops = character(0),
#'                            population_size = 200L, generations = 40L, seed = 1L)
#' print(res)
#' }
#'
#' @export
print.rsymbolic2 <- function(x, ...) {
    df <- x$pareto_front
    cat(sprintf("<rsymbolic2: %d Pareto member%s, n_features=%s>\n",
                nrow(df), if (nrow(df) == 1L) "" else "s",
                if (is.null(x$n_features)) "?" else x$n_features))
    legend <- format_feature_legend(x$feature_names, x$n_features)
    if (!is.null(legend))
        cat("  variables:          ", paste(legend, collapse = ", "), "\n", sep = "")
    cat("  recommended:        ", x$recommended, "\n", sep = "")
    cat(sprintf("  best (lowest loss): %s  (loss=%s, complexity=%s)\n",
                x$expression, fmt_g(x$loss), x$complexity))
    cat("  Pareto front (> = recommended; complexity | loss | expression):\n")
    cat(format_pareto_lines(df, x$best_index), sep = "\n")
    cat("\n")
    invisible(x)
}
