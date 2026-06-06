#' Plot the Pareto front of an rsymbolic2 fit
#'
#' Draws a complexity vs.\ loss scatter plot of the non-dominated expressions
#' found by \code{\link{symbolic_regression}}.  The expression with the
#' lowest loss is highlighted in red.  Requires the \pkg{ggplot2} package.
#'
#' @param x An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param log_loss Logical; if \code{TRUE} (default) the loss axis uses a
#'   log10 scale.
#' @param label_exprs Logical; if \code{TRUE} (default) annotates each point
#'   with its expression string.  Set to \code{FALSE} for large Pareto fronts
#'   where labels overlap.
#' @param ... Ignored (present for S3 compatibility).
#'
#' @return A \code{ggplot2} plot object, returned invisibly.
#'   The plot is also printed as a side effect.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- sin(X[, 1]) + X[, 1]^2
#' res <- symbolic_regression(X, y, population_size = 200L,
#'                            generations = 40L, seed = 1L)
#' plot(res)
#' }
#'
#' @export
plot.rsymbolic2 <- function(x, log_loss = TRUE, label_exprs = TRUE, ...) {
    if (!requireNamespace("ggplot2", quietly = TRUE))
        stop("Package 'ggplot2' is required for plot.rsymbolic2(). ",
             "Install it with: install.packages('ggplot2')")

    df       <- x$pareto_front
    best_idx <- which.min(df$loss)

    p <- ggplot2::ggplot(df, ggplot2::aes(x = complexity, y = loss)) +
        ggplot2::geom_line(colour = "grey60") +
        ggplot2::geom_point(size = 3) +
        ggplot2::geom_point(data = df[best_idx, , drop = FALSE],
                            size = 4, colour = "firebrick") +
        ggplot2::labs(x = "Complexity (nodes)", y = "Loss (SSE)",
                      title = "rsymbolic2 Pareto front") +
        ggplot2::theme_bw()

    if (log_loss && all(df$loss > 0))
        p <- p + ggplot2::scale_y_log10()

    if (label_exprs)
        p <- p + ggplot2::geom_text(
            ggplot2::aes(label = expression),
            vjust = -0.6, hjust = 0, size = 3)

    print(p)
    invisible(p)
}
