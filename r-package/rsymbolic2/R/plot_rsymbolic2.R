# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Plot an rsymbolic2 fit
#'
#' Three views of a fit returned by \code{\link{symbolic_regression}}, selected
#' with \code{type}:
#'
#' \describe{
#'   \item{\code{"pareto"} (default)}{Complexity vs.\ loss scatter plot of the
#'     non-dominated expressions found by the search.  The expression with the
#'     lowest loss is highlighted in red.}
#'   \item{\code{"fit"}}{The fitted expression against the data.  With a single
#'     feature, the fitted curve is overlaid on the observed scatter; with
#'     several features, predicted values are plotted against observed ones with
#'     a dashed \eqn{y = x} reference line.}
#'   \item{\code{"tree"}}{The structure of one expression as a syntax tree:
#'     operators as inner nodes, data columns and fitted constants as leaves
#'     (distinguished by fill).  Needs no data.  The node count is that of the
#'     expression as printed, which can be smaller than the \code{complexity}
#'     column of \code{x$pareto_front} -- that counts the raw tree the search
#'     archived, before the display-only simplification.}
#' }
#'
#' The result object stores no training data (only \code{n_obs} and \code{sst},
#' from which \code{summary()} derives R-squared), so \code{type = "fit"}
#' requires the data to be passed back in via \code{newdata} and \code{y}.
#' Supplying both without naming \code{type} selects \code{"fit"}.
#'
#' Requires the \pkg{ggplot2} package.
#'
#' @param x An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param y Observed target values for \code{type = "fit"}, one per row of
#'   \code{newdata}.  Ignored by \code{type = "pareto"}.
#' @param type Which plot to draw: \code{"pareto"} (default), \code{"fit"} or
#'   \code{"tree"}.
#' @param newdata Input features for \code{type = "fit"}, in the form
#'   \code{\link{predict.rsymbolic2}} accepts: a numeric matrix for a
#'   matrix-fitted model, or a \code{data.frame} holding the formula's predictor
#'   columns for a formula-fitted one.  Pass the training data to inspect the
#'   fit, or held-out data to inspect generalisation.
#' @param expression Which fitted expression to draw for \code{type = "fit"} or
#'   \code{type = "tree"}.  \code{NULL} (default) uses \code{x$recommended} (for
#'   \code{"tree"}, its display-simplified form when the fit carries one);
#'   otherwise an expression string, e.g.\ any row of
#'   \code{x$pareto_front$expression}.  For \code{type = "fit"} it is passed
#'   straight to \code{\link{predict.rsymbolic2}}.
#' @param variable_names Character vector of names to label the leaves with for
#'   \code{type = "tree"}.  Defaults to \code{x$feature_names} when set, else the
#'   0-based \code{x0, x1, ...} the expression strings use.  Ignored by the other
#'   plots (the fit plot labels its axes from \code{newdata}).
#' @param log_loss Logical; if \code{TRUE} (default) the loss axis of the Pareto
#'   plot uses a log10 scale.  Ignored by \code{type = "fit"}.
#' @param label_exprs Logical; if \code{TRUE} (default) annotates each Pareto
#'   point with its expression string.  Set to \code{FALSE} for large Pareto
#'   fronts where labels overlap.  Ignored by \code{type = "fit"}.
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
#' plot(res)                                  # Pareto front
#' plot(res, type = "fit", newdata = X, y = y)  # fitted curve over the data
#' plot(res, type = "tree")                     # structure of the recommended expression
#' }
#'
#' @export
plot.rsymbolic2 <- function(x, y = NULL, type = c("pareto", "fit", "tree"),
                            newdata = NULL, expression = NULL,
                            log_loss = TRUE, label_exprs = TRUE,
                            variable_names = NULL, ...) {
    if (!requireNamespace("ggplot2", quietly = TRUE))
        stop("Package 'ggplot2' is required for plot.rsymbolic2(). ",
             "Install it with: install.packages('ggplot2')")

    # Data is only ever used by the fit plot, so supplying it is an unambiguous
    # request for that plot; an explicit `type` always wins.
    if (missing(type) && !is.null(newdata) && !is.null(y)) type <- "fit"
    type <- match.arg(type)

    p <- if (type == "pareto") {
        pareto_plot(x, log_loss = log_loss, label_exprs = label_exprs)
    } else if (type == "fit") {
        fit_plot(x, newdata = newdata, y = y, expression = expression)
    } else {
        tree_plot(tree_expression(x, expression),
                  tree_variable_names(x, variable_names))
    }

    print(p)
    invisible(p)
}

# Complexity vs. loss over the archived front.
pareto_plot <- function(object, log_loss, label_exprs) {
    df       <- object$pareto_front
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

    p
}

# One expression against the data it was fitted to (or held-out data). The
# single-feature overlay is the more direct reading, but it only exists when
# there is one x-axis to draw; predicted-vs-actual is the general fallback.
fit_plot <- function(object, newdata, y, expression) {
    if (is.null(newdata) || is.null(y))
        stop('plot(type = "fit") needs the data to compare against: pass both ',
             "newdata = <features> and y = <observed target>. The result object ",
             "stores no training data.")

    X    <- design_matrix(object, newdata)
    yhat <- stats::predict(object, newdata, expression = expression)
    y    <- as.numeric(y)
    if (length(y) != length(yhat))
        stop(sprintf("y has %d value(s) but newdata has %d row(s).",
                     length(y), length(yhat)))

    # Column named `predicted`, not `fitted`: `fitted` is a stats generic, and a
    # data-frame column shadowing it inside aes() is exactly the ambiguity
    # R CMD check reports.
    if (ncol(X) == 1L) {
        df <- data.frame(feature = X[, 1], observed = y, predicted = yhat)
        p <- ggplot2::ggplot(df, ggplot2::aes(x = feature, y = observed)) +
            ggplot2::geom_point(size = 2, colour = "grey30") +
            ggplot2::geom_line(data = df[order(df$feature), , drop = FALSE],
                               ggplot2::aes(y = predicted), colour = "firebrick") +
            ggplot2::labs(x = feature_labels(X)[1], y = "observed",
                          title = "rsymbolic2 fit")
    } else {
        df <- data.frame(observed = y, predicted = yhat)
        p <- ggplot2::ggplot(df, ggplot2::aes(x = observed, y = predicted)) +
            ggplot2::geom_abline(slope = 1, intercept = 0,
                                 linetype = "dashed", colour = "grey50") +
            ggplot2::geom_point(size = 2, colour = "grey30") +
            ggplot2::labs(x = "observed", y = "predicted",
                          title = "rsymbolic2 fit: predicted vs observed")
    }

    p + ggplot2::theme_bw()
}

# Which expression the tree draws. Display surfaces prefer the display-simplified
# companion (docs/52) -- it is what print() and the LaTeX rendering already show -- while
# `recommended` stays the frozen evaluatable string and the fallback for fits made before
# the simplified field existed. An explicit `expression` always wins, so any Pareto member
# (or the raw searched form, x$expression) can be drawn.
tree_expression <- function(object, expression) {
    if (!is.null(expression)) return(expression)
    simplified <- object$recommended_simplified
    if (!is.null(simplified) && nzchar(simplified)) simplified else object$recommended
}

# Leaf labels: the caller's names, else the fitted column names, else the 0-based x0, x1,
# ... the expression strings themselves use (tree_plot's fallback when the vector is empty).
tree_variable_names <- function(object, variable_names) {
    nms <- if (is.null(variable_names)) object$feature_names else variable_names
    if (length(nms) > 0L && !is.null(object$n_features) &&
        length(nms) != object$n_features) {
        stop("variable_names has ", length(nms), " name(s) but the model was fitted on ",
             object$n_features, " feature(s).")
    }
    nms
}

# Axis labels for the design matrix: the fitted column names when they survived
# (formula fits, and matrices carrying colnames), else the 0-based x0, x1, ...
# naming the expression strings themselves use.
feature_labels <- function(X) {
    nm <- colnames(X)
    if (is.null(nm)) paste0("x", seq_len(ncol(X)) - 1L) else nm
}
