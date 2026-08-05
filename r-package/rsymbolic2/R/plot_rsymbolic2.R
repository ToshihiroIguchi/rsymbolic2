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
#' \code{type = "fit"} draws against the training data stored on the object when
#' neither \code{newdata} nor \code{y} is given, which needs \code{keep_data = TRUE}
#' (the default) at fit time.  Pass \emph{both} to plot against other data --
#' held-out data, to inspect generalisation.  Supplying both without naming
#' \code{type} selects \code{"fit"}.
#'
#' Requires the \pkg{ggplot2} package.
#'
#' @param x An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param y Observed target values for \code{type = "fit"}, one per row of
#'   \code{newdata}.  \code{NULL} (default) with \code{newdata} also \code{NULL}
#'   uses the stored training data.  Ignored by \code{type = "pareto"}.
#' @param type Which plot to draw: \code{"pareto"} (default), \code{"fit"} or
#'   \code{"tree"}.
#' @param newdata Input features for \code{type = "fit"}, in the form
#'   \code{\link{predict.rsymbolic2}} accepts: a numeric matrix for a
#'   matrix-fitted model, or a \code{data.frame} holding the formula's predictor
#'   columns for a formula-fitted one.  \code{NULL} (default) with \code{y} also
#'   \code{NULL} uses the training data stored on the object; pass both to
#'   inspect generalisation on held-out data.
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
#' @seealso \code{\link{predict.rsymbolic2}} for the values \code{type = "fit"}
#'   draws, \code{\link{as.data.frame.rsymbolic2}} for the front behind
#'   \code{type = "pareto"}, and \code{\link{symbolic_regression}} for the fit
#'   itself.
#'
#' @examples
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- sin(X[, 1]) + X[, 1]^2
#' res <- symbolic_regression(X, y, population_size = 200L,
#'                            generations = 40L, seed = 1L, n_threads = 2L)
#'
#' # ggplot2 is a suggested package, so draw only when it is available.
#' if (requireNamespace("ggplot2", quietly = TRUE)) {
#'   plot(res)                     # Pareto front
#'   plot(res, type = "fit")       # fitted curve over the training data
#'   plot(res, type = "tree")      # structure of the recommended expression
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
    # No data supplied: draw the fit against the data it was trained on, which is what
    # plot(fit) means for an R regression object. Available whenever the fit kept its
    # data (keep_data = TRUE, the default); training_matrix() raises with instructions
    # when it did not (docs/81 P2). Supplying both arguments still plots held-out data.
    if (is.null(newdata) && is.null(y)) {
        X <- training_matrix(object, "plot")
        if (is.null(object$y))
            stop("this model was fitted with keep_data = FALSE, so it carries no ",
                 "training response to plot against. Re-fit with keep_data = TRUE, or ",
                 "pass newdata = <features> and y = <observed target>.", call. = FALSE)
        y    <- as.numeric(object$y)
        yhat <- eval_on_matrix(X, object, expression)
    } else {
        if (is.null(newdata) || is.null(y))
            stop('plot(type = "fit") needs both halves of the data: pass newdata = ',
                 "<features> and y = <observed target>, or neither to use the training ",
                 "data stored on the object.", call. = FALSE)
        X    <- design_matrix(object, newdata)
        yhat <- stats::predict(object, newdata, expression = expression)
        y    <- as.numeric(y)
    }
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
