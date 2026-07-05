# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Summarise a symbolic regression fit
#'
#' Builds a detailed summary of an \code{\link{symbolic_regression}} result: the
#' complete accuracy-vs-complexity Pareto front with a per-member \emph{score}
#' (the loss-drop-per-complexity the engine uses to pick the recommended model),
#' plus headline metrics. The returned object has a \code{print} method; see
#' \code{\link{print.rsymbolic2}} for the compact form.
#'
#' @param object An object of class \code{"rsymbolic2"} returned by
#'   \code{\link{symbolic_regression}}.
#' @param ... Ignored (present for S3 compatibility).
#'
#' @return An object of class \code{"summary.rsymbolic2"}: a list with elements
#'   \describe{
#'     \item{pareto}{Data frame of the Pareto front with columns
#'       \code{complexity}, \code{loss}, \code{score}, \code{r_squared},
#'       \code{recommended} (logical), and \code{expression}. \code{score} is
#'       \eqn{(\log L_{i-1} - \log L_i) / (c_i - c_{i-1})}, computed by the C++
#'       core (the same value \code{model_selection} ranks by); \code{0} for the
#'       simplest member. \code{r_squared} is \eqn{1 - L_i / SST} on the training
#'       data (weighted when \code{weights} were used); \code{NA} when the
#'       target is constant (\eqn{SST = 0}).}
#'     \item{n_features}{Number of input features used during fitting.}
#'     \item{feature_names}{Column names of \code{X} (display-only), or \code{NULL}
#'       when \code{X} had none.}
#'     \item{n_members}{Number of Pareto-front members.}
#'     \item{best_index}{Row of \code{pareto} flagged as \code{recommended}.}
#'     \item{recommended_expression, best_expression}{The recommended (Pareto
#'       "knee") and lowest-loss expressions.}
#'     \item{best_loss, best_complexity}{Loss and complexity of the lowest-loss
#'       expression.}
#'   }
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- sin(X[, 1]) + X[, 1]^2
#' res <- symbolic_regression(X, y, population_size = 200L,
#'                            generations = 40L, seed = 1L)
#' summary(res)
#' }
#'
#' @export
summary.rsymbolic2 <- function(object, ...) {
    df <- object$pareto_front
    # Training R^2 per member: 1 - loss/SST. NA when the target was constant
    # (SST = 0) or the object predates the sst field. Negative values are valid
    # (a fit worse than the mean).
    sst <- object$sst
    r_squared <- if (!is.null(sst) && is.finite(sst) && sst > 0) {
        1 - df$loss / sst
    } else {
        rep(NA_real_, nrow(df))
    }
    recommended <- seq_len(nrow(df)) == object$best_index
    pareto <- data.frame(
        complexity  = df$complexity,
        loss        = df$loss,
        score       = df$score,
        r_squared   = r_squared,
        recommended = recommended,
        expression  = df$expression,
        stringsAsFactors = FALSE
    )
    structure(
        list(
            pareto                 = pareto,
            n_features             = object$n_features,
            feature_names          = object$feature_names,
            n_members              = nrow(df),
            best_index             = object$best_index,
            recommended_expression = object$recommended,
            best_expression        = object$expression,
            best_loss              = object$loss,
            best_complexity        = object$complexity,
            r_squared              = if (any(recommended)) r_squared[recommended]
                                     else NA_real_
        ),
        class = "summary.rsymbolic2"
    )
}

#' @rdname summary.rsymbolic2
#' @param x An object of class \code{"summary.rsymbolic2"}.
#' @return \code{print.summary.rsymbolic2} returns \code{x}, invisibly.
#' @export
print.summary.rsymbolic2 <- function(x, ...) {
    cat("Symbolic regression fit (rsymbolic2)\n")
    cat(sprintf("  features: %s   Pareto members: %d\n",
                if (is.null(x$n_features)) "?" else x$n_features, x$n_members))
    legend <- format_feature_legend(x$feature_names, x$n_features)
    if (!is.null(legend))
        cat("  variables:        ", paste(legend, collapse = ", "), "\n", sep = "")
    if (x$n_members > 0L) {
        cat(sprintf("  loss range:       %s ... %s\n",
                    fmt_g(min(x$pareto$loss)), fmt_g(max(x$pareto$loss))))
        cat(sprintf("  complexity range: %d ... %d\n",
                    min(x$pareto$complexity), max(x$pareto$complexity)))
    }
    cat("  recommended:        ", x$recommended_expression, "\n", sep = "")
    cat(sprintf("  best (lowest loss): %s  (loss=%s, complexity=%s)\n",
                x$best_expression, fmt_g(x$best_loss), x$best_complexity))
    cat(sprintf("  R-squared (recommended): %s\n",
                if (is.na(x$r_squared)) "NA" else fmt_g(x$r_squared)))
    cat("\nPareto front:\n")

    disp <- x$pareto
    disp$loss  <- fmt_g(disp$loss)
    disp$score <- ifelse(is.na(disp$score), "", fmt_g(disp$score))
    disp$r_squared <- ifelse(is.na(disp$r_squared), "", fmt_g(disp$r_squared))
    disp$recommended <- ifelse(disp$recommended, "*", "")
    names(disp)[names(disp) == "recommended"] <- "rec"
    print(disp, row.names = FALSE, right = FALSE)
    invisible(x)
}
