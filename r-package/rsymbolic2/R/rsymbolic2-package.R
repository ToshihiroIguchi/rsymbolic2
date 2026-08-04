# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' @seealso \code{\link{symbolic_regression}} is the package's entry point: it
#'   runs the search and returns a fit that \code{\link{print.rsymbolic2}},
#'   \code{\link{summary.rsymbolic2}}, \code{\link{plot.rsymbolic2}},
#'   \code{\link{predict.rsymbolic2}}, \code{\link{fitted.rsymbolic2}},
#'   \code{\link{residuals.rsymbolic2}}, \code{\link{as.data.frame.rsymbolic2}},
#'   \code{\link{to_latex}} and \code{\link{to_sympy}} all work on.
#'
#'   Useful links:
#'   \itemize{
#'     \item \url{https://github.com/ToshihiroIguchi/rsymbolic2}
#'     \item Report bugs at
#'       \url{https://github.com/ToshihiroIguchi/rsymbolic2/issues}
#'   }
#'
#' @useDynLib rsymbolic2, .registration = TRUE
"_PACKAGE"

# Data-frame column names referenced by non-standard evaluation inside
# ggplot2::aes() in the plot.rsymbolic2() helpers -- the Pareto front's own
# columns (pareto_plot), the frame fit_plot assembles, and the node/edge table
# tree_plot builds. Declared here so R CMD check does not flag them as undefined
# global variables.
utils::globalVariables(c(
    "complexity", "loss", "expression",
    "feature", "observed", "predicted",
    "x", "y", "xend", "yend", "drawn", "kind"
))
