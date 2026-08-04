# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' Render fitted expressions as SymPy-parseable Python
#'
#' Generic for extracting a Python/SymPy rendering of a fitted model. See
#' \code{\link{to_sympy.rsymbolic2}} for the method on symbolic regression fits.
#'
#' @param x A fitted model object.
#' @param ... Passed to methods.
#'
#' @return A character vector of Python expression strings.
#'
#' @export
to_sympy <- function(x, ...) UseMethod("to_sympy")

#' @describeIn to_sympy Python source for one or more Pareto-front members of a
#'   \code{\link{symbolic_regression}} fit, in the form SymPy's
#'   \code{sympify()} accepts.
#'
#'   What this rewrites is \code{^}. The engine's power operator is \code{^} on
#'   every display surface, and Python reads that as \emph{xor}: \code{eval()},
#'   \code{parse_expr()}, \code{lambdify()} and NumPy all evaluate the wrong
#'   function, silently, for any expression containing one — and since
#'   \code{square} prints as \code{(a ^ 2)}, that is most of them.
#'   (\code{sympify()} alone is the exception: it passes
#'   \code{convert_xor=True}.) Every other operator in an \code{expression}
#'   string already carries its Python name, so this rendering is valid under
#'   \code{sympify()}, \code{parse_expr()}, \code{eval()} and NumPy alike.
#'
#'   Display-only, like \code{\link{to_latex.rsymbolic2}}:
#'   \code{\link{predict.rsymbolic2}} keeps using the \code{expression} strings.
#'   \strong{This is the mathematical form, not the engine's.} rsymbolic2's
#'   operators are domain-guarded — \code{sqrt}, \code{log} and \code{^} return
#'   \code{NaN} outside their domain, where SymPy returns a complex or symbolic
#'   value. Use \code{predict()} to evaluate, and this to differentiate,
#'   simplify or typeset.
#'
#' @param index Integer vector of rows of \code{x$pareto_front} to render.
#'   Defaults to \code{x$best_index}, the member chosen by
#'   \code{model_selection}.
#' @param variable_names Character vector substituted for the \code{x0},
#'   \code{x1}, ... tokens. Defaults to \code{NULL}, which keeps them — unlike
#'   \code{\link{to_latex.rsymbolic2}}, feature names are \strong{not}
#'   substituted automatically, because a column name is free text and
#'   \code{"flow rate"} is not a Python identifier. Names passed here must be
#'   syntactically valid Python identifiers.
#'
#' @seealso \code{x$pareto_front$sympy_simplified} for the same rendering of the
#'   display-simplified form (what \code{print()} and \code{summary()} show),
#'   \code{\link{to_latex.rsymbolic2}} for the LaTeX rendering, and
#'   \code{\link{predict.rsymbolic2}} to evaluate a fit.
#'
#' @examples
#' \donttest{
#' X <- matrix(seq(-3, 3, length.out = 20), ncol = 1)
#' y <- 2 * X[, 1]^2 - 1
#' res <- symbolic_regression(X, y, unary_ops = c("square"),
#'                            population_size = 200L, generations = 40L,
#'                            seed = 1L)
#' to_sympy(res)                                  # recommended member
#' to_sympy(res, variable_names = "t")            # x0 -> t
#' }
#'
#' @export
to_sympy.rsymbolic2 <- function(x, index = NULL, variable_names = NULL, ...) {
    df <- x$pareto_front
    if (is.null(df$sympy))
        stop("this fit has no 'sympy' column; re-fit with the current ",
             "rsymbolic2 version to use to_sympy().")
    if (is.null(index)) index <- x$best_index
    if (length(index) == 0L || anyNA(index) ||
        any(index < 1L | index > nrow(df)))
        stop("index must select rows of x$pareto_front (1..", nrow(df), ").")
    out <- df$sympy[index]

    if (is.null(variable_names)) return(out)
    if (length(variable_names) != x$n_features)
        stop("variable_names has ", length(variable_names), " name(s) but the ",
             "model was fitted on ", x$n_features, " feature(s).")
    # A name that is not a Python identifier would produce a string SymPy cannot
    # parse, which is the one thing this function exists to prevent. Rejecting it
    # is better than emitting it: the caller still has the x0 form to fall back on.
    bad <- !grepl("^[A-Za-z_][A-Za-z0-9_]*$", variable_names)
    if (any(bad))
        stop("variable_names must be valid Python identifiers; rejected: ",
             paste(sQuote(variable_names[bad]), collapse = ", "))

    # Two passes through a control-character placeholder, so a name that is
    # itself a token (variable_names = c("x1", "x0"), a swap) cannot be
    # re-substituted by a later iteration.
    for (j in seq_along(variable_names))
        out <- gsub(paste0("\\bx", j - 1L, "\\b"),
                    paste0("\001", j, "\002"), out, perl = TRUE)
    for (j in seq_along(variable_names))
        out <- gsub(paste0("\001", j, "\002"), variable_names[[j]], out,
                    fixed = TRUE)
    out
}
