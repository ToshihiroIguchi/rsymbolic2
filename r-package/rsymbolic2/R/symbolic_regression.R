#' Symbolic Regression
#'
#' Discover a mathematical expression that fits the data using genetic
#' programming with Levenberg-Marquardt constant optimisation.
#'
#' @param X Numeric matrix of input features (rows = observations, columns =
#'   features). A numeric vector is treated as a single-column matrix.
#' @param y Numeric vector of target values; \code{length(y)} must equal
#'   \code{nrow(X)}.
#' @param population_size Number of candidate expressions kept in each island
#'   population (default 200).
#' @param n_populations Number of independent island populations to evolve in
#'   parallel (default 1). Values greater than 1 enable OpenMP-parallel island
#'   evolution with periodic ring migration. When OpenMP is unavailable the
#'   islands still run but sequentially.
#' @param generations Number of evolution generations (default 50).
#' @param tournament_size Tournament size for selection and replacement
#'   (default 4).
#' @param unary_ops Character vector of unary operators to allow.  Recognised
#'   values: \code{"neg"}, \code{"exp"}, \code{"log"}, \code{"sin"},
#'   \code{"cos"}.  Default uses all five.
#' @param binary_ops Character vector of binary operators.  Recognised values:
#'   \code{"add"}, \code{"sub"}, \code{"mul"}, \code{"div"}.  Default is
#'   \code{c("add","sub","mul")}.
#' @param max_depth Maximum tree depth for randomly generated expressions
#'   (default 4).
#' @param max_nodes Soft upper bound on expression-tree size (default 40).
#' @param target_loss Early-stop threshold: search halts once the best
#'   training loss falls below this value (default \code{1e-10}).
#' @param simplify Algebraically simplify fitted candidates (default
#'   \code{TRUE}).
#' @param crossover_probability Probability of subtree crossover vs. mutation
#'   per evolution step (default 0.5).
#' @param seed Integer random seed for reproducibility; 0 uses a
#'   non-deterministic seed (default 0).
#'
#' @return A list with elements:
#'   \describe{
#'     \item{expression}{Best expression found, as an infix string.}
#'     \item{loss}{Training loss (sum of squared residuals) of the best
#'       expression.}
#'     \item{complexity}{Number of nodes in the expression tree.}
#'     \item{pareto_front}{Data frame of non-dominated (complexity, loss,
#'       expression) trade-offs from accuracy-vs-complexity Pareto front.}
#'   }
#'
#' @examples
#' \donttest{
#' # Single-variable linear: y = 2.5*x + 1.7
#' X <- matrix(seq(-5, 5, length.out = 20), ncol = 1)
#' y <- 2.5 * X[, 1] + 1.7
#' res <- symbolic_regression(X, y, unary_ops = character(0),
#'                            population_size = 200L, generations = 40L,
#'                            seed = 42L)
#' cat(res$expression, "\n")
#' }
#'
#' @export
symbolic_regression <- function(
    X,
    y,
    population_size       = 200L,
    n_populations         = 1L,
    generations           = 50L,
    tournament_size       = 4L,
    unary_ops             = c("neg", "exp", "log", "sin", "cos"),
    binary_ops            = c("add", "sub", "mul"),
    max_depth             = 4L,
    max_nodes             = 40L,
    target_loss           = 1e-10,
    simplify              = TRUE,
    crossover_probability = 0.5,
    seed                  = 0L
) {
    X <- as.matrix(X)
    y <- as.numeric(y)

    if (nrow(X) == 0L) stop("X must have at least one row")
    if (nrow(X) != length(y)) stop("nrow(X) must equal length(y)")
    if (!is.numeric(X)) stop("X must be numeric")

    symbolic_regression_cpp(
        X,
        y,
        as.integer(population_size),
        as.integer(generations),
        as.integer(tournament_size),
        as.character(unary_ops),
        as.character(binary_ops),
        as.integer(max_depth),
        as.integer(max_nodes),
        as.double(target_loss),
        as.logical(simplify),
        as.double(crossover_probability),
        as.double(seed),
        as.integer(n_populations)
    )
}
