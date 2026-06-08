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
#'   \code{"cos"}, \code{"sqrt"}, \code{"tanh"}, \code{"abs"}, \code{"square"}.
#'   Default uses the first five; \code{"sqrt"}, \code{"tanh"}, \code{"abs"},
#'   and \code{"square"} must be added explicitly when needed.
#'   \code{"square"} computes \eqn{x^2} as a single node (cheaper than
#'   \code{pow} for the common quadratic case).
#' @param binary_ops Character vector of binary operators.  Recognised values:
#'   \code{"add"}, \code{"sub"}, \code{"mul"}, \code{"div"}, \code{"pow"}.
#'   Default is \code{c("add","sub","mul")}.
#'   \code{"pow"} uses a safe implementation that avoids \code{NaN} when the
#'   base is non-positive: \code{x^y} returns 0 for undefined inputs instead
#'   of \code{NaN}, preventing LM-solver poisoning.
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
#' @param parsimony Complexity penalty added to the selection cost.
#'   Selection cost = \code{loss / y_norm + parsimony * complexity}, where
#'   \code{y_norm = sum((y - mean(y))^2)} (the NMSE denominator). This makes
#'   the penalty scale-stable across problems of different y-variance, so a
#'   given \code{parsimony} value transfers across datasets. Default 1e-3,
#'   chosen by sweeping \code{c(0, 1e-4, 5e-4, 1e-3, 5e-3)} on Nguyen
#'   N1/N5/N7/N9/N10: p=1e-3 gave ~10x speedup on Nguyen N9 (123s→12s) with
#'   no fast-case regression and full recovery. Positive values penalise
#'   large expressions in tournament selection and migration, which reduces
#'   bloat without changing the HallOfFame Pareto archive (which is still keyed
#'   on raw loss vs. complexity). Use 0 to restore the pre-B2 behaviour
#'   (selection by raw loss only).
#' @param optimize_probability Probability that a newly produced child
#'   expression has its constants refined by the Levenberg-Marquardt optimizer
#'   (default 0.1). Values less than 1 make constant optimization a
#'   probabilistic event, trading some fit quality for substantially faster
#'   generation on large or constant-heavy trees (cf. PySR
#'   \code{weight_optimize}). The default 0.1 was chosen by sweeping
#'   \code{c(1.0, 0.5, 0.2, 0.1)} on Nguyen N1/N5/N7/N9/N10: all values
#'   recovered at every p; p=0.1 gave 3.5–4x speedup on slow cases with no
#'   fast-case regression. Children that skip optimization are still evaluated
#'   with their inherited constants and remain valid candidates for selection.
#'   Use \code{1.0} to restore the pre-B1 behaviour (optimize every child).
#' @param timeout_seconds Wall-clock time limit in seconds (default 0 = no
#'   limit). When positive, the search stops after approximately this many
#'   seconds and returns the best expression found so far. \strong{Note:} a run
#'   that hits the timeout is not reproducible across machines; only runs that
#'   complete within the budget are bit-reproducible for a fixed seed.
#' @param verbosity Integer verbosity level (default 0 = silent). Set to 1 to
#'   print one diagnostic line per epoch on stderr showing elapsed time, best
#'   loss, and population shape (median/max tree size and constant count). Useful
#'   for diagnosing slow runs.
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
    seed                  = 0L,
    parsimony             = 1e-3,
    optimize_probability  = 0.1,
    timeout_seconds       = 0,
    verbosity             = 0L
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
        as.integer(n_populations),
        as.double(timeout_seconds),
        as.integer(verbosity),
        as.double(optimize_probability),
        as.double(parsimony)
    )
}
