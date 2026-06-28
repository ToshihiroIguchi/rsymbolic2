# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

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
#'   population (default 27, PySR \code{population_size}).
#' @param n_populations Number of independent island populations to evolve in
#'   parallel (default 31, PySR \code{populations}). Values greater than 1 enable
#'   OpenMP-parallel island evolution with periodic ring migration. When OpenMP is
#'   unavailable the islands still run but sequentially.
#' @param generations Number of evolution generations (default 2800). One
#'   generation performs \code{population_size} tournament-and-replace mutation
#'   steps. The default reproduces PySR's per-population mutation budget: PySR runs
#'   \code{niterations = 100} iterations of \code{ncycles_per_iteration = 380}
#'   reg-evolution cycles, each cycle doing
#'   \code{ceil(population_size / tournament_size)} mutations and migrating once per
#'   iteration, so one iteration is \code{380 * ceil(27/15) = 760} mutations =
#'   \code{round(760/27) = 28} generations and the whole run is
#'   \code{100 * 28 = 2800} (see \code{docs/28} \eqn{C}).
#' @param tournament_size Tournament size for selection and replacement
#'   (default 15, PySR \code{tournament_selection_n}).
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
#'   (default 30, PySR \code{maxdepth}, which defaults to \code{maxsize}).
#' @param max_nodes Soft upper bound on expression-tree size (default 30, PySR
#'   \code{maxsize}). Also sizes the adaptive-parsimony frequency histogram.
#' @param target_loss Early-stop threshold: search halts once the best
#'   training loss falls below this value (default \code{1e-10}).
#' @param early_stop_condition Additional early-stop loss threshold (PySR
#'   \code{early_stop_condition}, numeric form), default \code{0} (off). When
#'   positive, the search also stops once the best loss falls below it, so the
#'   effective threshold is \code{max(target_loss, early_stop_condition)}; a value
#'   larger than \code{target_loss} stops the run sooner at a looser fit. Only the
#'   numeric form is supported (not PySR's loss/complexity function form).
#' @param max_evals Maximum number of candidate evaluations before the search
#'   stops (PySR \code{max_evals}), default \code{0} (no limit). Counts forward-pass
#'   loss evaluations and the residual evaluations consumed by constant-optimisation
#'   fits, summed across islands. Enforced via a deterministic per-island fair share
#'   plus a global check at epoch boundaries, so unlike \code{timeout_seconds} a
#'   capped run is still reproducible across machines and thread counts; the cap is
#'   coarse and may overshoot by up to one generation per island.
#' @param model_selection Which Pareto-front member is reported as
#'   \code{recommended}/\code{best_index} (PySR \code{model_selection}): one of
#'   \code{"best"} (default), \code{"accuracy"}, or \code{"score"}. \code{"accuracy"}
#'   picks the lowest-loss (most complex) member; \code{"score"} the highest
#'   loss-drop-per-complexity "knee" over the whole front; \code{"best"} the same
#'   knee but only among members within 1.5x of the most accurate loss.
#' @param weights Optional numeric vector of per-point weights for a weighted
#'   least-squares fit (PySR \code{weights}); \code{NULL} (default) fits unweighted.
#'   When supplied, \code{length(weights)} must equal \code{length(y)} and the loss
#'   becomes the weighted SSE \eqn{\sum_i w_i (\hat y_i - y_i)^2}. Weights must be
#'   non-negative.
#' @param simplify Algebraically simplify fitted candidates (default
#'   \code{TRUE}).
#' @param crossover_probability Probability of subtree crossover vs. mutation
#'   per evolution step (default 0.0259, PySR \code{crossover_probability}).
#' @param seed Integer random seed for reproducibility; 0 uses a
#'   non-deterministic seed (default 0).
#' @param parsimony Complexity penalty added to the selection cost.
#'   Selection cost = \code{loss / y_norm + parsimony * complexity}, where
#'   \code{y_norm = sum((y - mean(y))^2)} (the NMSE denominator). This makes
#'   the penalty scale-stable across problems of different y-variance, so a
#'   given \code{parsimony} value transfers across datasets. Default \strong{0.0}
#'   to match PySR's \code{parsimony} default (\code{docs/28}): the
#'   frequency-adaptive term (\code{adaptive_parsimony_scaling}) carries the size
#'   pressure instead, so this fixed linear penalty is off. Positive values penalise
#'   large expressions in tournament selection and migration, which reduces
#'   bloat without changing the HallOfFame Pareto archive (which is still keyed
#'   on raw loss vs. complexity).
#' @param adaptive_parsimony_scaling Strength of frequency-based adaptive
#'   parsimony, borrowed from PySR / SymbolicRegression.jl (default \strong{1040},
#'   PySR's installed default; \code{0} disables it).
#'   When positive, tournament selection multiplies each candidate's base cost by
#'   \code{exp(adaptive_parsimony_scaling * f)}, where \code{f} is the normalised
#'   frequency of that candidate's complexity in a per-island running histogram.
#'   This penalises \emph{over-represented} complexities rather than \emph{large}
#'   ones, so when the population piles up at a single size that size is pushed
#'   back. It is a self-balancing diversity pressure that counters the premature
#'   collapse the fixed-scalar \code{parsimony} causes on low-variance targets
#'   (see \code{docs/23}, \code{docs/24}). The cost factor matches
#'   SymbolicRegression.jl: \code{member.cost * exp(scaling * f)} with \code{f} the
#'   normalised frequency (sums to 1 over the \code{maxsize} bins). rsymbolic2 now
#'   matches PySR: ON at \strong{1040} with the histogram sized exactly by
#'   \code{maxsize} (= \code{max_nodes}), so the value transfers directly (see
#'   \code{docs/28}). The HallOfFame Pareto archive is unaffected (still keyed on
#'   raw loss).
#' @param optimize_probability Probability that a newly produced child
#'   expression has its constants refined by the Levenberg-Marquardt optimizer
#'   (default 0.14, PySR \code{optimize_probability}). Values less than 1 make
#'   constant optimization a probabilistic event, trading some fit quality for
#'   substantially faster generation on large or constant-heavy trees (cf. PySR
#'   \code{weight_optimize}). Children that skip optimization are still evaluated
#'   with their inherited constants and remain valid candidates for selection.
#'   Use \code{1.0} to optimize every child.
#' @param tournament_selection_p Probabilistic tournament strength (default 0.982,
#'   PySR's installed default). In each parent selection the rank-\eqn{r} best of the
#'   \code{tournament_size} sampled candidates is chosen with probability
#'   \eqn{p(1-p)^r}, so a slightly worse parent is occasionally taken to maintain
#'   diversity. \code{1.0} reproduces a deterministic best-of-k tournament.
#' @param should_optimize_constants Re-optimise the constants of every
#'   hall-of-fame member after each epoch (default \code{TRUE}, matching PySR's
#'   \code{should_optimize_constants}). Because \code{optimize_probability} < 1 lets
#'   most candidates enter the archive with un-tuned constants, this polish can push a
#'   structurally-correct near-miss below the recovery threshold.
#' @param fraction_replaced_hof Hall-of-fame migration fraction (default 0.0614,
#'   PySR's installed default). After each epoch the global elite archive is reinjected into every
#'   island population, replacing \code{round(fraction_replaced_hof * population_size)}
#'   of the worst members when the elite is better. \code{0} disables it.
#' @param mutation_weights Optional named numeric vector overriding the relative
#'   mutation-kind weights (defaults match PySR's \code{MutationWeights}). Recognised
#'   names: \code{mutate_constant}, \code{mutate_operator}, \code{swap_operands},
#'   \code{rotate_tree}, \code{add_node}, \code{insert_node}, \code{delete_node},
#'   \code{do_nothing}, \code{simplify}, \code{randomize}. Absent names keep their
#'   defaults; e.g. \code{c(rotate_tree = 0)} disables tree rotation only.
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
#' @return A list of class \code{"rsymbolic2"} with elements:
#'   \describe{
#'     \item{expression}{Best expression found, as an infix string.}
#'     \item{loss}{Training loss (sum of squared residuals) of the best
#'       expression.}
#'     \item{complexity}{Number of nodes in the expression tree.}
#'     \item{recommended}{Recommended expression chosen from the Pareto front
#'       according to \code{model_selection} (default \code{"best"}: the
#'       accuracy/complexity "knee").}
#'     \item{best_index}{Row of \code{pareto_front} the recommendation came from
#'       (1-based); \code{NA} if the front is empty.}
#'     \item{pareto_front}{Data frame of non-dominated (complexity, loss,
#'       expression) trade-offs from accuracy-vs-complexity Pareto front.}
#'     \item{n_features}{Number of input features (columns of \code{X}) used
#'       during fitting; required by \code{\link{predict.rsymbolic2}}.}
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
    population_size       = 27L,
    n_populations         = 31L,
    generations           = 2800L,
    tournament_size       = 15L,
    unary_ops             = c("neg", "exp", "log", "sin", "cos"),
    binary_ops            = c("add", "sub", "mul"),
    max_depth             = 30L,
    max_nodes             = 30L,
    target_loss           = 1e-10,
    simplify              = TRUE,
    crossover_probability = 0.0259,
    seed                  = 0L,
    parsimony             = 0.0,
    adaptive_parsimony_scaling = 1040.0,
    optimize_probability  = 0.14,
    tournament_selection_p     = 0.982,
    should_optimize_constants  = TRUE,
    fraction_replaced_hof      = 0.0614,
    mutation_weights      = NULL,
    early_stop_condition  = 0,
    max_evals             = 0,
    model_selection       = c("best", "accuracy", "score"),
    weights               = NULL,
    timeout_seconds       = 0,
    verbosity             = 0L
) {
    X <- as.matrix(X)
    y <- as.numeric(y)

    if (nrow(X) == 0L) stop("X must have at least one row")
    if (nrow(X) != length(y)) stop("nrow(X) must equal length(y)")
    if (!is.numeric(X)) stop("X must be numeric")

    model_selection <- match.arg(model_selection)

    # Optional per-point weights; NULL => unweighted (passed as a length-0 vector).
    if (is.null(weights)) {
        weights <- numeric(0)
    } else {
        weights <- as.numeric(weights)
        if (length(weights) != length(y))
            stop("weights must have the same length as y")
        if (any(!is.finite(weights)) || any(weights < 0))
            stop("weights must be non-negative and finite")
    }

    # The C++ bridge takes a named numeric vector; NULL means "use PySR defaults".
    if (is.null(mutation_weights)) {
        mutation_weights <- stats::setNames(numeric(0), character(0))
    } else if (is.null(names(mutation_weights)) ||
               any(names(mutation_weights) == "")) {
        stop("mutation_weights must be a named numeric vector")
    }

    result <- symbolic_regression_cpp(
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
        as.double(parsimony),
        as.double(adaptive_parsimony_scaling),
        as.double(tournament_selection_p),
        as.logical(should_optimize_constants),
        as.double(fraction_replaced_hof),
        mutation_weights,
        as.character(model_selection),
        as.double(max_evals),
        as.double(early_stop_condition),
        as.double(weights)
    )
    result$n_features <- ncol(X)
    result
}
