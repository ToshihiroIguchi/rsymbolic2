"""rsymbolic2: native symbolic regression with PySR-compatible defaults.

The search engine is a C++ core (shared with the R package) exposed through a thin
pybind11 bridge. The public entry point is :func:`symbolic_regression`, whose default
arguments are byte-for-byte identical to PySR's documented defaults — only the
*implementation* (a dependency-free C++ engine, no Julia runtime) differs.

Example
-------
>>> import numpy as np
>>> from rsymbolic2 import symbolic_regression
>>> X = np.linspace(-3, 3, 40).reshape(-1, 1)
>>> y = 2.5 * X[:, 0] ** 2 - 1.3
>>> res = symbolic_regression(X, y, unary_ops=["square"],
...                           population_size=200, generations=40, seed=1)
>>> print(res.expression)            # doctest: +SKIP
>>> res.predict(np.array([[0.0], [1.0]]))   # doctest: +SKIP
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ._core import symbolic_regression_cpp

__all__ = ["symbolic_regression", "SymbolicRegressionResult"]
__version__ = "0.1.0"

# Recognised operator names (kept in sync with the C++ bridge parsers).
_UNARY_OPS = {"neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"}
_BINARY_OPS = {"add", "sub", "mul", "div", "pow"}

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


class SymbolicRegressionResult:
    """Result of :func:`symbolic_regression`.

    Attributes
    ----------
    expression : str
        Best (lowest-loss) expression found, as an infix string. Variables are named
        ``x0, x1, ...`` (0-based, matching the column order of ``X``).
    loss : float
        Training loss (sum of squared residuals, or weighted SSE) of ``expression``.
    complexity : int
        Number of nodes in the best expression's tree.
    recommended : str
        Expression chosen from the Pareto front according to ``model_selection``
        (default ``"best"``: the accuracy/complexity "knee"). May differ from
        ``expression``, which is always the lowest-loss member.
    best_index : Optional[int]
        Row of :attr:`pareto_front` the recommendation came from (0-based), or
        ``None`` if the front is empty.
    pareto_front : list[dict]
        Non-dominated ``{"complexity", "loss", "expression"}`` trade-offs, sorted by
        increasing complexity.
    n_features : int
        Number of input features (columns of ``X``) used during fitting.
    feature_names : Optional[list[str]]
        Column names of ``X`` when it is a pandas DataFrame (or otherwise carries
        names), else ``None``. Display-only metadata shown by ``repr()`` as an
        ``x0 = name`` legend; the fitted expression strings stay 0-based
        (``x0, x1, ...``) and :meth:`predict` is unaffected.
    """

    def __init__(
        self,
        raw: dict,
        n_features: int,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.expression: str = raw["expression"]
        self.loss: float = raw["loss"]
        self.complexity: int = raw["complexity"]
        self.recommended: str = raw["recommended"]
        self.best_index: Optional[int] = raw["best_index"]
        pf = raw["pareto_front"]
        self.pareto_front = [
            {"complexity": c, "loss": l, "expression": e}
            for c, l, e in zip(pf["complexity"], pf["loss"], pf["expression"])
        ]
        self.n_features: int = n_features
        self.feature_names: Optional[list] = (
            list(feature_names) if feature_names is not None else None
        )

    def predict(
        self, newdata: ArrayLike, *, expression: Optional[str] = None
    ) -> np.ndarray:
        """Evaluate a fitted expression on new input data.

        Parameters
        ----------
        newdata : array-like, shape (n_samples, n_features)
            New inputs. A 1-D array is treated as a single column. Must have
            :attr:`n_features` columns in the same order as the training ``X``.
        expression : str, optional
            Which expression to evaluate. Defaults to :attr:`recommended` (the
            Pareto "best"). Pass :attr:`expression` for the lowest-loss model, or any
            string from :attr:`pareto_front`.

        Notes
        -----
        ``pow(x, y)`` nodes are rendered as ``x ^ y`` and evaluated with Python's
        ``**`` (NumPy ``power``), which yields ``nan`` for negative bases with
        non-integer exponents. This differs slightly from the safe-pow semantics used
        during training (which returns 0 there); it only matters if the training
        domain included negative bases under a fractional power.
        """
        X = np.atleast_2d(np.asarray(newdata, dtype=float))
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"newdata has {X.shape[1]} column(s) but the model was fitted on "
                f"{self.n_features} feature(s)."
            )
        expr = self.recommended if expression is None else expression
        return _eval_expression(expr, X)

    def __repr__(self) -> str:
        lines = [
            f"<SymbolicRegressionResult: {len(self.pareto_front)} Pareto members, "
            f"n_features={self.n_features}>",
        ]
        if self.feature_names is not None and len(self.feature_names) == self.n_features:
            legend = ", ".join(
                f"x{i} = {name}" for i, name in enumerate(self.feature_names)
            )
            lines.append(f"  variables: {legend}")
        lines += [
            f"  recommended: {self.recommended}",
            f"  best (lowest loss): {self.expression}  "
            f"(loss={self.loss:.6g}, complexity={self.complexity})",
        ]
        if self.pareto_front:
            lines.append("  Pareto front (complexity | loss | expression):")
            for m in self.pareto_front:
                lines.append(
                    f"    {m['complexity']:>3} | {m['loss']:.6g} | {m['expression']}"
                )
        return "\n".join(lines)

    def to_pandas(self):
        """Return the Pareto front as a :class:`pandas.DataFrame` (requires pandas)."""
        import pandas as pd  # imported lazily; pandas is an optional extra

        return pd.DataFrame(self.pareto_front, columns=["complexity", "loss", "expression"])


# Math namespace used by predict(). neg/square are not Python builtins; the rest map to
# NumPy ufuncs so evaluation is vectorised over the input columns.
def _eval_namespace(X: np.ndarray) -> dict:
    ns = {
        "__builtins__": {},
        # %.6g rendering of a constant can emit these tokens; bind them so eval() does
        # not raise NameError (builtins are disabled above for safety).
        "inf": float("inf"),
        "nan": float("nan"),
        "neg": lambda v: -v,
        "square": lambda v: v * v,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "sqrt": np.sqrt,
        "tanh": np.tanh,
        "abs": np.abs,
    }
    for j in range(X.shape[1]):
        ns[f"x{j}"] = X[:, j]
    return ns


def _eval_expression(expr: str, X: np.ndarray) -> np.ndarray:
    # The C++ to_string() renders power as `^`; Python uses `**`. Variables and the
    # function/operator forms are otherwise valid Python.
    py_expr = expr.replace("^", "**")
    value = eval(py_expr, _eval_namespace(X))  # noqa: S307 — expr is engine-generated
    return np.broadcast_to(np.asarray(value, dtype=float), (X.shape[0],)).copy()


def symbolic_regression(
    X: ArrayLike,
    y: ArrayLike,
    *,
    population_size: int = 27,
    n_populations: int = 31,
    generations: int = 2800,
    tournament_size: int = 15,
    unary_ops: Sequence[str] = ("neg", "exp", "log", "sin", "cos"),
    binary_ops: Sequence[str] = ("add", "sub", "mul"),
    max_depth: int = 30,
    max_nodes: int = 30,
    target_loss: float = 1e-10,
    simplify: bool = True,
    crossover_probability: float = 0.0259,
    seed: int = 0,
    parsimony: float = 0.0,
    adaptive_parsimony_scaling: float = 1040.0,
    optimize_probability: float = 0.14,
    tournament_selection_p: float = 0.982,
    should_optimize_constants: bool = True,
    fraction_replaced_hof: float = 0.0614,
    mutation_weights: Optional[Mapping[str, float]] = None,
    early_stop_condition: float = 0.0,
    max_evals: float = 0,
    model_selection: str = "best",
    weights: Optional[ArrayLike] = None,
    batching: bool = False,
    batch_size: int = 50,
    warmup_maxsize_by: float = 0.0,
    timeout_seconds: float = 0.0,
    verbosity: int = 1,
) -> SymbolicRegressionResult:
    """Discover a mathematical expression that fits ``y`` from ``X``.

    Uses steady-state genetic programming with Levenberg-Marquardt constant
    optimisation. Every default below is identical to PySR's documented default; only
    the implementation differs (a dependency-free C++ engine, no Julia runtime). See
    the README "Algorithm" section for the full method and references.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features. A 1-D array is treated as a single column.
    y : array-like, shape (n_samples,)
        Target values; ``len(y)`` must equal ``X.shape[0]``.
    population_size : int, default 27
        Candidate expressions per island (PySR ``population_size``).
    n_populations : int, default 31
        Number of island populations evolved in parallel with ring migration (PySR
        ``populations``). >1 enables OpenMP parallelism; islands still run (serially)
        when OpenMP is unavailable.
    generations : int, default 2800
        Evolution generations. One generation performs ``population_size``
        tournament-and-replace steps. The default reproduces PySR's per-population
        mutation budget (``niterations=100`` x 28; see README / docs/28).
    tournament_size : int, default 15
        Tournament size for selection/replacement (PySR ``tournament_selection_n``).
    unary_ops : sequence of str, default ("neg","exp","log","sin","cos")
        Allowed unary operators. Recognised: neg, exp, log, sin, cos, sqrt, tanh,
        abs, square. (PySR ships no default operator set; this is the shared problem
        input, given identically to both tools.)
    binary_ops : sequence of str, default ("add","sub","mul")
        Allowed binary operators. Recognised: add, sub, mul, div, pow.
    max_depth : int, default 30
        Maximum tree depth (PySR ``maxdepth``).
    max_nodes : int, default 30
        Soft upper bound on tree size (PySR ``maxsize``); also sizes the
        adaptive-parsimony histogram.
    target_loss : float, default 1e-10
        Early stop once the best training loss falls below this.
    simplify : bool, default True
        Algebraically simplify fitted candidates.
    crossover_probability : float, default 0.0259
        Probability of subtree crossover vs. mutation per step (PySR
        ``crossover_probability``).
    seed : int, default 0
        Random seed; 0 uses a non-deterministic seed.
    parsimony : float, default 0.0
        Fixed linear complexity penalty (PySR ``parsimony``; off by default — the
        adaptive term below carries the size pressure).
    adaptive_parsimony_scaling : float, default 1040.0
        Strength of frequency-based adaptive parsimony (PySR's installed default;
        0 disables it).
    optimize_probability : float, default 0.14
        Probability a population member is LM-optimised each iteration (PySR
        ``optimize_probability``).
    tournament_selection_p : float, default 0.982
        Probabilistic tournament strength (PySR's installed default); 1.0 is a
        deterministic best-of-k tournament.
    should_optimize_constants : bool, default True
        Run the once-per-iteration constant-optimisation pass (PySR
        ``should_optimize_constants``).
    fraction_replaced_hof : float, default 0.0614
        Hall-of-fame migration fraction (PySR's installed default); 0 disables it.
    mutation_weights : mapping of str to float, optional
        Override relative mutation-kind weights. Recognised keys: mutate_constant,
        mutate_operator, swap_operands, rotate_tree, add_node, insert_node,
        delete_node, do_nothing, simplify, randomize. Absent keys keep PySR defaults.
    early_stop_condition : float, default 0.0
        Additional early-stop loss threshold (PySR ``early_stop_condition``, numeric
        form); 0 = off.
    max_evals : float, default 0
        Cap on total candidate evaluations across islands (PySR ``max_evals``);
        0 = no limit. Deterministic and thread-count independent.
    model_selection : {"best", "accuracy", "score"}, default "best"
        Which Pareto member is reported as ``recommended`` (PySR ``model_selection``).
    weights : array-like, optional
        Per-point non-negative weights for a weighted least-squares fit (PySR
        ``weights``); None fits unweighted.
    batching : bool, default False
        Evaluate the evolution and constant-optimisation passes on a random subsample
        of ``batch_size`` rows per iteration instead of the full dataset (PySR
        ``batching``) — the lever for large row counts, making each candidate
        evaluation cost ``O(batch_size)`` rather than ``O(len(y))``. The hall of fame,
        early-stop test and the reported result are always computed on the full
        dataset, so batching changes only which candidates are explored, never the
        accuracy attributed to a returned model. Rows are sampled with replacement and
        re-sampled each iteration. Fewer than ~10,000 rows are usually enough without it.
    batch_size : int, default 50
        Rows sampled per iteration when ``batching`` is True (PySR ``batch_size``).
        Must be >= 1; values larger than ``len(y)`` are clamped to ``len(y)``. Ignored
        when ``batching`` is False.
    warmup_maxsize_by : float, default 0.0
        Fraction of the run over which the maximum expression size grows linearly from
        3 up to ``max_nodes``, then stays there (PySR ``warmup_maxsize_by``). 0 (default)
        disables the ramp, so the size cap is ``max_nodes`` throughout (PySR's default).
        E.g. 0.5 reaches ``max_nodes`` halfway through the run, biasing the early search
        toward small expressions. Must be a finite number >= 0. Only the
        mutation/crossover size cap ramps; the initial population is drawn at ``max_nodes``.
    timeout_seconds : float, default 0.0
        Wall-clock limit; 0 = no limit. A timed-out run is not reproducible across
        machines (only runs that finish within budget are bit-reproducible).
    verbosity : int, default 1
        Default 1 matches PySR's default (``verbosity=1``), printing one
        diagnostic line per epoch to stderr; 0 = silent. The line is emitted by
        the C++ core to ``stderr``; redirect the process ``stderr`` to log it.
        The compact one-liner rendering differs from PySR's live table; only the
        on/off default is matched.

    Returns
    -------
    SymbolicRegressionResult
        Best expression, Pareto front, and a :meth:`~SymbolicRegressionResult.predict`
        method.
    """
    # Capture display-only column names before coercion (pandas DataFrame carries
    # them in `.columns`). They are surfaced in repr() as an `x0 = name` legend and
    # never fed back into the evaluable expression strings, which stay 0-based.
    columns = getattr(X, "columns", None)
    feature_names = [str(c) for c in columns] if columns is not None else None

    X_arr = np.atleast_2d(np.asarray(X, dtype=float))
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).ravel()
    if X_arr.shape[0] == 0:
        raise ValueError("X must have at least one row.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X.shape[0] must equal len(y).")

    unary_list = [str(s) for s in unary_ops]
    binary_list = [str(s) for s in binary_ops]
    for s in unary_list:
        if s not in _UNARY_OPS:
            raise ValueError(
                f"Unknown unary operator: {s!r}. Choose from {sorted(_UNARY_OPS)}."
            )
    for s in binary_list:
        if s not in _BINARY_OPS:
            raise ValueError(
                f"Unknown binary operator: {s!r}. Choose from {sorted(_BINARY_OPS)}."
            )
    if not binary_list:
        raise ValueError("binary_ops must contain at least one operator.")

    if model_selection not in ("best", "accuracy", "score"):
        raise ValueError(
            "model_selection must be 'best', 'accuracy', or 'score'; "
            f"got {model_selection!r}."
        )

    if weights is None:
        weights_arr = np.empty(0, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float).ravel()
        if weights_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("weights must have the same length as y.")
        if not np.all(np.isfinite(weights_arr)) or np.any(weights_arr < 0):
            raise ValueError("weights must be non-negative and finite.")

    mw_dict = {} if mutation_weights is None else {str(k): float(v) for k, v in mutation_weights.items()}

    if int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer.")

    if not np.isfinite(warmup_maxsize_by) or float(warmup_maxsize_by) < 0:
        raise ValueError("warmup_maxsize_by must be a finite number >= 0.")

    raw = symbolic_regression_cpp(
        X_arr,
        y_arr,
        int(population_size),
        int(generations),
        int(tournament_size),
        unary_list,
        binary_list,
        int(max_depth),
        int(max_nodes),
        float(target_loss),
        bool(simplify),
        float(crossover_probability),
        float(seed),
        int(n_populations),
        float(timeout_seconds),
        int(verbosity),
        float(optimize_probability),
        float(parsimony),
        float(adaptive_parsimony_scaling),
        float(tournament_selection_p),
        bool(should_optimize_constants),
        float(fraction_replaced_hof),
        mw_dict,
        str(model_selection),
        float(max_evals),
        float(early_stop_condition),
        weights_arr,
        bool(batching),
        int(batch_size),
        float(warmup_maxsize_by),
    )
    if feature_names is not None and len(feature_names) != int(X_arr.shape[1]):
        feature_names = None  # shape changed (e.g. 1-D promoted); drop mismatched names
    return SymbolicRegressionResult(
        raw, n_features=int(X_arr.shape[1]), feature_names=feature_names
    )
