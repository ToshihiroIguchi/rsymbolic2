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
    expression_simplified : Optional[str]
        Display-only, algebraically-simplified rewrite of ``expression`` (docs/52) —
        e.g. collapses a chain of constant multiplications/divisions into one constant.
        Never used by :meth:`predict`; ``expression`` remains the evaluatable
        round-trip source (docs/48 D2).
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
        Non-dominated ``{"complexity", "loss", "score", "r_squared", "expression",
        "latex", "expression_simplified", "latex_simplified"}`` trade-offs, sorted by
        increasing complexity. ``score`` is the drop in log-loss per unit of added
        complexity relative to the next-simpler member — the value ``model_selection``
        ranks by; ``0.0`` for the simplest member. ``r_squared`` is the training
        ``1 - loss / sst`` (``None`` when the target is constant). ``latex`` is a
        display-only LaTeX rendering of the expression (variables as ``x_{i}``); see
        :meth:`latex`. ``expression_simplified``/``latex_simplified`` are a further
        algebraically-simplified (display-only) rewrite of ``expression``/``latex``
        (docs/52) — never used by :meth:`predict`, which always evaluates the frozen
        ``expression``/``recommended`` strings.
    n_obs : Optional[int]
        Number of training observations (rows of ``X``).
    sst : Optional[float]
        Total sum of squares of ``y`` about its (weighted) mean on the training
        data. Basis for the per-member ``r_squared`` (``1 - loss / sst``),
        which is consistent with the (weighted) SSE ``loss``.
    n_evals : Optional[int]
        Total number of candidate evaluations spent by the search, in
        ``max_evals`` units: forward-pass loss evaluations plus the residual
        evaluations consumed by constant-optimisation fits, summed across
        islands. Deterministic for a fixed seed.
    eval_counts : Optional[dict]
        Breakdown of :attr:`n_evals` with keys ``forward`` (forward-pass loss
        evaluations), ``lm_resid`` (Levenberg-Marquardt residual evaluations;
        ``forward + lm_resid == n_evals``), ``lm_jac`` (LM Jacobian builds,
        reported for accounting only — never charged to ``n_evals`` or the
        ``max_evals`` budget), and ``cache_hits``/``cache_misses``
        (duplicate-evaluation cache statistics; both 0 unless ``eval_cache=True``.
        A hit is still counted in ``forward``, so ``cache_hits + cache_misses``
        is the number of forward passes routed through the cache, not extra work),
        and ``strong_simplify_attempts``/``strong_simplify_adopted``
        (search-time strong-simplification statistics; both 0 unless
        ``strong_simplify=True``. ``strong_simplify_adopted`` is always
        ``<= strong_simplify_attempts``).
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
        # Display-only companion to `expression` (docs/52): a shorter/more-readable
        # algebraic rewrite. None when the raw dict predates this field. Never used by
        # predict() (docs/48 D2 frozen-expression rule: `expression` stays the
        # evaluatable round-trip source).
        self.expression_simplified: Optional[str] = raw.get("expression_simplified")
        self.loss: float = raw["loss"]
        self.complexity: int = raw["complexity"]
        self.recommended: str = raw["recommended"]
        self.best_index: Optional[int] = raw["best_index"]
        self.n_obs: Optional[int] = raw.get("n_obs")
        self.sst: Optional[float] = raw.get("sst")
        # Evaluation accounting (None when the raw dict predates these fields).
        self.n_evals: Optional[int] = raw.get("n_evals")
        self.eval_counts: Optional[dict] = raw.get("eval_counts")
        # Training R^2 per member: 1 - loss/sst. None when the target was constant
        # (sst == 0) or the raw dict predates the sst field. Negative values are
        # valid (a fit worse than the mean).
        has_sst = self.sst is not None and np.isfinite(self.sst) and self.sst > 0
        pf = raw["pareto_front"]
        # expression_simplified/latex_simplified (docs/52) are display-only companions
        # to expression/latex, parallel arrays of the same length; None per-member when
        # the raw dict predates these fields (an older compiled extension).
        n_pf = len(pf["complexity"])
        pf_expr_simplified = pf.get("expression_simplified", [None] * n_pf)
        pf_latex_simplified = pf.get("latex_simplified", [None] * n_pf)
        self.pareto_front = [
            {
                "complexity": c,
                "loss": l,
                "score": s,
                "r_squared": (1.0 - l / self.sst) if has_sst else None,
                "expression": e,
                "latex": t,
                "expression_simplified": es,
                "latex_simplified": ts,
            }
            for c, l, s, e, t, es, ts in zip(
                pf["complexity"], pf["loss"], pf["score"], pf["expression"],
                pf["latex"], pf_expr_simplified, pf_latex_simplified,
            )
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

    def get_best(self, index: Optional[int] = None) -> dict:
        """Return a Pareto-front member as a dict.

        Parameters
        ----------
        index : int, optional
            0-based row of :attr:`pareto_front`. Defaults to :attr:`best_index`,
            the member chosen by ``model_selection`` (mirrors PySR's
            ``get_best``).
        """
        if not self.pareto_front:
            raise ValueError("get_best() called on an empty Pareto front.")
        if index is None:
            index = self.best_index
        if not 0 <= index < len(self.pareto_front):
            raise IndexError(
                f"index {index} is out of range for a Pareto front with "
                f"{len(self.pareto_front)} member(s)."
            )
        return self.pareto_front[index]

    def latex(
        self,
        index: Optional[int] = None,
        variable_names: Optional[Sequence[str]] = None,
    ) -> str:
        """Return a Pareto-front member as LaTeX math (no surrounding ``$``).

        Rendered by the C++ core with minimal parentheses (``\\frac`` for
        division, ``\\cdot``, ``\\sqrt``, ...). Display-only: :meth:`predict`
        keeps using the plain ``expression`` strings.

        Parameters
        ----------
        index : int, optional
            0-based row of :attr:`pareto_front`. Defaults to :attr:`best_index`,
            the member chosen by ``model_selection``.
        variable_names : sequence of str, optional
            Names substituted for the ``x_{i}`` tokens (underscores are escaped
            for LaTeX). Defaults to :attr:`feature_names` when set; pass an
            explicit list to override, or ``[]`` to force the ``x_{i}`` form.
        """
        member = self.get_best(index)
        out = member["latex"]
        names = variable_names if variable_names is not None else self.feature_names
        if names:
            if len(names) != self.n_features:
                raise ValueError(
                    f"variable_names has {len(names)} name(s) but the model was "
                    f"fitted on {self.n_features} feature(s)."
                )
            for i, name in enumerate(names):
                out = out.replace(f"x_{{{i}}}", str(name).replace("_", "\\_"))
        return out

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
        if self.pareto_front and self.best_index is not None:
            r2 = self.pareto_front[self.best_index].get("r_squared")
            if r2 is not None:
                lines.append(f"  R-squared (recommended): {r2:.6g}")
        if self.pareto_front:
            lines.append(
                "  Pareto front (> = recommended; "
                "complexity | loss | score | expression):"
            )
            lines += self._format_pareto_lines()
        return "\n".join(lines)

    def _format_pareto_lines(self, max_rows: int = 20) -> list:
        # Mirrors the R package's format_pareto_lines(): a ">" marker on the
        # recommended row, and head/tail with an elided middle when the front is
        # longer than max_rows.
        rows = []
        for i, m in enumerate(self.pareto_front):
            marker = ">" if i == self.best_index else " "
            rows.append(
                f"  {marker} {m['complexity']:>3} | {m['loss']:.6g} | "
                f"{m['score']:.6g} | {m['expression']}"
            )
        n = len(rows)
        if n <= max_rows:
            return rows
        head = max_rows // 2
        tail = max_rows - head
        return rows[:head] + [f"      ... ({n - max_rows} more) ..."] + rows[n - tail:]

    def to_pandas(self):
        """Return the Pareto front as a :class:`pandas.DataFrame` (requires pandas)."""
        import pandas as pd  # imported lazily; pandas is an optional extra

        return pd.DataFrame(
            self.pareto_front, columns=["complexity", "loss", "score", "expression"]
        )

    def plot(self, *, log_loss: bool = True, label_exprs: bool = True, ax=None):
        """Plot the Pareto front as a complexity vs. loss scatter (requires matplotlib).

        The Python counterpart of the R package's ``plot.rsymbolic2()``: a line
        through the non-dominated members with the lowest-loss point highlighted.

        Parameters
        ----------
        log_loss : bool
            Use a log scale for the loss axis (skipped automatically when any
            loss is zero). Default ``True``.
        label_exprs : bool
            Annotate each point with its expression string. Set to ``False``
            for large fronts where labels overlap. Default ``True``.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into. A new figure is created when omitted.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into (nothing is shown; call ``plt.show()`` yourself).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # matplotlib is the optional "plot" extra
            raise ImportError(
                "matplotlib is required for plot(); install it with: "
                "pip install rsymbolic2[plot]"
            ) from e

        if ax is None:
            _, ax = plt.subplots()
        complexity = [m["complexity"] for m in self.pareto_front]
        loss = [m["loss"] for m in self.pareto_front]
        ax.plot(complexity, loss, color="0.6", zorder=1)
        ax.scatter(complexity, loss, s=36, color="black", zorder=2)
        if loss:
            i = int(np.argmin(loss))
            ax.scatter([complexity[i]], [loss[i]], s=64, color="firebrick", zorder=3)
        if log_loss and loss and all(l > 0 for l in loss):
            ax.set_yscale("log")
        if label_exprs:
            for m in self.pareto_front:
                ax.annotate(
                    m["expression"], (m["complexity"], m["loss"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8,
                )
        ax.set_xlabel("Complexity (nodes)")
        ax.set_ylabel("Loss (SSE)")
        ax.set_title("rsymbolic2 Pareto front")
        return ax


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
    n_threads: Optional[int] = None,
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
    eval_cache: bool = False,
    linear_scaling: bool = False,
    strong_simplify: bool = False,
    X_units: Optional[Sequence[str]] = None,
    y_units: Optional[str] = None,
    dimensional_constraint_penalty: Optional[float] = None,
    dimensionless_constants_only: bool = False,
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
    n_threads : int, optional
        OpenMP worker threads for the island-parallel search. ``None`` (default) uses
        every available core (the OpenMP default, overridable with the ``OMP_NUM_THREADS``
        environment variable); a positive integer requests exactly that many. The team is
        capped internally at ``n_populations``. Pure wall-clock knob: the island model is
        bit-deterministic across thread counts, so ``n_threads`` changes only speed, never
        the result. No effect when built without OpenMP. More logical (hyper-threaded) cores
        finish sooner than restricting to physical cores, so the default is "all cores", not
        "physical cores" (see docs/37).
    generations : int, default 2800
        Evolution generations. One generation performs ``population_size``
        tournament-and-replace steps. The default reproduces PySR's per-population
        mutation budget (``niterations=100`` x 28; see README / docs/28).
        Raising ``generations`` is the sanctioned opt-in accuracy lever: recovery
        scales with budget on trajectory-limited problems (measured at 5x budget:
        newtons_grav 0 -> 0.9, center_mass 0.3 -> 0.8, boltzmann_dist 0.1 -> 0.4,
        interference 0 -> 0.27 structurally-verified recovery; some problems do
        not respond — docs/44, docs/47). Compute cost grows linearly; the default
        stays at PySR parity.
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
    eval_cache : bool, default False
        Enable an opt-in duplicate-evaluation cache. Implementation-only memoisation:
        each island keeps a small fixed-size table of recently evaluated expression
        trees and reuses the stored loss when an evaluation-identical tree recurs,
        instead of re-evaluating it. Results are bit-identical with the cache on or
        off (a hit is charged to ``n_evals``/``eval_counts`` exactly like a real
        evaluation, so even ``max_evals``-budgeted runs are unchanged) — a speed
        knob, never a search setting; PySR parity is unaffected. Ignored when
        ``batching=True`` (each iteration's random subsample makes cached losses
        unreusable). The ``cache_hits``/``cache_misses`` entries of ``eval_counts``
        report its effectiveness.
    linear_scaling : bool, default False
        Enable Keijzer (2003) linear scaling. An opt-in high-accuracy option that
        deliberately diverges from PySR (which has no such mechanism): every
        candidate is scored by the sum of squared errors of its best affine
        transform ``a*f(x) + b``, with the slope ``a`` and intercept ``b`` solved in
        closed form (the weighted least squares of ``y`` on the candidate's
        predictions), so the search only has to discover the *shape* of the target,
        never its scale or offset. The fitted ``a`` and ``b`` are materialised into
        every returned expression as ``((f * a) + b)`` — skipped when they equal the
        identity to numerical precision — so ``expression``, ``loss``,
        ``complexity`` and :meth:`~SymbolicRegressionResult.predict` stay
        self-consistent; the wrap may push a returned expression up to 4 nodes past
        ``max_nodes``. Not compatible with dimensional analysis
        (``X_units``/``y_units``). The default False keeps the search at exact PySR
        parity.
    strong_simplify : bool, default False
        Enable search-time strong simplification. An opt-in high-accuracy option
        (the project's second layer): it deliberately diverges from PySR, which has
        no such mechanism, but defaults to False so the search stays at exact
        PySR-parity behaviour. When enabled, applies docs/54's display simplifier to
        candidates during the search itself (not just for display) under a small
        deterministic budget, and adopts the simplified form only when it is
        strictly smaller *and* stays within the search's enabled operator set (e.g.
        a simplification that introduces ``neg``, ``square``, or ``abs`` is rejected
        unless that operator is already enabled, so simplification never grows the
        effective search space). See docs/55 for the accuracy evidence backing this
        option.
    X_units : sequence of str, optional
        Units for each column of ``X``, enabling dimensional analysis (PySR ``X_units``).
        Each entry is a DynamicQuantities-style unit string such as ``"m/s^2"``, ``"kg"``,
        or ``"1"`` (dimensionless). SI base units, common derived units (N, J, W, Pa, C, V,
        Ohm, T, Hz, ...), decimal prefixes and ``* / ^ ( )`` are supported. ``None``
        (default) disables dimensional analysis, leaving the search identical to PySR's.
    y_units : str, optional
        Unit for the target ``y`` (PySR ``y_units``). Requires ``X_units``. Expressions
        whose output dimension differs are penalised. ``None`` (default) leaves the output
        dimension unconstrained.
    dimensional_constraint_penalty : float, optional
        Penalty added to the loss of a dimensionally inconsistent expression (PySR
        ``dimensional_constraint_penalty``). ``None`` (default) uses PySR's effective
        default of 1000. Inert unless ``X_units``/``y_units`` are set.
    dimensionless_constants_only : bool, default False
        If True, fitted constants are treated as dimensionless during dimensional analysis
        instead of adopting whatever dimension keeps the expression consistent (PySR
        ``dimensionless_constants_only``).
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

    # OpenMP team size. None (default) => 0 = auto (all cores, honouring OMP_NUM_THREADS);
    # a positive integer caps the worker threads. The core caps it at n_populations.
    if n_threads is None:
        n_threads_val = 0
    else:
        n_threads_val = int(n_threads)
        if n_threads_val < 1:
            raise ValueError("n_threads must be None or a positive integer.")

    # Opt-in dimensional analysis (PySR X_units / y_units / dimensional_constraint_penalty /
    # dimensionless_constants_only; docs/46). All default-off: with X_units=None the search is
    # unchanged. Units are DynamicQuantities-style strings parsed by the shared C++ core.
    if X_units is None:
        x_units_list = []
    else:
        x_units_list = [str(u) for u in X_units]
        if len(x_units_list) != int(X_arr.shape[1]):
            raise ValueError(
                f"X_units must have length n_features (= {int(X_arr.shape[1])}); "
                f"got {len(x_units_list)}."
            )
    if y_units is None:
        y_units_str = ""
    else:
        if not isinstance(y_units, str):
            raise ValueError("y_units must be a single unit string.")
        y_units_str = y_units
        if not x_units_list:
            raise ValueError("y_units requires X_units to be specified.")
    # Opt-in Keijzer-2003 linear scaling: refits (and materialises) a free affine
    # transform of every candidate, which has no defined dimensional semantics; the
    # combination is rejected here (bindings own validation; the core never sees it).
    if linear_scaling and (x_units_list or y_units_str):
        raise ValueError(
            "linear scaling is not supported with dimensional analysis "
            "(X_units/y_units)."
        )
    # PySR's signature default None maps to an effective penalty of 1000.0.
    if dimensional_constraint_penalty is None:
        dcp = 1000.0
    else:
        dcp = float(dimensional_constraint_penalty)
        if not np.isfinite(dcp) or dcp < 0:
            raise ValueError(
                "dimensional_constraint_penalty must be a finite number >= 0."
            )

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
        int(n_threads_val),
        x_units_list,
        y_units_str,
        float(dcp),
        bool(dimensionless_constants_only),
        bool(eval_cache),
        bool(linear_scaling),
        bool(strong_simplify),
    )
    if feature_names is not None and len(feature_names) != int(X_arr.shape[1]):
        feature_names = None  # shape changed (e.g. 1-D promoted); drop mismatched names
    return SymbolicRegressionResult(
        raw, n_features=int(X_arr.shape[1]), feature_names=feature_names
    )
