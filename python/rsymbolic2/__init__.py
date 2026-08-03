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

import ast
import inspect
import math
import re
import warnings
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ._core import symbolic_regression_cpp

__all__ = ["symbolic_regression", "SymbolicRegressionResult", "SymbolicRegressor"]
__version__ = "0.1.0"

# Recognised operator names (kept in sync with the C++ bridge parsers). `_UNARY_OPS` is
# the single source of truth for both jobs it serves: validating the caller's `unary_ops`
# and recognising the call form the core prints when drawing an equation tree. Defining it
# twice would let the two drift silently, so both read this one set.
_UNARY_OPS = {"neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square", "inv",
              "erf", "sinh", "cosh"}
_BINARY_OPS = {"add", "sub", "mul", "div", "pow"}

# Row count above which an unbatched run is worth warning about. Not a limit and not a
# measured cliff: it is the figure the batching documentation already names ("for most
# problems fewer than ~10,000 rows are enough without batching"), kept in one place so
# the prose and the warning cannot drift apart.
_LARGE_DATA_ROWS = 10000


def _column_names(a) -> Optional[list]:
    """Column names of a labelled tabular input (pandas DataFrame), else None.

    Duck-typed on `.columns` so pandas stays an optional extra: rsymbolic2 never imports
    it. Anything without column labels — an ndarray, a list of lists — has no names to
    check against, and returns None rather than inventing any.
    """
    columns = getattr(a, "columns", None)
    return None if columns is None else [str(c) for c in columns]


def _is_numeric_dtype(dtype) -> bool:
    """Whether a column dtype holds numbers, pandas extension dtypes included.

    `np.dtype(dtype)` raises TypeError on pandas' own dtypes (StringDtype, Categorical,
    nullable Int64), so the numpy test alone cannot answer this. pandas is imported
    lazily here — it stays an optional extra, and this line is only ever reached because
    the caller already handed us one of its objects (the same idiom as `to_pandas()`).
    """
    try:
        from pandas.api.types import is_numeric_dtype  # optional extra, already in use

        return bool(is_numeric_dtype(dtype))
    except ImportError:
        pass
    try:
        np_dtype = np.dtype(dtype)
    except TypeError:
        return False  # an extension dtype numpy cannot even name is not a numeric one
    return bool(np.issubdtype(np_dtype, np.number) or np.issubdtype(np_dtype, np.bool_))


def _non_numeric_columns(a) -> list:
    """Names of a DataFrame's non-numeric columns, for an error message that names them.

    Returns an empty list for anything that is not a labelled table, in which case the
    caller falls back to the blanket message: there is nothing more specific to say.
    """
    names = _column_names(a)
    dtypes = getattr(a, "dtypes", None)
    if names is None or dtypes is None:
        return []
    return [name for name, dtype in zip(names, list(dtypes))
            if not _is_numeric_dtype(dtype)]


def _as_design_matrix(a, name: str, *, require_finite: bool = False) -> np.ndarray:
    """Coerce feature input to a float (n_samples, n_features) matrix.

    A 1-D input is one *column* (n samples of a single feature) — the reading the R
    package's `as.matrix()` gives, and the one the single-feature case that dominates
    symbolic regression wants. It is deliberately never read as one row of several
    features: guessing between the two from the shape alone is how a silently wrong
    prediction gets made, so a caller with a single multi-feature sample passes an
    explicit `(1, n_features)` array.

    `require_finite` applies scikit-learn's `check_array` rule at the point of coercion.
    It is on for prediction inputs and off for the training path, which runs its own
    finiteness check after the shape checks so that a shape mistake is reported as one.
    """
    try:
        arr = np.asarray(a, dtype=float)
    except (TypeError, ValueError) as exc:
        # np.asarray's own message names the offending *value* ("could not convert string
        # to float: 'red'") but not the column it sits in, which is what the caller has to
        # go and fix. Name the columns when the input carries labels (docs/81 P3).
        bad = _non_numeric_columns(a)
        if bad:
            raise ValueError(
                f"{name} must be numeric; non-numeric column(s): {', '.join(bad)}. "
                "rsymbolic2 discovers transformations itself and does not dummy-code "
                "categorical inputs, because that would enlarge the search space with "
                "columns you did not choose. Encode them explicitly first, e.g. "
                f"pandas.get_dummies({name}, columns={bad!r})."
            ) from exc
        raise ValueError(f"{name} must be numeric: {exc}") from exc
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim != 2:
        raise ValueError(
            f"{name} must be 1-D (a single column) or 2-D (n_samples, n_features); "
            f"got a {arr.ndim}-D array."
        )
    if require_finite and not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{name} must not contain NaN or infinite values. Drop the incomplete rows "
            f"first (for example {name}.dropna() for a DataFrame, or "
            f"{name}[np.isfinite({name}).all(axis=1)] for an array)."
        )
    return arr


def _check_feature_names(fitted: Optional[Sequence[str]], newdata, name: str) -> None:
    """Refuse a feature-name mismatch between the fit and prediction inputs (docs/81 P1).

    Only fires when *both* sides carry names: the fit captured them from a DataFrame, and
    `newdata` is a DataFrame too. Then the lists must be equal, in order.

    It refuses rather than reordering. Reordering by name is R's rule, and it follows from
    the R formula interface binding columns by name; Python's fit binds by *position* —
    `feature_names` is metadata captured on the way past, not the thing the model was
    built on — so permuting the caller's columns would invent a guarantee the fit never
    made. scikit-learn refuses here for the same reason.

    When only one side has names nothing is checked and nothing is said. scikit-learn
    warns in that case; staying quiet is a deliberate divergence, because
    `predict(X_test.to_numpy())` is an ordinary idiom and there is nothing to verify
    either way.
    """
    if fitted is None:
        return
    supplied = _column_names(newdata)
    if supplied is None or list(supplied) == list(fitted):
        return
    if sorted(supplied) == sorted(fitted):
        detail = ("the same names in a different order: pass the columns in the fitted "
                  "order, e.g. " f"{name}[{list(fitted)!r}]")
    else:
        missing = [n for n in fitted if n not in supplied]
        extra = [n for n in supplied if n not in fitted]
        detail = "; ".join(
            part for part in (
                f"missing: {missing}" if missing else "",
                f"unexpected: {extra}" if extra else "",
            ) if part
        )
    raise ValueError(
        f"{name} feature names do not match those seen during fitting.\n"
        f"  fitted:   {list(fitted)}\n"
        f"  supplied: {list(supplied)}\n"
        f"  {detail}"
    )


def _as_target_vector(a, name: str) -> np.ndarray:
    """Coerce the target to a 1-D float vector, refusing a multi-output target.

    A `(n, 1)` column vector is the same single target written differently, so it is
    flattened silently. Anything wider is not: `ravel()` on an `(n, k)` array interleaves
    k unrelated series into one vector of length n*k, and when that length happens to
    match `nrow(X)` every length check downstream passes and the search dutifully fits the
    interleaving (docs/80). rsymbolic2 fits one target; several are several runs.
    """
    arr = np.asarray(a, dtype=float)
    if arr.ndim > 1 and int(np.prod(arr.shape[1:])) > 1:
        raise ValueError(
            f"{name} must be a single target: a 1-D array, or 2-D with one column; got "
            f"shape {arr.shape}. Fit one column at a time rather than passing several."
        )
    return arr.ravel()


def _check_degenerate_data(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
    """Warn about the three exact, checkable degeneracies a dataset can carry (docs/80).

    No invented thresholds: each condition is either true or false of the data, and each
    one makes a specific reported number meaningless. Warnings rather than errors — the
    run is well-defined and a wide table with a constant column is ordinary — but nothing
    else the caller sees says the answer does not mean what it looks like.

    `w` is the already-validated weight vector (empty = unweighted); the weighted mean and
    SST below use the same formula `compute_y_norm()` uses in the core, so a dataset that
    is constant under the core's own weighting is the one named.
    """
    weights = w if w.size == y.size else np.ones_like(y)
    with np.errstate(over="ignore", invalid="ignore"):
        mu = float(np.sum(weights * y) / np.sum(weights))
        sst = float(np.sum(weights * (y - mu) ** 2))

    if not math.isfinite(sst):
        # The data itself is finite (checked by the caller) but its squares are not, so the
        # SSE loss and everything derived from it (R^2, score) overflow. Rescaling y costs
        # nothing: symbolic regression is not scale-bound.
        warnings.warn(
            "y's scale overflows the sum-of-squares loss (its total sum of squares is not "
            "finite), so the reported loss and R-squared are meaningless. Rescale y (for "
            "example y / np.max(np.abs(y))) before fitting.",
            UserWarning,
            stacklevel=3,
        )
    elif sst <= 0.0:
        # Zero variance: every constant fits perfectly, so no expression is better than any
        # other and R^2 (1 - loss/sst) is undefined — which is why `r_squared` is None. This
        # says why. A single-row dataset lands here too, by construction.
        warnings.warn(
            "y is constant (zero variance), so no expression can explain it and R-squared "
            "is undefined. The search will return an arbitrary constant.",
            UserWarning,
            stacklevel=3,
        )

    # A constant feature cannot explain any variation in y, so it only enlarges the search
    # space. Named "x0" because the fitted expression strings are 0-based.
    const = np.flatnonzero(np.all(X == X[0], axis=0))
    if const.size:
        names = ", ".join(f"x{j}" for j in const)
        verb = "is" if const.size == 1 else "are"
        warnings.warn(
            f"feature{'' if const.size == 1 else 's'} {names} {verb} constant and cannot "
            "explain any variation in y; dropping "
            f"{'that column' if const.size == 1 else 'those columns'} shrinks the search "
            "space.",
            UserWarning,
            stacklevel=3,
        )

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
        "latex", "sympy", "expression_simplified", "latex_simplified",
        "sympy_simplified"}`` trade-offs, sorted by
        increasing complexity. ``score`` is the drop in log-loss per unit of added
        complexity relative to the next-simpler member — the value ``model_selection``
        ranks by; ``0.0`` for the simplest member. ``r_squared`` is the training
        ``1 - loss / sst`` (``None`` when the target is constant). ``latex`` is a
        display-only LaTeX rendering of the expression (variables as ``x_{i}``); see
        :meth:`latex`. ``sympy`` is display-only Python source that SymPy's
        ``sympify()`` parses; see :meth:`sympy`. ``expression_simplified``/
        ``latex_simplified``/``sympy_simplified`` are those three renderings of a
        further algebraically-simplified (display-only) rewrite (docs/52) — never
        used by :meth:`predict`, which always evaluates the frozen
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
        # sympy/sympy_simplified are the same kind of display-only companion, added
        # later; .get() keeps an older compiled extension loadable.
        pf_sympy = pf.get("sympy", [None] * n_pf)
        pf_sympy_simplified = pf.get("sympy_simplified", [None] * n_pf)
        self.pareto_front = [
            {
                "complexity": c,
                "loss": l,
                "score": s,
                "r_squared": (1.0 - l / self.sst) if has_sst else None,
                "expression": e,
                "latex": t,
                "sympy": y,
                "expression_simplified": es,
                "latex_simplified": ts,
                "sympy_simplified": ys,
            }
            for c, l, s, e, t, y, es, ts, ys in zip(
                pf["complexity"], pf["loss"], pf["score"], pf["expression"],
                pf["latex"], pf_sympy, pf_expr_simplified, pf_latex_simplified,
                pf_sympy_simplified,
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
            New inputs. A 1-D array is treated as a single column, matching the
            training-side rule — so a model fitted on one feature can be predicted
            from a plain 1-D array, and a multi-feature model raises rather than
            silently reading the same array as one row. Must have
            :attr:`n_features` columns in the same order as the training ``X``, and
            must contain no NaN or infinite values (scikit-learn's ``check_array``
            rule; the R package follows ``predict.lm`` and propagates ``NA`` instead).

            When the fit captured :attr:`feature_names` (``X`` was a DataFrame) *and*
            ``newdata`` is a DataFrame too, the column names must match in order —
            a mismatch raises rather than being silently reordered, because this
            interface binds columns by position (docs/81 P1). If either side carries
            no names, nothing is checked.
        expression : str, optional
            Which expression to evaluate. Defaults to :attr:`recommended` (the
            Pareto "best"). Pass :attr:`expression` for the lowest-loss model, or any
            string from :attr:`pareto_front`.

        Notes
        -----
        NumPy's operators agree with the guarded operators used during training on the
        ordinary out-of-domain cases: ``np.sqrt`` of a negative number, and ``pow(x, y)``
        nodes (rendered ``x ^ y``, evaluated with ``**``) for a negative base under a
        fractional exponent, are ``nan`` both here and in the engine (see ``docs/69``).
        They part only at the edges the engine guards explicitly, where NumPy follows
        IEEE: ``0 ** -1`` and ``(-inf) ** 0.5`` are ``inf`` here and ``nan`` in the
        engine, and ``np.log(0)`` is ``-inf`` here and ``nan`` in the engine (the
        ``safe_log`` guard, ``docs/77``). It matters only if the prediction inputs
        reach them.
        """
        _check_feature_names(self.feature_names, newdata, "newdata")
        X = _as_design_matrix(newdata, "newdata", require_finite=True)
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
        if index is None:
            raise ValueError(
                "get_best() has no index to fall back on: this fit carries no "
                "recommended member (best_index is None). Pass an explicit index."
            )
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

    def sympy(
        self,
        index: Optional[int] = None,
        variable_names: Optional[Sequence[str]] = None,
    ) -> str:
        """Return a Pareto-front member as Python source SymPy's ``sympify()`` parses.

        What this rewrites is ``^``. The engine's power operator is ``^`` on every
        display surface, and Python reads that as *xor*: ``eval()``,
        ``parse_expr()``, ``lambdify()`` and NumPy all evaluate the wrong
        function, silently, for any expression that contains one — and since
        ``square`` prints as ``(a ^ 2)``, that is most of them. (``sympify()``
        alone is the exception: it passes ``convert_xor=True``.) Every other
        operator in an ``expression`` string already carries its Python name, so
        this rendering is valid under ``sympify()``, ``parse_expr()``, ``eval()``
        and NumPy alike.

        Display-only, like :meth:`latex`: :meth:`predict` keeps using the
        ``expression`` strings. **This is the mathematical form, not the
        engine's.** rsymbolic2's operators are domain-guarded — ``sqrt``, ``log``
        and ``^`` return NaN outside their domain, where SymPy returns a complex
        or symbolic value. Use :meth:`predict` to evaluate, and this to
        differentiate, simplify or typeset.

        SymPy is not a dependency of rsymbolic2; this returns a string.

        Parameters
        ----------
        index : int, optional
            0-based row of :attr:`pareto_front`. Defaults to :attr:`best_index`,
            the member chosen by ``model_selection``.
        variable_names : sequence of str, optional
            Names substituted for the ``x0``, ``x1``, ... tokens. Defaults to
            ``None``, which keeps them — unlike :meth:`latex`, feature names are
            **not** substituted automatically, because a column name is free text
            and ``"flow rate"`` is not a Python identifier. Names passed here must
            be valid Python identifiers.

        See Also
        --------
        The ``sympy_simplified`` key of :attr:`pareto_front`, for the same
        rendering of the display-simplified form.

        Examples
        --------
        >>> from sympy import sympify, diff, symbols   # doctest: +SKIP
        >>> expr = sympify(res.sympy())                # doctest: +SKIP
        >>> diff(expr, symbols("x0"))                  # doctest: +SKIP
        """
        member = self.get_best(index)
        out = member["sympy"]
        if out is None:
            raise ValueError(
                "this fit carries no 'sympy' rendering; it was produced by an "
                "older compiled extension. Re-fit with the current version."
            )
        if variable_names is None:
            return out
        if len(variable_names) != self.n_features:
            raise ValueError(
                f"variable_names has {len(variable_names)} name(s) but the model "
                f"was fitted on {self.n_features} feature(s)."
            )
        # A name that is not a Python identifier would produce a string SymPy cannot
        # parse, which is the one thing this method exists to prevent. Rejecting it is
        # better than emitting it: the caller still has the x0 form to fall back on.
        bad = [n for n in variable_names if not str(n).isidentifier()]
        if bad:
            raise ValueError(
                f"variable_names must be valid Python identifiers; rejected: {bad}"
            )
        # Two passes through a placeholder that cannot occur in the output, so a name
        # that is itself a token (a swap such as ["x1", "x0"]) is not re-substituted.
        for i in range(len(variable_names)):
            out = re.sub(rf"\bx{i}\b", f"\x00{i}\x00", out)
        for i, name in enumerate(variable_names):
            out = out.replace(f"\x00{i}\x00", str(name))
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
        """Return the Pareto front as a :class:`pandas.DataFrame` (requires pandas).

        Columns are ``complexity``, ``loss``, ``score``, ``recommended`` (True on the
        member ``model_selection`` chose) and ``expression`` — the same frame the R
        package's ``as.data.frame()`` returns, so the two counterparts agree.
        """
        import pandas as pd  # imported lazily; pandas is an optional extra

        rows = [
            {
                "complexity": m["complexity"],
                "loss": m["loss"],
                "score": m["score"],
                "recommended": i == self.best_index,
                "expression": m["expression"],
            }
            for i, m in enumerate(self.pareto_front)
        ]
        return pd.DataFrame(
            rows,
            columns=["complexity", "loss", "score", "recommended", "expression"],
        )

    def plot(
        self,
        *,
        kind: Optional[str] = None,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        expression: Optional[str] = None,
        log_loss: bool = True,
        label_exprs: bool = True,
        variable_names: Optional[Sequence[str]] = None,
        ax=None,
    ):
        """Plot the fit (requires matplotlib).

        The Python counterpart of the R package's ``plot.rsymbolic2()``, with the
        same three views:

        ``"pareto"``
            Complexity vs. loss over the non-dominated members, with the
            lowest-loss point highlighted.
        ``"fit"``
            One expression against the data. With a single feature the fitted
            curve is overlaid on the observed scatter; with several features,
            predicted values are plotted against observed ones with a dashed
            ``y = x`` reference line.
        ``"tree"``
            The structure of one expression as a syntax tree: operators as inner
            nodes, data columns and fitted constants as leaves (distinguished by
            fill). Needs no data. Its node count is that of the expression as
            printed, which can be smaller than the ``complexity`` field — that
            counts the raw tree the search archived, before the display-only
            simplification (docs/52).

        The result object stores no training data (only :attr:`n_obs` and
        :attr:`sst`, which give ``r_squared``), so ``kind="fit"`` needs the data
        passed back in via ``X`` and ``y``.

        Parameters
        ----------
        kind : {"pareto", "fit", "tree"}, optional
            Which view to draw. Defaults to ``"fit"`` when both ``X`` and ``y``
            are given (nothing else uses them), otherwise ``"pareto"``.
        X, y : array-like, optional
            Inputs and observed target for ``kind="fit"``; ``X`` in the form
            :meth:`predict` accepts. Pass the training data to inspect the fit,
            or held-out data to inspect generalisation.
        expression : str, optional
            Which expression to draw for ``kind="fit"`` or ``kind="tree"``.
            Defaults to :attr:`recommended` (for ``"tree"``, its
            display-simplified form when the fit carries one); for ``"fit"`` it is
            passed straight to :meth:`predict`.
        variable_names : sequence of str, optional
            Names to label the leaves with for ``kind="tree"``. Defaults to
            :attr:`feature_names` when set, else the 0-based ``x0, x1, ...`` the
            expression strings use. Ignored by the other views.
        log_loss : bool
            Use a log scale for the loss axis (skipped automatically when any
            loss is zero). Default ``True``. ``kind="pareto"`` only.
        label_exprs : bool
            Annotate each point with its expression string. Set to ``False``
            for large fronts where labels overlap. Default ``True``.
            ``kind="pareto"`` only.
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

        if kind is None:
            kind = "fit" if X is not None and y is not None else "pareto"
        if kind not in ("pareto", "fit", "tree"):
            raise ValueError(f"kind must be 'pareto', 'fit' or 'tree', got {kind!r}.")

        if ax is None:
            _, ax = plt.subplots()
        if kind == "fit":
            return self._plot_fit(ax, X, y, expression)
        if kind == "tree":
            return self._plot_tree(ax, expression, variable_names)
        return self._plot_pareto(ax, log_loss, label_exprs)

    def _plot_pareto(self, ax, log_loss: bool, label_exprs: bool):
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

    # One expression against the data it was fitted to (or held-out data). The
    # single-feature overlay is the more direct reading, but it only exists when there
    # is one x-axis to draw; predicted-vs-observed is the general fallback.
    def _plot_fit(self, ax, X, y, expression: Optional[str]):
        if X is None or y is None:
            raise ValueError(
                "plot(kind='fit') needs the data to compare against: pass both "
                "X=<features> and y=<observed target>. The result object stores no "
                "training data."
            )
        # A 1-D X is one column here (the single-feature case this view exists for),
        # never one row: y pairs with the rows, so a row vector could not be plotted.
        # `predict` is handed the caller's original object, not this coerced copy, so a
        # DataFrame still gets its feature names checked (docs/81 P1).
        Xa = _as_design_matrix(X, "X")
        ya = np.asarray(y, dtype=float).ravel()
        yhat = self.predict(X, expression=expression)
        if ya.shape[0] != yhat.shape[0]:
            raise ValueError(
                f"y has {ya.shape[0]} value(s) but X has {Xa.shape[0]} row(s)."
            )

        if Xa.shape[1] == 1:
            xs = Xa[:, 0]
            order = np.argsort(xs)
            ax.scatter(xs, ya, s=20, color="0.3", label="observed")
            ax.plot(xs[order], yhat[order], color="firebrick", label="model")
            names = self.feature_names
            ax.set_xlabel(names[0] if names else "x0")
            ax.set_ylabel("observed")
            ax.set_title("rsymbolic2 fit")
        else:
            # Non-finite predictions (log of a negative, a division by zero on these
            # inputs) would otherwise stretch the reference line across an empty axis.
            finite = np.isfinite(ya) & np.isfinite(yhat)
            lo, hi = (
                (float(np.min([ya[finite], yhat[finite]])), float(np.max([ya[finite], yhat[finite]])))
                if finite.any()
                else (0.0, 1.0)
            )
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.5", label="predicted = observed")
            ax.scatter(ya, yhat, s=20, color="0.3", label="predictions")
            ax.set_xlabel("observed")
            ax.set_ylabel("predicted")
            ax.set_title("rsymbolic2 fit: predicted vs observed")
        ax.legend(fontsize=8)
        return ax

    # One expression as a syntax tree (docs/48 D6). Needs no data: the structure comes
    # from the printed string, which Python's own `ast` parses — the same route the R
    # package and the web GUI take in their languages, so no tree is exported from the
    # C++ core.
    def _plot_tree(self, ax, expression: Optional[str], variable_names):
        expr = expression
        if expr is None:
            # Display surfaces prefer the display-simplified companion (docs/52), which is
            # what repr() and the LaTeX rendering already show; `recommended` stays the
            # frozen evaluatable string and the fallback for older result dicts.
            best = (
                self.pareto_front[self.best_index]
                if self.best_index is not None and self.pareto_front
                else None
            )
            expr = (best or {}).get("expression_simplified") or self.recommended
        names = variable_names if variable_names is not None else self.feature_names
        if names is not None and len(names) != self.n_features:
            raise ValueError(
                f"variable_names has {len(names)} name(s) but the model was fitted on "
                f"{self.n_features} feature(s)."
            )
        nodes = _tree_layout(expr, names)

        for n in nodes:
            if n["parent"] is None:
                continue
            p = nodes[n["parent"]]
            ax.plot([p["x"], n["x"]], [-p["depth"], -n["depth"]],
                    color="0.6", linewidth=1.0, zorder=1)
        for n in nodes:
            fc, tc = _TREE_COLORS[n["kind"]]
            # A capsule, matching the R package and the web GUI: `rounding_size` is half
            # the box height (1 font unit + 2 * pad), and the space on each side of the
            # label buys the width a one-character node would otherwise lack — without it
            # the corner arcs overlap and the text spills out of the shape.
            ax.text(
                n["x"], -n["depth"], f" {n['label']} ",
                ha="center", va="center", fontsize=9, color=tc, zorder=2,
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.75",
                          facecolor=fc, edgecolor="#cbd2db", linewidth=0.8),
            )

        xs = [n["x"] for n in nodes]
        depth = max(n["depth"] for n in nodes)
        ax.set_xlim(min(xs) - 0.75, max(xs) + 0.75)
        ax.set_ylim(-depth - 0.5, 0.5)
        ax.set_axis_off()
        ax.set_title("rsymbolic2 equation tree")
        return ax


# --- Equation tree (docs/48 D6) --------------------------------------------------------
# The unary operators the core can print in call form are `_UNARY_OPS` above (one
# definition for both validation and drawing); `neg`, `square` and `inv` are no longer
# among them — the core renders those three as `(-a)`, `(a ^ 2)` and `(1 / a)` (docs/71) —
# but they stay accepted so a string saved by an earlier version still draws. Below are
# the binary glyphs. "/" and "*" are
# rewritten: "/" reads as part of a fraction that is not there, and "*" is a raised
# asterisk that sits off-centre inside a node. The R package and the web GUI use the same
# substitutions, so one equation draws identically on all three surfaces.
_BINARY_LABEL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "×", ast.Div: "÷",
    # The core renders power as `^`, which Python's parser reads as a bitwise xor. It is
    # never anything else in an engine-generated string.
    ast.BitXor: "^", ast.Pow: "^",
}
# Node fills by kind: operator, variable (a data column), constant (a fitted number).
# Uniform shape, three fills — the distinction a symbolic-regression reader wants.
_TREE_COLORS = {
    "operator": ("#eaeef4", "#1c2430"),
    "variable": ("#e8f0ff", "#1d4ed8"),
    "constant": ("#f4f5f7", "#5b6472"),
}


def _tree_const_label(v: float) -> str:
    # The same "%.6g" the core's to_string() uses, so a node reads exactly as it does
    # inside the expression string.
    return "%.6g" % v


def _tree_draw_node(node, variable_names) -> dict:
    """One `ast` node -> {"kind", "label", "children"}, the shape all three surfaces draw.

    A negated literal folds into a single constant: "%.6g" emits "-1.3", every parser
    reads that as unary minus over 1.3, and two nodes there are noise (the C++ macro
    parser folds the same case).
    """
    if isinstance(node, ast.Expression):
        return _tree_draw_node(node.body, variable_names)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return {"kind": "constant", "label": _tree_const_label(node.value), "children": []}
    if isinstance(node, ast.Name):
        # `inf` / `nan` parse as names, not numbers; they are values, not data columns.
        if node.id in ("inf", "nan"):
            return {"kind": "constant", "label": node.id, "children": []}
        label = node.id
        if variable_names:
            try:
                idx = int(node.id[1:]) if node.id.startswith("x") else -1
            except ValueError:
                idx = -1
            if 0 <= idx < len(variable_names):
                label = str(variable_names[idx])
        return {"kind": "variable", "label": label, "children": []}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = node.operand
        if isinstance(inner, ast.Constant) and isinstance(inner.value, (int, float)):
            return {"kind": "constant", "label": _tree_const_label(-inner.value),
                    "children": []}
        # Labelled with the sign the expression string shows, not the engine's "neg":
        # one operator must not appear under two names on one screen. A one-child "-"
        # is unambiguous beside the two-child subtraction.
        return {"kind": "operator", "label": "-",
                "children": [_tree_draw_node(inner, variable_names)]}
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_LABEL:
        return {
            "kind": "operator",
            "label": _BINARY_LABEL[type(node.op)],
            "children": [_tree_draw_node(node.left, variable_names),
                         _tree_draw_node(node.right, variable_names)],
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _UNARY_OPS
        and len(node.args) == 1
    ):
        return {"kind": "operator", "label": node.func.id,
                "children": [_tree_draw_node(node.args[0], variable_names)]}
    raise ValueError(
        f"cannot draw this expression: unsupported element {ast.dump(node)[:60]}"
    )


def _tree_layout(expr: str, variable_names=None) -> list:
    """Flat node table for one expression string: [{id, parent, depth, x, label, kind}].

    Leaves take consecutive integer columns and an inner node sits at the mean of its
    children, so sibling subtrees own disjoint leaf ranges and same-depth nodes can never
    collide; a unary node lands directly above its only child. Reingold-Tilford would pack
    wide trees tighter, but at the default max_nodes = 30 this draws the same picture for a
    fraction of the code.
    """
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"could not parse the expression {expr!r}: {e}") from e
    root = _tree_draw_node(tree, variable_names)

    nodes: list = []
    next_leaf = [0]

    def walk(node, depth, parent):
        rec = {"id": len(nodes), "parent": parent, "depth": depth, "x": 0.0,
               "label": node["label"], "kind": node["kind"]}
        nodes.append(rec)
        if not node["children"]:
            rec["x"] = float(next_leaf[0])
            next_leaf[0] += 1
        else:
            xs = [walk(c, depth + 1, rec["id"]) for c in node["children"]]
            rec["x"] = sum(xs) / len(xs)
        return rec["x"]

    walk(root, 0, None)
    return nodes


# erf is the one operator with no NumPy ufunc (it lives in scipy.special, and scipy is not
# a dependency of this package). `math.erf` is the C library's, so this matches the core's
# std::erf to the last bit on the same platform; frompyfunc only supplies the vectorisation
# NumPy would otherwise give for free. predict() is not a hot path, so the per-element call
# overhead is irrelevant here.
_erf_ufunc = np.frompyfunc(math.erf, 1, 1)


def _erf(v):
    return np.asarray(_erf_ufunc(v), dtype=float)


# Math namespace used by predict(). neg/square/inv/erf are not Python builtins; the rest map
# to NumPy ufuncs so evaluation is vectorised over the input columns. `neg`, `square` and
# `inv` are no longer emitted — to_string() prints those nodes as `(-x)`, `(x ^ 2)` and
# `(1 / x)` (docs/71) — but they stay bound so a string saved by an earlier version still
# evaluates.
def _eval_namespace(X: np.ndarray) -> dict:
    ns = {
        "__builtins__": {},
        # %.6g rendering of a constant can emit these tokens; bind them so eval() does
        # not raise NameError (builtins are disabled above for safety).
        "inf": float("inf"),
        "nan": float("nan"),
        "neg": lambda v: -v,
        "square": lambda v: v * v,
        "inv": lambda v: 1.0 / v,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "sqrt": np.sqrt,
        "tanh": np.tanh,
        "abs": np.abs,
        "erf": _erf,
        "sinh": np.sinh,
        "cosh": np.cosh,
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
    macro_ops: Optional[Mapping[str, str]] = None,
    timeout_seconds: float = 0.0,
    verbosity: int = 1,
    variable_names: Optional[Sequence[str]] = None,
) -> SymbolicRegressionResult:
    """Discover a mathematical expression that fits ``y`` from ``X``.

    Uses steady-state genetic programming with Levenberg-Marquardt constant
    optimisation. Every default below is identical to PySR's documented default; only
    the implementation differs (a dependency-free C++ engine, no Julia runtime). See
    the README "Algorithm" section for the full method and references.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input features. A 1-D array is treated as a single column (n samples of one
        feature), never as one row of several features. Must have at least one row
        and one column, and must contain no NaN or infinite values. A pandas
        DataFrame is accepted and its column names are kept as
        :attr:`~SymbolicRegressionResult.feature_names`; every column must be
        numeric, since rsymbolic2 does not dummy-code categorical inputs.
    y : array-like, shape (n_samples,)
        Target values; ``len(y)`` must equal ``X.shape[0]``, with no NaN or infinite
        values. A ``(n, 1)`` column vector is accepted as the same single target;
        a wider array is refused rather than flattened, because rsymbolic2 fits one
        target at a time. A constant ``y`` is accepted with a warning: it has zero
        variance, so no expression can explain it and ``r_squared`` is None (see
        Degenerate data below).
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
        abs, square, inv, erf, sinh, cosh. (PySR ships no default operator set; this
        is the shared problem input, given identically to both tools — every name
        here exists in SymbolicRegression.jl with the same meaning.) ``square``
        (x**2) and ``inv`` (1/x) are single-node forms of structures that otherwise
        cost a ``pow``/``div`` plus a fitted constant; ``inv`` is unguarded like
        ``div``, so a zero argument yields a non-finite value that the loss guard
        rejects. ``erf``/``sinh``/``cosh`` are physical-science motifs the primitive
        set cannot reach (each needs its argument twice, so a macro cannot express
        it); ``sinh``/``cosh`` are unguarded like ``exp`` and overflow to a
        non-finite value for a large argument (docs/62). ``sqrt`` and ``log`` are
        domain-guarded (SymbolicRegression.jl ``safe_sqrt``/``safe_log``): outside
        their domain they yield NaN, which rejects the candidate rather than
        substituting a finite value (docs/69, docs/77).
    binary_ops : sequence of str, default ("add","sub","mul")
        Allowed binary operators. Recognised: add, sub, mul, div, pow. ``pow`` is
        domain-guarded the same way (``safe_pow``): where x**y is undefined it
        yields NaN and the candidate is rejected.
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
        Per-point weights for a weighted least-squares fit (PySR ``weights``); None
        fits unweighted. Must be finite and non-negative, with at least one positive:
        an all-zero vector makes every candidate's loss identically 0, so the search
        would report a perfect fit it never found, and is refused.
    batching : bool, default False
        Evaluate the evolution and constant-optimisation passes on a random subsample
        of ``batch_size`` rows per iteration instead of the full dataset (PySR
        ``batching``) — the lever for large row counts, making each candidate
        evaluation cost ``O(batch_size)`` rather than ``O(len(y))``. The hall of fame,
        early-stop test and the reported result are always computed on the full
        dataset, so batching changes only which candidates are explored, never the
        accuracy attributed to a returned model. Rows are sampled with replacement and
        re-sampled each iteration. Fewer than ~10,000 rows are usually enough without
        it; above that, an unbatched call emits an advisory ``UserWarning`` pointing
        here. The warning changes nothing about the search and the standard
        ``warnings`` filters silence it.
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
    macro_ops : mapping of str to str, optional
        User-defined **macro operators**: single-argument expression templates built from
        the primitive operators, e.g. ``{"gauss": "exp(neg(square(x)))"}``. The body is
        written in infix over the argument ``x`` and is *expanded* into the expression
        whenever a growth mutation creates a unary node, so the engine's node set stays
        closed (docs/57). Consequences: complexity is counted after expansion (a 4-node
        macro costs 4 nodes), results print the expanded primitive form, and numeric
        literals in a body become ordinary tunable constants seeded at that value.
        Off by default (``None``), and with no macros the search is bit-identical to the
        PySR-parity default. rsymbolic2 has no runtime language, so macro bodies are the
        supported form of a user-defined operator: arbitrary functions are not.
    timeout_seconds : float, default 0.0
        Wall-clock limit; 0 = no limit. A timed-out run is not reproducible across
        machines (only runs that finish within budget are bit-reproducible).
    verbosity : int, default 1
        Default 1 matches PySR's default (``verbosity=1``), printing one
        diagnostic line per epoch to stderr; 0 = silent. The line is emitted by
        the C++ core to ``stderr``; redirect the process ``stderr`` to log it.
        The compact one-liner rendering differs from PySR's live table; only the
        on/off default is matched.
    variable_names : sequence of str, optional
        Display-only names for the columns of ``X``, one per feature (PySR
        ``fit(variable_names=)``). Takes precedence over names read off a pandas
        DataFrame, and is the only way to name the columns of a plain array. They
        appear in ``repr()``, :meth:`~SymbolicRegressionResult.latex` and the plots,
        and are checked by :meth:`~SymbolicRegressionResult.predict` against a
        DataFrame's columns; the fitted expression strings stay 0-based
        (``x0, x1, ...``) and prediction is otherwise unaffected. A list of the wrong
        length raises.

    Returns
    -------
    SymbolicRegressionResult
        Best expression, Pareto front, and a :meth:`~SymbolicRegressionResult.predict`
        method.

    Notes
    -----
    **Degenerate data.** Some datasets are legal but carry no information for the
    search to find. These are not refused — the run is well defined and the caller
    may know exactly what they are doing — but each raises a ``UserWarning``, because
    the result would otherwise look like an ordinary answer (docs/80):

    * a constant ``y`` (zero variance, using the weighted variance when ``weights``
      is given): every constant fits it perfectly, so no expression is better than
      any other and ``r_squared`` is undefined. A single-row dataset lands here by
      construction.
    * a constant feature column: it cannot explain any variation in ``y``, so it only
      enlarges the search space.
    * a ``y`` whose total sum of squares is not finite: the values are each finite,
      but the sum-of-squares loss computed from them overflows, so the reported
      ``loss`` and ``r_squared`` are meaningless. Rescaling ``y`` fixes it and costs
      nothing.

    Data with no defensible reading raises ``ValueError`` instead, naming the
    argument: an ``X`` with no rows or no columns, an all-zero ``weights`` vector, a
    non-finite value in ``X`` or ``y``, and a multi-column ``y``.
    """
    # Capture display-only column names before coercion (pandas DataFrame carries
    # them in `.columns`). They are surfaced in repr() as an `x0 = name` legend and
    # never fed back into the evaluable expression strings, which stay 0-based.
    # An explicit `variable_names` wins over names read off a DataFrame: it is the only
    # way an ndarray caller can name their columns, and PySR's `fit(variable_names=)` has
    # the same precedence. Its length is checked against X below, once the shape is known.
    feature_names = (
        [str(n) for n in variable_names] if variable_names is not None
        else _column_names(X)
    )

    X_arr = _as_design_matrix(X, "X")
    y_arr = _as_target_vector(y, "y")
    if X_arr.shape[0] == 0:
        raise ValueError("X must have at least one row.")
    # No feature columns means there is no function of X to discover: the search can only
    # ever return a constant, which it did — silently, as expression "1" (docs/80). The C++
    # bridge guards this too; checking here is what names the argument.
    if X_arr.shape[1] == 0:
        raise ValueError("X must have at least one column.")
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X.shape[0] must equal len(y).")

    # NaN/Inf in the data are rejected rather than carried into the search. The core maps
    # a non-finite prediction to an infinite loss per candidate, so a single bad point
    # does not stop the run — it quietly makes the result meaningless, and the reported
    # loss can still look ordinary (a NaN in X gave a finite-looking loss and a nonsense
    # expression). `weights` has been checked this way all along; this is the same check
    # on the data it weights (docs/74).
    #
    # rsymbolic2 has no `na.action`-style option: dropping rows is a decision worth making
    # visibly, so the message says how (docs/81 P3) rather than doing it silently.
    if not np.all(np.isfinite(X_arr)):
        raise ValueError(
            "X must not contain NaN or infinite values. Drop the incomplete rows first, "
            "keeping y aligned: for example df = df.dropna() before splitting into X "
            "and y, or mask = np.isfinite(X).all(axis=1) & np.isfinite(y) then X[mask], "
            "y[mask]."
        )
    if not np.all(np.isfinite(y_arr)):
        raise ValueError(
            "y must not contain NaN or infinite values. Drop the incomplete rows first, "
            "keeping X aligned: for example mask = np.isfinite(y) then X[mask], y[mask]."
        )

    # Count-like arguments must be positive. Each is cast to an unsigned type in the C++
    # core, so a negative value wraps to an enormous count; population_size = -1 then
    # aborted the whole interpreter from inside the OpenMP region (docs/74). The binding
    # guards these too; checking here is what makes the message useful.
    for _name, _value in (
        ("population_size", population_size),
        ("generations", generations),
        ("tournament_size", tournament_size),
        ("max_depth", max_depth),
        ("max_nodes", max_nodes),
        ("n_populations", n_populations),
    ):
        if int(_value) < 1:
            raise ValueError(f"{_name} must be a positive integer.")

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
        # Non-negative and finite was not enough: with every weight zero the weighted SSE
        # is identically 0, so every candidate ties at a perfect loss and the returned
        # expression is whichever one the tournament happened to hold. "loss = 0.0" is the
        # most confidence-inspiring number this API prints, and there it means the
        # opposite (docs/80).
        if float(np.sum(weights_arr)) <= 0.0:
            raise ValueError(
                "weights must not be all zero; their sum must be positive."
            )

    _check_degenerate_data(X_arr, y_arr, weights_arr)

    mw_dict = {} if mutation_weights is None else {str(k): float(v) for k, v in mutation_weights.items()}

    if int(batch_size) < 1:
        raise ValueError("batch_size must be a positive integer.")

    # Advisory only. Every candidate evaluation is O(len(y)) and the default budget
    # spends millions of them, so a large table turns a seconds-long run into an
    # hours-long one with nothing on screen to say why. batching is the lever, and it
    # is off by default because PySR's is (parity is not ours to trade) — so the next
    # best thing is to say so. Warning changes no setting and no result; the standard
    # warnings filters silence it.
    if not batching and y_arr.shape[0] > _LARGE_DATA_ROWS:
        warnings.warn(
            f"Fitting {y_arr.shape[0]} rows. Every candidate evaluation is O(rows), "
            f"so this run may take a long time. Consider batching=True (evaluates the "
            f"search on {int(batch_size)} rows per iteration; the reported result is "
            f"still computed on all rows), a smaller subsample, or timeout_seconds.",
            stacklevel=2,
        )

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
    # Macro operators: split the mapping into two parallel lists for the bridge. The C++
    # side parses and validates the bodies (one shared parser for every interface), so a
    # bad body raises the same message here, in R and in the browser.
    if macro_ops is None:
        macro_names: list = []
        macro_bodies: list = []
    else:
        macro_names = [str(k) for k in macro_ops.keys()]
        macro_bodies = [str(v) for v in macro_ops.values()]

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
        macro_names,
        macro_bodies,
    )
    if feature_names is not None and len(feature_names) != int(X_arr.shape[1]):
        if variable_names is not None:
            # An explicit name list of the wrong length is a caller mistake, not a shape
            # coincidence: say so rather than discarding what was asked for.
            raise ValueError(
                f"variable_names has {len(feature_names)} name(s) but X has "
                f"{int(X_arr.shape[1])} column(s)."
            )
        feature_names = None  # shape changed (e.g. 1-D promoted); drop mismatched names
    return SymbolicRegressionResult(
        raw, n_features=int(X_arr.shape[1]), feature_names=feature_names
    )


class SymbolicRegressor:
    """An estimator-shaped wrapper around :func:`symbolic_regression`.

    It exists for one reason: code written against PySR's
    ``PySRRegressor(...).fit(X, y)`` should port without being restructured. Every
    hyperparameter is a keyword to the constructor, the search runs in :meth:`fit`, and
    the fitted :class:`SymbolicRegressionResult` is available as :attr:`result_`.

    **scikit-learn is not imported and is not a dependency.** The estimator protocol
    (``fit``/``predict``/``score``/``get_params``/``set_params``) is implemented by duck
    typing, which is all ``sklearn.base.clone``, ``train_test_split`` and
    ``cross_val_score`` require.

    Two things it deliberately does *not* recommend, despite being mechanically possible:

    ``Pipeline`` with a scaler
        ``make_pipeline(StandardScaler(), SymbolicRegressor())`` returns an expression in
        *standardised* coordinates. The interpretable expression in the caller's own
        variables is symbolic regression's entire product, so putting a scaler in front
        throws away the reason to run it.
    ``GridSearchCV`` over the hyperparameters
        rsymbolic2's defaults are PySR's defaults by policy, not by tuning, and replacing
        them with values chosen by a local search is the thing that policy exists to
        prevent. One default fit is also 2800 generations over 31 populations.

    Cross-validating the *default* configuration for an honest generalisation estimate is
    a different matter, and works: ``cross_val_score(SymbolicRegressor(generations=200),
    X, y)``. Note that the search is stochastic, so repeated fits differ unless ``seed``
    is fixed — spread across seeds is a real part of the answer (CLAUDE.md, Benchmarking).

    Examples
    --------
    >>> from rsymbolic2 import SymbolicRegressor
    >>> model = SymbolicRegressor(generations=40, population_size=200, seed=1)
    >>> model.fit(X, y)                      # doctest: +SKIP
    >>> model.predict(X_test)                # doctest: +SKIP
    >>> model.score(X_test, y_test)          # doctest: +SKIP
    >>> print(model.result_.recommended)     # doctest: +SKIP

    Attributes
    ----------
    result_ : SymbolicRegressionResult
        The full search result, set by :meth:`fit`. Everything the functional interface
        returns — the Pareto front, LaTeX/SymPy renderings, plots — is reached through it.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : Optional[list[str]]
        Feature names seen during :meth:`fit`, when ``X`` carried them.
    """

    # The hyperparameters are exactly `symbolic_regression`'s keyword-only arguments,
    # read from its signature rather than copied. A second hand-maintained list of ~40
    # names would drift from the function the moment one of them changed, and the
    # divergence would be silent.
    @classmethod
    def _param_names(cls) -> tuple:
        return tuple(
            name for name, p in inspect.signature(symbolic_regression).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        )

    def __init__(self, **params):
        unknown = sorted(set(params) - set(self._param_names()))
        if unknown:
            raise TypeError(
                f"unknown parameter(s) for SymbolicRegressor: {', '.join(unknown)}. "
                f"Valid parameters: {', '.join(self._param_names())}."
            )
        # Stored verbatim, and returned verbatim by get_params(): sklearn.base.clone()
        # reconstructs an estimator with `klass(**est.get_params())` and then checks that
        # each value is the *same object* it passed in, which only holds if nothing here
        # normalises or copies them.
        self._params = dict(params)
        for name, value in self._params.items():
            setattr(self, name, value)

    def get_params(self, deep: bool = True) -> dict:
        """Return the hyperparameters set on this estimator (``deep`` is accepted and
        ignored: rsymbolic2 has no nested estimators)."""
        return dict(self._params)

    def set_params(self, **params):
        """Set hyperparameters, returning ``self``."""
        unknown = sorted(set(params) - set(self._param_names()))
        if unknown:
            raise ValueError(
                f"Invalid parameter(s) for SymbolicRegressor: {', '.join(unknown)}."
            )
        self._params.update(params)
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def fit(self, X, y, weights=None, variable_names=None):
        """Run the search, and return ``self``.

        ``weights`` keeps rsymbolic2's (and PySR's) spelling rather than scikit-learn's
        ``sample_weight``; the functional interface, the R package and PySR all call it
        ``weights``, and one library speaking two names for the same vector is worse than
        differing from a convention it does not otherwise claim to implement.
        """
        call = dict(self._params)
        if weights is not None:
            call["weights"] = weights
        if variable_names is not None:
            call["variable_names"] = variable_names
        self.result_ = symbolic_regression(X, y, **call)
        self.n_features_in_ = self.result_.n_features
        self.feature_names_in_ = self.result_.feature_names
        return self

    def predict(self, X, *, expression: Optional[str] = None) -> np.ndarray:
        """Evaluate the fitted expression on ``X``. See
        :meth:`SymbolicRegressionResult.predict` for the column-name and finiteness
        rules."""
        return self._fitted().predict(X, expression=expression)

    def score(self, X, y, sample_weight=None) -> float:
        """Coefficient of determination :math:`R^2` of the prediction on ``X``, ``y``.

        The scikit-learn convention: 1.0 is a perfect fit, 0.0 is the accuracy of always
        predicting the mean of ``y``, and it can be negative. Computed on the data passed
        in, so it is a held-out figure when held-out data is passed — unlike
        ``result_.pareto_front[...]["r_squared"]``, which is always training-set.

        ``sample_weight`` follows scikit-learn's spelling here because it names a scoring
        weight, not the fit weights (`fit(weights=)`).
        """
        y_true = _as_target_vector(y, "y")
        y_pred = self.predict(X)
        w = (np.ones_like(y_true) if sample_weight is None
             else np.asarray(sample_weight, dtype=float).ravel())
        if w.shape != y_true.shape:
            raise ValueError("sample_weight must have the same length as y.")
        mean = float(np.sum(w * y_true) / np.sum(w))
        ss_res = float(np.sum(w * (y_true - y_pred) ** 2))
        ss_tot = float(np.sum(w * (y_true - mean) ** 2))
        if ss_tot <= 0.0:
            # Constant y: R^2 is undefined (the denominator is zero). scikit-learn returns
            # 1.0 for an exact prediction and 0.0 otherwise; matching that is less
            # surprising than a NaN, and symbolic_regression() has already warned.
            return 1.0 if ss_res <= 0.0 else 0.0
        return 1.0 - ss_res / ss_tot

    # scikit-learn >= 1.6 routes every estimator through `get_tags()`, which reads this
    # method and raises AttributeError without it -- so `clone()` works on a plain
    # duck-typed object but `cross_val_score()` does not. This is the one scikit-learn
    # internal protocol implemented here, and the import stays inside the method: it runs
    # only when scikit-learn is already driving, so scikit-learn is still not a dependency
    # (docs/81 P5). If a future version changes the Tags dataclass, this degrades to the
    # pre-1.6 behaviour rather than breaking the estimator for callers not using it.
    def __sklearn_tags__(self):
        try:
            from sklearn.utils import RegressorTags, Tags, TargetTags
        except ImportError:  # pragma: no cover - scikit-learn absent or older than 1.6
            raise AttributeError("__sklearn_tags__ requires scikit-learn >= 1.6")
        return Tags(
            estimator_type="regressor",
            target_tags=TargetTags(required=True),
            transformer_tags=None,
            regressor_tags=RegressorTags(),
            classifier_tags=None,
            # The search is stochastic unless `seed` is fixed: two fits on identical data
            # can return different expressions, which is a fact about symbolic regression
            # rather than a defect, and scikit-learn's checks need to be told.
            non_deterministic=True,
        )

    def _fitted(self) -> SymbolicRegressionResult:
        result = getattr(self, "result_", None)
        if result is None:
            raise RuntimeError(
                "This SymbolicRegressor is not fitted yet. Call fit(X, y) first."
            )
        return result

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self._params.items()))
        state = ""
        if getattr(self, "result_", None) is not None:
            state = f"  # fitted: {self.result_.recommended}"
        return f"SymbolicRegressor({args}){state}"
