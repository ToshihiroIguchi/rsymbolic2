"""Tests for the SymPy export (docs/70).

The C++ serializer's *syntax* is pinned by the standalone test test_to_sympy.cpp.
This file checks the property that actually matters and that C++ cannot check on its
own: that SymPy parses the string into the same mathematical function the engine
evaluates, for a whole fitted Pareto front over the operators whose to_string() form
is not valid Python (square, inv, neg, pow).

SymPy is NOT a dependency of rsymbolic2 — the export is a plain string. The
round-trip tests skip when it is absent; the ones that need only the string do not.
"""

import numpy as np
import pytest

from rsymbolic2 import symbolic_regression

sympy = pytest.importorskip("sympy", reason="SymPy is a dev-only check, not a dependency")


# The four operators the plain `expression` string spells in a way Python does not
# accept, plus enough binary operators to nest them.
FULL_OPS = dict(
    unary_ops=["square", "inv", "neg", "sqrt", "abs", "exp", "log", "sin", "cos"],
    binary_ops=["add", "sub", "mul", "div", "pow"],
)


def _fit(seed=7, n=40):
    rng = np.random.default_rng(0)
    X = np.column_stack([np.linspace(0.5, 4.0, n), rng.uniform(0.5, 3.0, n)])
    y = 2.0 * X[:, 0] ** 2 + 1.0 / X[:, 1] - 0.5
    return X, y, symbolic_regression(
        X, y, population_size=40, n_populations=6, generations=80, seed=seed, **FULL_OPS
    )


def test_front_carries_sympy_renderings():
    _, _, res = _fit()
    for m in res.pareto_front:
        assert isinstance(m["sympy"], str) and m["sympy"]
        assert isinstance(m["sympy_simplified"], str) and m["sympy_simplified"]


def test_sympy_parses_without_undefined_functions():
    """The failure this export exists to remove.

    `square(a)`, `inv(a)` and `neg(a)` are not SymPy functions, and sympify() turns
    each into an undefined APPLIED FUNCTION rather than raising — a silently wrong
    expression. Nothing in the exported form may do that.
    """
    from sympy.core.function import AppliedUndef

    _, _, res = _fit()
    seen_a_rewrite = False
    for m in res.pareto_front:
        for key in ("sympy", "sympy_simplified"):
            expr = sympy.sympify(m[key])
            assert not expr.atoms(AppliedUndef), (
                f"{key}={m[key]!r} parsed with undefined functions "
                f"{expr.atoms(AppliedUndef)}"
            )
        if any(t in m["expression"] for t in ("square(", "inv(", "neg(", "^")):
            seen_a_rewrite = True
    assert seen_a_rewrite, "the front exercised none of the rewritten tokens"


def test_sympy_round_trips_to_the_same_numbers_as_predict():
    """The exported string must evaluate to what the engine evaluates.

    Compared against predict() on the training inputs, over the whole front. Points
    where the engine's domain guards fire (NaN) are excluded and reported separately:
    the export is the MATHEMATICAL form, and SymPy has no safe_sqrt.
    """
    X, _, res = _fit()
    x0, x1 = sympy.symbols("x0 x1")

    compared = 0
    for i, m in enumerate(res.pareto_front):
        expr = sympy.sympify(m["sympy"])
        f = sympy.lambdify((x0, x1), expr, "numpy")
        got = np.asarray(f(X[:, 0], X[:, 1]), dtype=float) * np.ones(X.shape[0])
        want = res.predict(X, expression=m["expression"])
        ok = np.isfinite(want) & np.isfinite(got)
        assert ok.sum() > 0, f"member {i} produced no comparable points"
        np.testing.assert_allclose(got[ok], want[ok], rtol=1e-8, atol=1e-8)
        compared += 1
    assert compared == len(res.pareto_front)


def test_sympy_method_defaults_to_the_recommended_member():
    _, _, res = _fit()
    assert res.sympy() == res.pareto_front[res.best_index]["sympy"]
    assert res.sympy(index=0) == res.pareto_front[0]["sympy"]


def test_sympy_method_variable_names():
    _, _, res = _fit()
    named = res.sympy(variable_names=["a", "bb"])
    assert "x0" not in named and "x1" not in named
    # Substitution is a rename, not a change of meaning.
    a, bb = sympy.symbols("a bb")
    x0, x1 = sympy.symbols("x0 x1")
    assert sympy.simplify(
        sympy.sympify(named).subs({a: x0, bb: x1}) - sympy.sympify(res.sympy())
    ) == 0

    # Defaults keep x0/x1 — unlike latex(), feature names are not applied silently,
    # because a column name is free text.
    res.feature_names = ["flow rate", "temp"]
    assert "x0" in res.sympy()

    with pytest.raises(ValueError, match="valid Python identifiers"):
        res.sympy(variable_names=["flow rate", "temp"])
    with pytest.raises(ValueError, match="name"):
        res.sympy(variable_names=["only_one"])


def test_variable_name_swap_is_not_resubstituted():
    """A rename whose targets are themselves tokens must not chain."""
    _, _, res = _fit()
    swapped = res.sympy(variable_names=["x1", "x0"])
    x0, x1 = sympy.symbols("x0 x1")
    expected = sympy.sympify(res.sympy()).subs({x0: sympy.Symbol("t"), x1: x0}).subs(
        {sympy.Symbol("t"): x1}
    )
    assert sympy.simplify(sympy.sympify(swapped) - expected) == 0


def test_large_data_warning_is_advisory_only():
    """It must fire, be suppressible, and change neither settings nor results."""
    rng = np.random.default_rng(3)
    X = rng.uniform(0.5, 2.0, (10_001, 1))
    y = 2.0 * X[:, 0]
    kwargs = dict(population_size=4, n_populations=1, generations=1, seed=5)

    with pytest.warns(UserWarning, match="Every candidate evaluation is O\\(rows\\)"):
        warned = symbolic_regression(X, y, **kwargs)

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning here would fail the test
        quiet = symbolic_regression(X, y, batching=True, **kwargs)
    assert quiet is not None

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        same = symbolic_regression(X, y, **kwargs)
    assert same.expression == warned.expression

    # Below the threshold nothing is emitted.
    with _w.catch_warnings():
        _w.simplefilter("error")
        symbolic_regression(X[:100], y[:100], **kwargs)
