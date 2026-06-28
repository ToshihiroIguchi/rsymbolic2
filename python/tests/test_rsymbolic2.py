"""Smoke + recovery tests for the rsymbolic2 Python binding.

Mirrors the spirit of the R package's testthat suite: defaults must match PySR, a
small problem must be recovered, and predict() must reproduce the fitted function.
Kept fast (small population/generations) so it runs in a few seconds.
"""

import inspect

import numpy as np
import pytest

from rsymbolic2 import symbolic_regression


def test_defaults_match_pysr():
    """The public defaults must be byte-for-byte PySR's documented defaults."""
    sig = inspect.signature(symbolic_regression)
    d = {k: v.default for k, v in sig.parameters.items()}
    assert d["population_size"] == 27
    assert d["n_populations"] == 31
    assert d["generations"] == 2800
    assert d["tournament_size"] == 15
    assert d["crossover_probability"] == 0.0259
    assert d["adaptive_parsimony_scaling"] == 1040.0
    assert d["optimize_probability"] == 0.14
    assert d["tournament_selection_p"] == 0.982
    assert d["fraction_replaced_hof"] == 0.0614
    assert d["parsimony"] == 0.0
    assert d["model_selection"] == "best"
    assert d["verbosity"] == 1  # PySR installed default (verbosity=1 / progress=True)


def test_verbosity_does_not_change_result():
    """verbosity is display-only: for a fixed seed the result is identical at 0 and 1."""
    X = np.linspace(-5, 5, 30).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    common = dict(unary_ops=[], population_size=60, generations=40, seed=5)
    res_silent = symbolic_regression(X, y, verbosity=0, **common)
    res_verbose = symbolic_regression(X, y, verbosity=1, **common)
    assert res_silent.expression == res_verbose.expression
    assert res_silent.loss == res_verbose.loss
    assert res_silent.pareto_front == res_verbose.pareto_front


def test_linear_recovery_and_predict():
    """Recover y = 2.5*x + 1.7 and predict on held-out points."""
    X = np.linspace(-5, 5, 30).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=200, generations=40, seed=42
    )
    assert res.loss < 1e-6
    assert res.n_features == 1
    X_new = np.array([[0.0], [2.0], [-3.0]])
    pred = res.predict(X_new)
    np.testing.assert_allclose(pred, 2.5 * X_new[:, 0] + 1.7, atol=1e-3)


def test_quadratic_recovery_with_square():
    X = np.linspace(-3, 3, 40).reshape(-1, 1)
    y = 2.0 * X[:, 0] ** 2 - 1.0
    res = symbolic_regression(
        X, y, unary_ops=["square"], population_size=200, generations=60, seed=1
    )
    assert res.loss < 1e-4
    pred = res.predict(X)
    np.testing.assert_allclose(pred, y, atol=1e-2)


def test_pareto_front_and_repr():
    X = np.linspace(-2, 2, 25).reshape(-1, 1)
    y = X[:, 0] + 3.0
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=100, generations=30, seed=7
    )
    assert len(res.pareto_front) >= 1
    assert all({"complexity", "loss", "expression"} <= set(m) for m in res.pareto_front)
    assert isinstance(repr(res), str)


def test_feature_names_from_dataframe_are_display_only():
    """A pandas DataFrame's columns become a display-only legend, not part of the
    fitted expression strings, and predict() is unaffected."""
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame(
        {"speed": np.linspace(-3, 3, 20), "mass": np.linspace(0, 1, 20)}
    )
    y = 2.0 * X["speed"].to_numpy() + X["mass"].to_numpy()
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=100, generations=15, seed=6
    )
    assert res.feature_names == ["speed", "mass"]
    assert "x0 = speed" in repr(res)
    # Expression strings stay 0-based; names never leak into them.
    assert all("speed" not in m["expression"] for m in res.pareto_front)
    # predict() still works positionally on a plain array.
    pred = res.predict(np.array([[1.0, 0.5], [0.0, 0.0]]))
    assert pred.shape == (2,)


def test_no_feature_names_for_plain_array():
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=100, generations=15, seed=7
    )
    assert res.feature_names is None
    assert "variables:" not in repr(res)


def test_input_validation():
    X = np.zeros((5, 1))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        symbolic_regression(X, y)
    with pytest.raises(ValueError):
        symbolic_regression(np.zeros((5, 1)), np.zeros(5), unary_ops=["bogus"])
