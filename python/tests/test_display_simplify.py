"""Tests for the display-only simplifier's Python-side fields (docs/52).

The C++ rewrite rules themselves are covered by the standalone C++ test
test_display_simplify.cpp; this file only checks that the pybind11 bridge and the
SymbolicRegressionResult wrapper surface expression_simplified/latex_simplified
correctly, and that predict()/get_best() are unaffected (docs/48 D2 frozen-expression
rule: `expression`/`recommended` remain the evaluatable round-trip source).
"""

import numpy as np
import pytest

from rsymbolic2 import SymbolicRegressionResult, symbolic_regression


def test_expression_simplified_present_on_a_real_fit():
    X = np.linspace(-5, 5, 20).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=60, n_populations=4, generations=40, seed=11
    )

    assert isinstance(res.expression_simplified, str)
    assert len(res.expression_simplified) > 0

    for m in res.pareto_front:
        assert "expression_simplified" in m
        assert "latex_simplified" in m
        assert isinstance(m["expression_simplified"], str)
        assert isinstance(m["latex_simplified"], str)
        assert len(m["expression_simplified"]) > 0
        assert len(m["latex_simplified"]) > 0


def test_predict_and_get_best_unaffected_by_simplification():
    X = np.linspace(-3, 3, 25).reshape(-1, 1)
    y = 2.0 * X[:, 0] ** 2 - 1.0
    res = symbolic_regression(
        X, y, unary_ops=["square"], population_size=80, n_populations=4,
        generations=60, seed=1,
    )

    # predict() (default: recommended) matches predicting the explicit `recommended`
    # string, and never the *_simplified variant.
    pred_default = res.predict(X)
    pred_explicit = res.predict(X, expression=res.recommended)
    np.testing.assert_array_equal(pred_default, pred_explicit)
    assert np.all(np.isfinite(pred_default))

    best = res.get_best()
    assert best["expression"] == res.pareto_front[res.best_index]["expression"]
    # get_best() still returns the frozen `expression`, not the simplified form, as the
    # dict's "expression" key (the simplified form lives alongside it, not in place of
    # it).
    assert "expression_simplified" in best


def test_same_seed_reports_the_same_expression_simplified():
    """The user-facing half of the docs/66 reproducibility contract.

    expression_simplified used to come out of an e-graph bounded partly by a wall-clock
    budget, so two runs of the same fixed-seed search could report different renderings
    of the identical model. The budget is now counts only, making display_simplify a
    pure function of the tree.

    A unary set and enough generations to reach non-trivial trees, because a two-node
    answer would render identically no matter how the e-graph were bounded.
    """
    X = np.linspace(-3, 3, 40).reshape(-1, 1)
    y = 1.4 * X[:, 0] ** 2 - 0.8 * X[:, 0] + 2.1

    def fit():
        return symbolic_regression(
            X, y, unary_ops=["square", "neg"], population_size=60, n_populations=4,
            generations=60, seed=909,
        )

    a, b = fit(), fit()

    assert a.expression_simplified == b.expression_simplified
    assert [m["expression_simplified"] for m in a.pareto_front] == [
        m["expression_simplified"] for m in b.pareto_front
    ]
    assert [m["latex_simplified"] for m in a.pareto_front] == [
        m["latex_simplified"] for m in b.pareto_front
    ]


def _synthetic_raw_with_simplified(n_members=3, best_index=1):
    return {
        "expression": f"expr{n_members - 1}",
        "expression_simplified": f"simp{n_members - 1}",
        "loss": 1.0 / n_members,
        "complexity": n_members,
        "recommended": f"expr{best_index}",
        "best_index": best_index,
        "pareto_front": {
            "complexity": list(range(1, n_members + 1)),
            "loss": [1.0 / (i + 1) for i in range(n_members)],
            "score": [0.0] + [0.1] * (n_members - 1),
            "expression": [f"expr{i}" for i in range(n_members)],
            "latex": [f"latex{i}" for i in range(n_members)],
            "expression_simplified": [f"simp{i}" for i in range(n_members)],
            "latex_simplified": [f"latexsimp{i}" for i in range(n_members)],
        },
    }


def test_synthetic_raw_dict_surfaces_simplified_fields():
    raw = _synthetic_raw_with_simplified()
    res = SymbolicRegressionResult(raw, n_features=1)

    assert res.expression_simplified == "simp2"
    assert res.pareto_front[0]["expression_simplified"] == "simp0"
    assert res.pareto_front[0]["latex_simplified"] == "latexsimp0"
    assert res.pareto_front[2]["expression_simplified"] == "simp2"


def test_backward_compatible_when_raw_dict_predates_simplified_fields():
    """An older raw dict without expression_simplified/latex_simplified must not crash
    the wrapper; the new attributes degrade to None rather than raising."""
    raw = {
        "expression": "x0", "loss": 0.1, "complexity": 1,
        "recommended": "x0", "best_index": 0,
        "pareto_front": {
            "complexity": [1], "loss": [0.1], "score": [0.0],
            "expression": ["x0"], "latex": ["x_{0}"],
        },
    }
    res = SymbolicRegressionResult(raw, n_features=1)
    assert res.expression_simplified is None
    assert res.pareto_front[0]["expression_simplified"] is None
    assert res.pareto_front[0]["latex_simplified"] is None
    # predict() still works off the frozen expression.
    pred = res.predict(np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(pred, [1.0, 2.0])
