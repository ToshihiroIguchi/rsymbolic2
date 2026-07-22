"""Tests for the opt-in search-time strong simplification (strong_simplify).

Mirrors the style of the eval_cache/linear_scaling tests in test_rsymbolic2.py. Like
linear_scaling this is a documented opt-in (docs/55); unlike eval_cache it is not
claimed to be bit-identical on/off (see docs/54's display simplifier). With the flag
off (the default) the search must be identical to not passing the argument at all
(PySR parity is untouched).
"""

import inspect

import numpy as np

from rsymbolic2 import symbolic_regression


def _line_data(n=30):
    X = np.linspace(-5, 5, n).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    return X, y


def test_strong_simplify_defaults_off_and_off_run_unchanged():
    """strong_simplify defaults to False, and passing False explicitly is identical
    to not passing it at all (the PySR-parity default path is untouched)."""
    assert (
        inspect.signature(symbolic_regression).parameters["strong_simplify"].default
        is False
    )
    X, y = _line_data()
    common = dict(unary_ops=[], population_size=60, n_populations=1,
                  generations=20, seed=11, n_threads=1, verbosity=0)
    r_default = symbolic_regression(X, y, **common)
    r_off = symbolic_regression(X, y, strong_simplify=False, **common)
    assert r_off.expression == r_default.expression
    assert r_off.loss == r_default.loss
    assert [m["expression"] for m in r_off.pareto_front] == [
        m["expression"] for m in r_default.pareto_front
    ]


def test_strong_simplify_runs_and_produces_finite_loss():
    """With strong_simplify on and a generous operator set, the run completes with a
    finite loss and the attempts/adopted counters are populated and consistent."""
    X, y = _line_data()
    res = symbolic_regression(
        X, y,
        unary_ops=["neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"],
        binary_ops=["add", "sub", "mul", "div", "pow"],
        population_size=60, n_populations=1, generations=20, seed=11,
        n_threads=1, strong_simplify=True, verbosity=0,
    )
    assert np.isfinite(res.loss)
    attempts = res.eval_counts["strong_simplify_attempts"]
    adopted = res.eval_counts["strong_simplify_adopted"]
    assert attempts > 0
    assert adopted <= attempts


def test_eval_counts_carries_strong_simplify_counters():
    """eval_counts reports strong_simplify_attempts/adopted: zero when off, populated
    when on."""
    X, y = _line_data()
    common = dict(
        unary_ops=["neg", "exp", "log", "sin", "cos", "sqrt", "tanh", "abs", "square"],
        binary_ops=["add", "sub", "mul", "div", "pow"],
        population_size=60, n_populations=1, generations=20, seed=11,
        n_threads=1, verbosity=0,
    )
    r_off = symbolic_regression(X, y, **common)
    r_on = symbolic_regression(X, y, strong_simplify=True, **common)

    assert set(r_off.eval_counts) == {
        "forward", "lm_resid", "lm_jac", "cache_hits", "cache_misses",
        "strong_simplify_attempts", "strong_simplify_adopted",
    }
    assert r_off.eval_counts["strong_simplify_attempts"] == 0
    assert r_off.eval_counts["strong_simplify_adopted"] == 0

    assert r_on.eval_counts["strong_simplify_attempts"] > 0
    assert r_on.eval_counts["strong_simplify_adopted"] <= r_on.eval_counts["strong_simplify_attempts"]
