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


def test_reciprocal_recovery_with_inv():
    """inv is opt-in (not a default) and must round-trip through predict()."""
    X = np.linspace(0.5, 4.0, 40).reshape(-1, 1)  # away from the pole: inv is unguarded
    y = 2.0 / X[:, 0]
    res = symbolic_regression(
        X, y, unary_ops=["inv"], population_size=200, generations=60, seed=1
    )
    assert res.loss < 1e-4
    assert "inv" in res.expression
    pred = res.predict(X, expression=res.expression)
    np.testing.assert_allclose(pred, y, atol=1e-2)


def test_macro_ops_default_none_is_inert():
    """macro_ops=None must be the same run as not passing it at all."""
    X = np.linspace(-2, 2, 20).reshape(-1, 1)
    y = 1.5 * X[:, 0] + 0.5
    common = dict(
        unary_ops=["exp"], binary_ops=["add", "mul"],
        population_size=60, generations=30, seed=11,
    )
    a = symbolic_regression(X, y, **common)
    b = symbolic_regression(X, y, macro_ops=None, **common)
    assert a.expression == b.expression
    assert a.loss == b.loss


def test_macro_op_reaches_an_otherwise_unreachable_target():
    """y = 3*exp(-x^2) is unreachable with no unary operators; the macro supplies it."""
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = 3.0 * np.exp(-(X[:, 0] ** 2))
    res = symbolic_regression(
        X, y,
        unary_ops=[], binary_ops=["add", "mul"],
        macro_ops={"gauss": "exp(neg(square(x)))"},
        population_size=200, generations=80, seed=4,
    )
    assert res.loss < 1e-3
    # Macros are expanded, so the reported expression is in primitive form.
    assert "exp" in res.expression
    assert "gauss" not in res.expression
    assert np.all(np.isfinite(res.predict(X, expression=res.expression)))


@pytest.mark.parametrize(
    "body, message",
    [
        ("exp(x) + x", "exactly once"),
        ("exp(1)", "exactly once"),
        ("gauss(x)", "unknown function"),
        ("exp(x", "expected"),
        ("x0 + x", "unknown identifier"),
    ],
)
def test_macro_body_validation(body, message):
    X = np.linspace(1.0, 2.0, 8).reshape(-1, 1)
    with pytest.raises(ValueError, match=message):
        symbolic_regression(
            X, X[:, 0], binary_ops=["add", "mul"], macro_ops={"f": body},
            population_size=10, generations=1,
        )


def test_unknown_operator_name_is_rejected():
    X = np.linspace(1.0, 2.0, 10).reshape(-1, 1)
    with pytest.raises(ValueError, match="Unknown unary operator"):
        symbolic_regression(X, X[:, 0], unary_ops=["reciprocal"], generations=1)


def test_pareto_front_and_repr():
    X = np.linspace(-2, 2, 25).reshape(-1, 1)
    y = X[:, 0] + 3.0
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=100, generations=30, seed=7
    )
    assert len(res.pareto_front) >= 1
    assert all(
        {"complexity", "loss", "score", "r_squared", "expression", "latex"} <= set(m)
        for m in res.pareto_front
    )
    # The simplest member has no predecessor, hence score 0 (PySR convention);
    # later members carry the engine-computed log-loss drop per complexity.
    assert res.pareto_front[0]["score"] == 0.0
    assert all(np.isfinite(m["score"]) for m in res.pareto_front)
    # r_squared is 1 - loss/sst on the training data; the line is recoverable,
    # so the best member's R^2 is ~1 and the repr reports it.
    assert res.sst > 0 and res.n_obs == 25
    assert all(
        abs(m["r_squared"] - (1.0 - m["loss"] / res.sst)) < 1e-12
        for m in res.pareto_front
    )
    assert max(m["r_squared"] for m in res.pareto_front) > 0.99
    assert "R-squared (recommended):" in repr(res)


def _synthetic_result(n_members, best_index=1):
    """Build a SymbolicRegressionResult from a hand-made raw dict (no search), so
    display formatting can be tested deterministically."""
    from rsymbolic2 import SymbolicRegressionResult

    raw = {
        "expression": f"expr{n_members - 1}",
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
        },
    }
    return SymbolicRegressionResult(raw, n_features=1)


def test_repr_marks_recommended_row_and_shows_score():
    res = _synthetic_result(3, best_index=1)
    out = repr(res)
    assert "complexity | loss | score | expression" in out
    lines = out.splitlines()
    marked = [l for l in lines if l.lstrip().startswith(">")]
    assert len(marked) == 1
    assert "expr1" in marked[0]
    assert "0.1" in marked[0]  # score column rendered


def test_repr_elides_long_fronts():
    res = _synthetic_result(30, best_index=2)
    out = repr(res)
    # 20 rows shown, the 10 in the middle elided (mirrors R format_pareto_lines).
    assert "... (10 more) ..." in out
    assert "expr0" in out and "expr29" in out
    assert "expr15" not in out


def test_get_best():
    res = _synthetic_result(3, best_index=1)
    assert res.get_best()["expression"] == "expr1"
    assert res.get_best(index=2)["expression"] == "expr2"
    with pytest.raises(IndexError):
        res.get_best(index=3)


def test_get_best_empty_front():
    from rsymbolic2 import SymbolicRegressionResult

    raw = {
        "expression": "", "loss": float("inf"), "complexity": 0,
        "recommended": "", "best_index": None,
        "pareto_front": {
            "complexity": [], "loss": [], "score": [],
            "expression": [], "latex": [],
        },
    }
    res = SymbolicRegressionResult(raw, n_features=1)
    with pytest.raises(ValueError):
        res.get_best()


def test_latex_accessor_and_name_substitution():
    from rsymbolic2 import SymbolicRegressionResult

    n = 11  # >= 10 features so x_{1} vs x_{10} substitution is exercised
    raw = {
        "expression": "x0", "loss": 0.1, "complexity": 1,
        "recommended": "x0", "best_index": 0,
        "pareto_front": {
            "complexity": [1], "loss": [0.1], "score": [0.0],
            "expression": ["x0"],
            "latex": ["x_{1} + x_{10} \\cdot x_{0}"],
        },
    }
    res = SymbolicRegressionResult(raw, n_features=n)
    assert res.latex() == "x_{1} + x_{10} \\cdot x_{0}"

    names = [f"v{i}" for i in range(n)]
    names[10] = "big_name"  # underscore must be escaped
    out = res.latex(variable_names=names)
    assert out == "v1 + big\\_name \\cdot v0"

    with pytest.raises(ValueError):
        res.latex(variable_names=["only_one"])
    # Empty list forces the raw x_{i} form even when feature_names exist.
    res.feature_names = names
    assert res.latex(variable_names=[]) == "x_{1} + x_{10} \\cdot x_{0}"


def test_latex_from_fit_recovers_line():
    X = np.linspace(-2, 2, 25).reshape(-1, 1)
    y = 2.0 * X[:, 0]
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=100, generations=30, seed=7
    )
    s = res.latex()
    assert isinstance(s, str) and len(s) > 0
    assert "x_{0}" in s


def test_plot_returns_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # headless-safe on Windows and CI

    res = _synthetic_result(3, best_index=1)
    ax = res.plot()
    assert ax.get_xlabel() == "Complexity (nodes)"
    assert ax.get_yscale() == "log"
    # log_loss=False keeps a linear axis.
    ax2 = res.plot(log_loss=False, label_exprs=False)
    assert ax2.get_yscale() == "linear"
    import matplotlib.pyplot as plt

    plt.close("all")


def test_plot_zero_loss_falls_back_to_linear():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    from rsymbolic2 import SymbolicRegressionResult

    raw = {
        "expression": "x0", "loss": 0.0, "complexity": 1,
        "recommended": "x0", "best_index": 1,
        "pareto_front": {
            "complexity": [1, 3], "loss": [1.0, 0.0],
            "score": [0.0, 345.0], "expression": ["1", "x0"],
            "latex": ["1", "x_{0}"],
        },
    }
    res = SymbolicRegressionResult(raw, n_features=1)
    ax = res.plot()  # must not crash on log-scale with a zero loss
    assert ax.get_yscale() == "linear"
    import matplotlib.pyplot as plt

    plt.close("all")


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


def test_batching_recovers_line_full_data_loss():
    """With many rows and a small per-iteration subsample, batching still recovers the
    line; the reported loss is computed on the full dataset, so an exact line gives ~0."""
    X = np.linspace(-10, 10, 300).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=60, n_populations=1, generations=80,
        seed=13, batching=True, batch_size=16,
    )
    assert res.loss < 1e-6
    assert len(res.pareto_front) >= 1


def test_batching_is_deterministic():
    X = np.linspace(-10, 10, 200).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    common = dict(
        unary_ops=[], population_size=50, n_populations=1, generations=40,
        seed=21, batching=True, batch_size=20,
    )
    r1 = symbolic_regression(X, y, **common)
    r2 = symbolic_regression(X, y, **common)
    assert r1.expression == r2.expression
    assert r1.loss == r2.loss


def test_batch_size_must_be_positive():
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    with pytest.raises(ValueError):
        symbolic_regression(X, y, batching=True, batch_size=0)


def test_warmup_maxsize_default_off_and_ramp():
    """warmup_maxsize_by defaults to 0 (off, unchanged search); a non-zero value still
    recovers the line, stays within max_nodes, and is deterministic for a fixed seed."""
    assert inspect.signature(symbolic_regression).parameters["warmup_maxsize_by"].default == 0.0

    X = np.linspace(-8, 8, 30).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    common = dict(unary_ops=[], population_size=60, n_populations=1, generations=40, seed=9)

    # Default vs explicit 0: identical search.
    base = symbolic_regression(X, y, **common)
    off = symbolic_regression(X, y, warmup_maxsize_by=0.0, **common)
    assert base.expression == off.expression
    assert base.loss == off.loss

    # Warmup on: recovers, capped, deterministic.
    warm = dict(common, generations=80, warmup_maxsize_by=0.5)
    r1 = symbolic_regression(X, y, **warm)
    r2 = symbolic_regression(X, y, **warm)
    assert r1.expression == r2.expression
    assert r1.loss == r2.loss
    assert r1.loss < 1e-6
    assert all(m["complexity"] <= 30 for m in r1.pareto_front)


def test_warmup_maxsize_rejects_negative():
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    with pytest.raises(ValueError):
        symbolic_regression(X, y, warmup_maxsize_by=-0.1)


def test_n_threads_default_none_and_result_invariant():
    """n_threads defaults to None (auto = all cores) and is a pure wall-clock knob:
    for a fixed seed an explicit thread count returns the identical result."""
    assert inspect.signature(symbolic_regression).parameters["n_threads"].default is None
    X = np.linspace(-5, 5, 30).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    common = dict(unary_ops=[], n_populations=4, population_size=100,
                  generations=20, seed=99)
    auto = symbolic_regression(X, y, **common)            # n_threads=None
    one = symbolic_regression(X, y, n_threads=1, **common)
    two = symbolic_regression(X, y, n_threads=2, **common)
    assert auto.expression == one.expression == two.expression
    assert auto.loss == one.loss == two.loss


def test_n_threads_rejects_non_positive():
    X = np.linspace(-3, 3, 20).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    with pytest.raises(ValueError):
        symbolic_regression(X, y, generations=1, n_threads=0)
    with pytest.raises(ValueError):
        symbolic_regression(X, y, generations=1, n_threads=-2)


def test_eval_accounting():
    """n_evals/eval_counts: consistent breakdown (n_evals = forward + lm_resid;
    lm_jac reported only) and deterministic for a fixed seed."""
    X = np.linspace(-8, 8, 16).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    # n_threads=1: in this Python host process, multi-threaded runs occasionally
    # flip between two trajectories for a fixed seed (an environmental effect the
    # standalone C++ determinism gates never show in a clean process), which is
    # orthogonal to what this test checks. Single-threaded runs are deterministic.
    common = dict(unary_ops=[], population_size=60, n_populations=2,
                  generations=40, seed=11, n_threads=1)
    r1 = symbolic_regression(X, y, **common)
    r2 = symbolic_regression(X, y, **common)

    assert isinstance(r1.n_evals, int) and r1.n_evals > 0
    assert set(r1.eval_counts) == {
        "forward", "lm_resid", "lm_jac", "cache_hits", "cache_misses",
        "strong_simplify_attempts", "strong_simplify_adopted",
    }
    assert all(v >= 0 for v in r1.eval_counts.values())
    # n_evals is the max_evals unit: forward passes + LM residual evaluations.
    # Jacobian builds are reported only and never charged to n_evals.
    assert r1.n_evals == r1.eval_counts["forward"] + r1.eval_counts["lm_resid"]
    # Deterministic for a fixed seed.
    assert r1.n_evals == r2.n_evals
    assert r1.eval_counts == r2.eval_counts


def test_eval_cache_defaults_off_and_bit_identical():
    """eval_cache defaults to False and is implementation-only: with the cache on the
    result (expression, loss, Pareto front, eval accounting) is bit-identical to off;
    only the cache_hits/cache_misses counters differ."""
    assert inspect.signature(symbolic_regression).parameters["eval_cache"].default is False

    X = np.linspace(-8, 8, 16).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    # n_threads=1 pins out a pre-existing, eval_cache-independent flake: in this
    # Python host process, multi-threaded runs occasionally flip between two
    # trajectories even with eval_cache off (the standalone C++ determinism gates,
    # run in a clean process, never do). Single-threaded runs are fully
    # deterministic, which is what an on-vs-off bit-identity comparison needs.
    common = dict(unary_ops=[], population_size=60, n_populations=2,
                  generations=40, seed=11, n_threads=1)
    r_off = symbolic_regression(X, y, **common)
    r_on = symbolic_regression(X, y, eval_cache=True, **common)

    assert r_on.expression == r_off.expression
    assert r_on.loss == r_off.loss
    assert [m["loss"] for m in r_on.pareto_front] == [m["loss"] for m in r_off.pareto_front]
    # A cache hit is charged like a real evaluation, so accounting matches too.
    assert r_on.n_evals == r_off.n_evals
    for k in ("forward", "lm_resid", "lm_jac"):
        assert r_on.eval_counts[k] == r_off.eval_counts[k]
    # Counters: zero when off; the tiny operator set guarantees duplicates when on.
    assert r_off.eval_counts["cache_hits"] == 0
    assert r_off.eval_counts["cache_misses"] == 0
    assert r_on.eval_counts["cache_hits"] > 0
    assert r_on.eval_counts["cache_misses"] > 0


def test_eval_cache_inactive_under_batching():
    """batching keeps the cache inactive: batching+eval_cache equals batching alone
    bitwise, with zero cache counters."""
    X = np.linspace(-10, 10, 200).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    common = dict(unary_ops=[], population_size=50, n_populations=1, generations=40,
                  seed=21, batching=True, batch_size=20)
    r_off = symbolic_regression(X, y, **common)
    r_on = symbolic_regression(X, y, eval_cache=True, **common)
    assert r_on.expression == r_off.expression
    assert r_on.loss == r_off.loss
    assert r_on.n_evals == r_off.n_evals
    assert r_on.eval_counts["cache_hits"] == 0
    assert r_on.eval_counts["cache_misses"] == 0


# --- Opt-in Keijzer-2003 linear scaling (linear_scaling) ---------------------------

def test_linear_scaling_defaults_off_and_off_run_unchanged():
    """linear_scaling defaults to False, and passing False explicitly is identical to
    not passing it at all (the PySR-parity default path is untouched)."""
    assert (
        inspect.signature(symbolic_regression).parameters["linear_scaling"].default
        is False
    )
    X = np.linspace(-5, 5, 30).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    common = dict(unary_ops=[], population_size=60, n_populations=1,
                  generations=20, seed=11, verbosity=0)
    r_default = symbolic_regression(X, y, **common)
    r_off = symbolic_regression(X, y, linear_scaling=False, **common)
    assert r_off.expression == r_default.expression
    assert r_off.loss == r_default.loss
    assert [m["expression"] for m in r_off.pareto_front] == [
        m["expression"] for m in r_default.pareto_front
    ]


def test_linear_scaling_self_consistent_fit():
    """With linear scaling on, the run returns a finite loss and the reported loss is
    reproducible from the returned (materialised a*f+b) expression string via
    predict(), up to the %.6g string round-trip of the constants."""
    X = np.linspace(-5, 5, 30).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 1.7
    res = symbolic_regression(
        X, y, unary_ops=[], population_size=60, n_populations=1, generations=20,
        seed=11, linear_scaling=True, verbosity=0,
    )
    assert np.isfinite(res.loss)
    pred = res.predict(X, expression=res.expression)
    sse = float(np.sum((pred - y) ** 2))
    assert abs(sse - res.loss) < 1e-4 * (1 + abs(res.loss))


def test_linear_scaling_rejects_units():
    X = np.linspace(1, 5, 20).reshape(-1, 1)
    y = 2 * X[:, 0]
    with pytest.raises(ValueError, match="not supported with dimensional analysis"):
        symbolic_regression(X, y, X_units=["m"], linear_scaling=True)


def test_input_validation():
    X = np.zeros((5, 1))
    y = np.zeros(4)
    with pytest.raises(ValueError):
        symbolic_regression(X, y)
    with pytest.raises(ValueError):
        symbolic_regression(np.zeros((5, 1)), np.zeros(5), unary_ops=["bogus"])


# --- Opt-in dimensional analysis (PySR X_units/y_units/...; docs/46) ---------------

def _force_data(n=24):
    """Force = mass * acceleration: a dimensionally consistent 2-feature target."""
    rng = np.random.default_rng(7)
    m = rng.uniform(1, 5, n)
    a = rng.uniform(1, 5, n)
    return np.column_stack([m, a]), m * a


def test_units_defaults_off():
    d = inspect.signature(symbolic_regression).parameters
    assert d["X_units"].default is None
    assert d["y_units"].default is None
    assert d["dimensional_constraint_penalty"].default is None
    assert d["dimensionless_constants_only"].default is False


def test_units_validation():
    X, y = _force_data()
    with pytest.raises(ValueError):  # wrong length
        symbolic_regression(X, y, X_units=["kg"])
    with pytest.raises(ValueError):  # bad unit string
        symbolic_regression(X, y, X_units=["kg", "flibble"])
    with pytest.raises(ValueError):  # y_units without X_units
        symbolic_regression(X, y, y_units="N")
    with pytest.raises(ValueError):  # negative penalty
        symbolic_regression(X, y, X_units=["kg", "m/s^2"],
                            dimensional_constraint_penalty=-1)


def test_units_run_completes():
    X, y = _force_data()
    res = symbolic_regression(
        X, y, unary_ops=[], X_units=["kg", "m/s^2"], y_units="N",
        population_size=60, n_populations=4, generations=150, seed=1,
    )
    assert np.isfinite(res.loss)
    assert len(res.pareto_front) >= 1


def test_units_mismatched_y_units_penalised():
    X, y = _force_data()
    # dimensionless_constants_only closes the "multiply by a free (wildcard) constant"
    # escape: otherwise a trailing *c makes the root a wildcard that satisfies any y_units
    # (faithful to SR.jl). With dimensionless constants, x0*x1 has a fixed output dimension
    # (N), so a mismatched y_units="s" is a genuine, deterministic violation.
    common = dict(unary_ops=[], X_units=["kg", "m/s^2"], dimensionless_constants_only=True,
                  population_size=60, n_populations=4, generations=200, seed=1)
    res_ok = symbolic_regression(X, y, y_units="N", **common)
    res_bad = symbolic_regression(X, y, y_units="s", **common)
    assert res_ok.loss < 1e-6            # x0*x1 recovered under correct units
    assert res_bad.loss > res_ok.loss    # every well-fitting model violates -> penalised


def test_units_do_not_degrade_recovery():
    X, y = _force_data()
    common = dict(unary_ops=[], population_size=60, n_populations=4,
                  generations=150, seed=3)
    res_off = symbolic_regression(X, y, **common)
    res_on = symbolic_regression(X, y, X_units=["kg", "m/s^2"], y_units="N", **common)
    assert res_off.loss < 1e-6
    assert res_on.loss < 1e-6
