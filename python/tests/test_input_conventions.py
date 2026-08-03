"""Input conventions on the Python side (docs/81).

Covers the feature-name check in predict (P1), the remedy-bearing refusals (P3), the
fit-time variable_names and the predict finiteness check (P4), and the estimator-shaped
wrapper (P5).
"""

import numpy as np
import pytest

from rsymbolic2 import SymbolicRegressor, symbolic_regression

FAST = dict(unary_ops=[], binary_ops=["add", "sub", "mul"], generations=60,
            n_populations=4, population_size=40, seed=1, verbosity=0)


# pandas is an optional extra, so the skip is per-test rather than module-level: only the
# column-name cases actually need it, and gating the whole file on it hid every P4/P5 test
# on a pandas-less machine -- which is exactly what happened on the Ubuntu venv.
@pytest.fixture
def pd():
    return pytest.importorskip("pandas")


@pytest.fixture
def frame_fit(pd):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
    y = 3.0 * df["a"].to_numpy() - df["b"].to_numpy()
    return df, y, symbolic_regression(df, y, **FAST)


# --- P1: feature names -------------------------------------------------------------

def test_matching_columns_predict_normally(frame_fit):
    df, y, res = frame_fit
    assert res.feature_names == ["a", "b"]
    assert np.all(np.isfinite(res.predict(df)))


def test_reordered_columns_raise_rather_than_being_reordered(frame_fit):
    df, y, res = frame_fit
    # The defect this check exists for: before docs/81 this returned numbers with no
    # error, off by whatever swapping the two features does to the expression.
    with pytest.raises(ValueError, match="feature names do not match"):
        res.predict(df[["b", "a"]])


def test_renamed_column_raises_and_names_the_difference(frame_fit):
    df, y, res = frame_fit
    with pytest.raises(ValueError, match="missing.*b.*unexpected.*c"):
        res.predict(df.rename(columns={"b": "c"}))


def test_unnamed_newdata_is_not_checked(frame_fit):
    """Only one side carrying names means nothing can be verified; stay silent."""
    df, y, res = frame_fit
    named = res.predict(df)
    plain = res.predict(df.to_numpy())
    assert np.allclose(named, plain)


def test_unnamed_fit_accepts_a_named_frame(frame_fit):
    df, y, _ = frame_fit
    res = symbolic_regression(df.to_numpy(), y, **FAST)
    assert res.feature_names is None
    assert np.all(np.isfinite(res.predict(df)))


# --- P3: refusals carry a remedy ---------------------------------------------------

def test_non_numeric_column_is_named(pd):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "f": ["x", "y", "x"]})
    with pytest.raises(ValueError, match=r"non-numeric column\(s\): f"):
        symbolic_regression(df, [1.0, 2.0, 3.0], generations=5, verbosity=0)


def test_categorical_column_is_named_too(pd):
    """pandas extension dtypes cannot go through np.dtype(); they must still be named."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "f": pd.Categorical(["x", "y", "x"])})
    with pytest.raises(ValueError, match=r"non-numeric column\(s\): f"):
        symbolic_regression(df, [1.0, 2.0, 3.0], generations=5, verbosity=0)


def test_nullable_integer_column_is_accepted(pd):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": pd.array([1, 2, 3], dtype="Int64")})
    res = symbolic_regression(df, [1.0, 2.0, 3.0], **FAST)
    assert res.feature_names == ["a", "b"]


def test_non_finite_training_data_names_the_fix():
    X = np.linspace(-3, 3, 10).reshape(-1, 1)
    y = 2 * X[:, 0] + 1
    Xbad = X.copy()
    Xbad[3, 0] = np.nan
    with pytest.raises(ValueError, match="dropna"):
        symbolic_regression(Xbad, y, generations=5, verbosity=0)
    ybad = y.copy()
    ybad[4] = np.inf
    with pytest.raises(ValueError, match="isfinite"):
        symbolic_regression(X, ybad, generations=5, verbosity=0)


# --- P4: variable_names and predict finiteness -------------------------------------

def test_variable_names_name_the_columns_of_a_plain_array(frame_fit):
    df, y, _ = frame_fit
    res = symbolic_regression(df.to_numpy(), y, variable_names=["u", "v"], **FAST)
    assert res.feature_names == ["u", "v"]
    # And they take part in the predict-time check.
    with pytest.raises(ValueError, match="feature names do not match"):
        res.predict(df)


def test_variable_names_override_dataframe_columns(frame_fit):
    df, y, _ = frame_fit
    res = symbolic_regression(df, y, variable_names=["u", "v"], **FAST)
    assert res.feature_names == ["u", "v"]


def test_variable_names_of_the_wrong_length_raise(frame_fit):
    df, y, _ = frame_fit
    with pytest.raises(ValueError, match="variable_names has 1 name"):
        symbolic_regression(df, y, variable_names=["u"], generations=5, verbosity=0)


def test_predict_rejects_non_finite_newdata(frame_fit):
    df, y, res = frame_fit
    bad = df.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="must not contain NaN"):
        res.predict(bad)


# --- P5: the estimator-shaped wrapper ----------------------------------------------

def test_fit_predict_score_round_trip():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    model = SymbolicRegressor(**FAST).fit(X[:40], y[:40])
    assert model.n_features_in_ == 2
    assert model.feature_names_in_ is None
    assert model.predict(X[40:]).shape == (20,)
    assert model.score(X[40:], y[40:]) > 0.99
    # The full functional result stays reachable.
    assert isinstance(model.result_.recommended, str)


def test_fit_records_feature_names(frame_fit):
    df, y, _ = frame_fit
    model = SymbolicRegressor(**FAST).fit(df, y)
    assert model.feature_names_in_ == ["a", "b"]
    with pytest.raises(ValueError, match="feature names do not match"):
        model.predict(df[["b", "a"]])


def test_get_set_params_round_trip():
    model = SymbolicRegressor(**FAST)
    params = model.get_params()
    assert params["seed"] == 1
    # What sklearn.base.clone() does: rebuild from get_params() and compare.
    assert type(model)(**params).get_params() == params
    model.set_params(seed=7)
    assert model.get_params()["seed"] == 7 and model.seed == 7


def test_unknown_parameters_are_refused():
    with pytest.raises(TypeError, match="unknown parameter"):
        SymbolicRegressor(nonsense=1)
    with pytest.raises(ValueError, match="Invalid parameter"):
        SymbolicRegressor().set_params(nonsense=1)


def test_predicting_before_fitting_raises():
    with pytest.raises(RuntimeError, match="not fitted yet"):
        SymbolicRegressor().predict(np.zeros((2, 1)))


def test_score_on_a_constant_target_does_not_divide_by_zero():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 1))
    model = SymbolicRegressor(**FAST).fit(X, 2.0 * X[:, 0])
    # ss_tot == 0: sklearn returns 1.0 for an exact prediction, 0.0 otherwise.
    assert model.score(np.zeros((5, 1)), np.zeros(5)) in (0.0, 1.0)


def test_scikit_learn_interoperability():
    """clone() needs only get_params; cross_val_score needs __sklearn_tags__ too."""
    sklearn_base = pytest.importorskip("sklearn.base")
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(2)
    X = rng.normal(size=(45, 2))
    y = 2.0 * X[:, 0] - X[:, 1]
    model = SymbolicRegressor(**FAST)
    assert sklearn_base.clone(model).get_params() == model.get_params()
    assert np.all(cross_val_score(model, X, y, cv=3) > 0.9)
