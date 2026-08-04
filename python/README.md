# rsymbolic2 (Python)

Python bindings for the rsymbolic2 native symbolic-regression engine. The full
documentation — installation, tutorial, worked examples, the parameter reference and
the algorithm description with references — lives in the
[repository README](https://github.com/ToshihiroIguchi/rsymbolic2#readme).
To see the engine work before installing anything, open the browser demo:
<https://toshihiroiguchi.github.io/rsymbolic2/>.

## Install

Requires Python >= 3.9, NumPy, and a C++17 compiler. rsymbolic2 is **not on PyPI yet**,
so install it from the repository — the package lives in the `python/` subdirectory and
its build references the shared C++ core outside it, so the URL must carry
`#subdirectory=python` and a local install must be made from a full clone:

```bash
# from GitHub
pip install "git+https://github.com/ToshihiroIguchi/rsymbolic2.git#subdirectory=python"

# ...or from a clone of the repository
pip install ./python
```

Either form pulls the build tools (scikit-build-core, pybind11, CMake, Ninja) into an
isolated build environment and compiles the extension; only the compiler has to be
present. On Windows either MSVC (Visual Studio Build Tools) or Rtools works — Rtools is
required for the *R* package, not for this one.

Optional extras enable the two convenience helpers:

```bash
pip install "./python[pandas,plot]"   # res.to_pandas() / res.plot()
```

`res.plot()` draws the Pareto front, `res.plot(X=X, y=y)` the equation against the data,
and `res.plot(kind="tree")` its structure as a syntax tree (matplotlib).

## Quick start

```python
import numpy as np
from rsymbolic2 import symbolic_regression

X = np.linspace(-3, 3, 40).reshape(-1, 1)
y = 2.5 * X[:, 0] ** 2 - 1.3

res = symbolic_regression(X, y, unary_ops=["square"], seed=1)
print(res.expression)
print(res.predict(np.array([[0.0], [1.0]])))
print(res)          # Pareto front with per-member score, training R-squared,
                    # and a ">" marker on the recommended row
print(res.latex())  # LaTeX of the recommended member (display-only)
print(res.sympy())  # the same, as Python that SymPy's sympify() parses
```

The defaults are PySR's, which means a thorough (2800-generation, 31-population) search;
for a quick first look pass e.g. `population_size=200, generations=60`.
`help(symbolic_regression)` documents every parameter.

An estimator-shaped wrapper is available for code written against
`PySRRegressor(...).fit(X, y)`; scikit-learn is not a dependency (the protocol is
duck-typed), so `clone`, `train_test_split` and `cross_val_score` work:

```python
from rsymbolic2 import SymbolicRegressor

model = SymbolicRegressor(binary_ops=["add", "sub", "mul"], seed=1).fit(X_train, y_train)
model.score(X_test, y_test)   # held-out R^2
model.result_.recommended     # the full search result is still there
```

## Licence and attribution

Defaults are matched to PySR's documented defaults; only the implementation differs
(a C++ engine with no Julia runtime, whose search core depends only on the C++
standard library). rsymbolic2 is an independent re-implementation and is **not
affiliated with or endorsed by PySR / SymbolicRegression.jl**. It is licensed under
the Apache License 2.0; see the [NOTICE](NOTICE) file for attribution.
