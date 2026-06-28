# rsymbolic2 (Python)

Python bindings for the rsymbolic2 native symbolic-regression engine. The full
documentation — installation, tutorial, worked examples, and the algorithm
description with references — lives in the
[repository README](https://github.com/ToshihiroIguchi/rsymbolic2#readme).

Quick start:

```python
import numpy as np
from rsymbolic2 import symbolic_regression

X = np.linspace(-3, 3, 40).reshape(-1, 1)
y = 2.5 * X[:, 0] ** 2 - 1.3

res = symbolic_regression(X, y, unary_ops=["square"], seed=1)
print(res.expression)
print(res.predict(np.array([[0.0], [1.0]])))
```

Defaults are matched to PySR's documented defaults; only the implementation differs
(a C++ engine with no Julia runtime, whose search core depends only on the C++
standard library). rsymbolic2 is an independent re-implementation and is **not
affiliated with or endorsed by PySR / SymbolicRegression.jl**. It is licensed under
the Apache License 2.0; see the [NOTICE](NOTICE) file for attribution.
