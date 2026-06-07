# rsymbolic2 0.1.0

Initial release.

- Symbolic regression via steady-state genetic programming, implemented in C++ and
  exposed through `symbolic_regression()`.
- Levenberg-Marquardt constant optimization (header-only Eigen backend) with
  random-restart fallback.
- OpenMP island model with inter-island migration; serial fallback when OpenMP is
  absent.
- Operators: `+`, `-`, `*`, `/`, `exp`, `log`, `sin`, `cos`, `sqrt`, `tanh`, `abs`.
- `predict()`, `print()`, and `plot()` methods for fitted models.
