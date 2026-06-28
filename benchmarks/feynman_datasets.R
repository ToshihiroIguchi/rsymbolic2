# Feynman symbolic regression benchmark: problem definitions and data generators.
#
# Reference: Udrescu & Tegmark (2020), "AI Feynman: A physics-inspired method for
# symbolic regression", Science Advances 6(16) eaay2631. Variable ranges from
# supplementary table S2.
#
# This file defines the 25-equation dev subset used for Stages 0 and 1 of the
# Feynman benchmark (docs/19). Stage 0 smoke test: equations with stage="smoke"
# (keys 1-10, pow-heavy). Stage 1 dev gate: all 25 equations.
# Full 117-equation set is handled separately when Stage 2 is run.
#
# Usage:
#   source("benchmarks/feynman_datasets.R")
#   ds <- feynman_dataset("gaussian", data_seed = 42L, split = "train")
#   # ds$X      : matrix of features  (1000 rows by default)
#   # ds$y      : numeric target vector
#   # ds$id     : "I.6.20a"
#   # ds$key    : "gaussian"
#   # ds$formula: "exp(-theta^2/2) / sqrt(2*pi)"
#   # ds$stage  : "smoke"
#
# Both train (seed DATA_SEED) and test (seed DATA_SEED+1) splits use the same
# variable domains and n. The split argument selects which seed is used.

feynman_problems <- list(

  # ---- Stage 0: smoke test (pow-heavy) ----------------------------------------

  gaussian = list(
    key     = "gaussian",
    id      = "I.6.20a",
    formula = "exp(-theta^2/2) / sqrt(2*pi)",
    n_vars  = 1L,
    domains = list(theta = c(1, 3)),
    fn      = function(x) exp(-x[1]^2 / 2) / sqrt(2 * pi),
    stage   = "smoke",
    ops     = c("square", "exp", "sqrt", "div")
  ),

  rel_mass = list(
    key     = "rel_mass",
    id      = "I.10.7",
    formula = "m0 / sqrt(1 - (v/c)^2)",
    n_vars  = 3L,
    domains = list(m0 = c(1, 5), v = c(1, 2), c = c(3, 10)),
    fn      = function(x) x[1] / sqrt(1 - (x[2] / x[3])^2),
    stage   = "smoke",
    ops     = c("div", "sqrt", "sub", "square")
  ),

  coulomb = list(
    key     = "coulomb",
    id      = "I.12.2",
    formula = "q1*q2 / (4*pi*eps*r^2)",
    n_vars  = 4L,
    domains = list(q1 = c(1, 5), q2 = c(1, 5), eps = c(1, 5), r = c(1, 5)),
    fn      = function(x) x[1] * x[2] / (4 * pi * x[3] * x[4]^2),
    stage   = "smoke",
    ops     = c("mul", "div", "square")
  ),

  spring_pe = list(
    key     = "spring_pe",
    id      = "I.14.4",
    formula = "0.5 * k * x^2",
    n_vars  = 2L,
    domains = list(k = c(1, 5), x = c(1, 5)),
    fn      = function(x) 0.5 * x[1] * x[2]^2,
    stage   = "smoke",
    ops     = c("mul", "square")
  ),

  lorentz_x = list(
    key     = "lorentz_x",
    id      = "I.15.1",
    formula = "(x - u*t) / sqrt(1 - (u/c)^2)",
    n_vars  = 4L,
    # x in [5,10] and u*t <= 2*2 = 4 ensures numerator >= 1
    domains = list(x = c(5, 10), u = c(1, 2), t = c(1, 2), c = c(3, 10)),
    fn      = function(x) (x[1] - x[2] * x[3]) / sqrt(1 - (x[2] / x[4])^2),
    stage   = "smoke",
    ops     = c("sub", "mul", "div", "sqrt", "square")
  ),

  rel_mom = list(
    key     = "rel_mom",
    id      = "I.15.10",
    formula = "m*v / sqrt(1 - (v/c)^2)",
    n_vars  = 3L,
    domains = list(m = c(1, 5), v = c(1, 2), c = c(3, 10)),
    fn      = function(x) x[1] * x[2] / sqrt(1 - (x[2] / x[3])^2),
    stage   = "smoke",
    ops     = c("mul", "div", "sqrt", "sub", "square")
  ),

  harmonic_ke = list(
    key     = "harmonic_ke",
    id      = "I.24.6",
    formula = "0.25 * m * (omega^2 + omega0^2) * x1^2",
    n_vars  = 4L,
    domains = list(m = c(1, 5), omega = c(1, 5), omega0 = c(1, 5), x1 = c(1, 5)),
    fn      = function(x) 0.25 * x[1] * (x[2]^2 + x[3]^2) * x[4]^2,
    stage   = "smoke",
    ops     = c("mul", "add", "square")
  ),

  interference = list(
    key     = "interference",
    id      = "I.37.4",
    formula = "I1 + I2 + 2*sqrt(I1*I2)*cos(delta)",
    n_vars  = 3L,
    domains = list(I1 = c(1, 5), I2 = c(1, 5), delta = c(1, 5)),
    fn      = function(x) x[1] + x[2] + 2 * sqrt(x[1] * x[2]) * cos(x[3]),
    stage   = "smoke",
    ops     = c("add", "sqrt", "mul", "cos")
  ),

  bohr_radius = list(
    key     = "bohr_radius",
    id      = "I.38.12",
    formula = "4*pi*eps0*hbar^2 / (m*q^2)",
    n_vars  = 4L,
    domains = list(eps0 = c(1, 5), hbar = c(1, 5), m = c(1, 5), q = c(1, 5)),
    fn      = function(x) 4 * pi * x[1] * x[2]^2 / (x[3] * x[4]^2),
    stage   = "smoke",
    ops     = c("mul", "div", "square")
  ),

  larmor_rad = list(
    key     = "larmor_rad",
    id      = "I.32.17",
    formula = "q^2 * a^2 / (6*pi*eps0*c^3)",
    n_vars  = 4L,
    domains = list(q = c(1, 5), a = c(1, 5), eps0 = c(1, 5), c = c(1, 5)),
    fn      = function(x) x[1]^2 * x[2]^2 / (6 * pi * x[3] * x[4]^3),
    stage   = "smoke",
    ops     = c("square", "mul", "div", "pow")
  ),

  # ---- Stage 1: dev gate additions --------------------------------------------

  planck = list(
    key     = "planck",
    id      = "I.41.16",
    formula = "hbar*omega^3 / (pi^2*c^2*(exp(hbar*omega/(k*T))-1))",
    n_vars  = 5L,
    # domains chosen so exp argument stays in [1/9, 9] to avoid exp overflow
    domains = list(hbar = c(1, 3), omega = c(1, 3), c = c(1, 3), k = c(1, 3), T = c(1, 3)),
    fn      = function(x) {
      x[1] * x[2]^3 / (pi^2 * x[3]^2 * (exp(x[1] * x[2] / (x[4] * x[5])) - 1))
    },
    stage   = "dev",
    ops     = c("mul", "div", "square", "pow", "exp", "sub")
  ),

  center_mass = list(
    key     = "center_mass",
    id      = "I.18.4",
    formula = "(m1*r1 + m2*r2) / (m1 + m2)",
    n_vars  = 4L,
    domains = list(m1 = c(1, 5), r1 = c(1, 5), m2 = c(1, 5), r2 = c(1, 5)),
    fn      = function(x) (x[1] * x[2] + x[3] * x[4]) / (x[1] + x[3]),
    stage   = "dev",
    ops     = c("mul", "add", "div")
  ),

  torque = list(
    key     = "torque",
    id      = "I.18.12",
    formula = "r * F * sin(theta)",
    n_vars  = 3L,
    domains = list(r = c(1, 5), F = c(1, 5), theta = c(1, 5)),
    fn      = function(x) x[1] * x[2] * sin(x[3]),
    stage   = "dev",
    ops     = c("mul", "sin")
  ),

  larmor_freq = list(
    key     = "larmor_freq",
    id      = "I.34.8",
    formula = "q*v*B / r",
    n_vars  = 4L,
    domains = list(q = c(1, 5), v = c(1, 5), B = c(1, 5), r = c(1, 5)),
    fn      = function(x) x[1] * x[2] * x[3] / x[4],
    stage   = "dev",
    ops     = c("mul", "div")
  ),

  doppler_rel = list(
    key     = "doppler_rel",
    id      = "I.34.14",
    formula = "omega0*(1 + v/c) / sqrt(1 - (v/c)^2)",
    n_vars  = 3L,
    domains = list(omega0 = c(1, 5), v = c(1, 2), c = c(3, 10)),
    fn      = function(x) x[1] * (1 + x[2] / x[3]) / sqrt(1 - (x[2] / x[3])^2),
    stage   = "dev",
    ops     = c("mul", "add", "div", "sqrt", "sub", "square")
  ),

  einstein_smol = list(
    key     = "einstein_smol",
    id      = "I.43.31",
    formula = "mob * kB * T",
    n_vars  = 3L,
    domains = list(mob = c(1, 5), kB = c(1, 5), T = c(1, 5)),
    fn      = function(x) x[1] * x[2] * x[3],
    stage   = "dev",
    ops     = c("mul")
  ),

  driven_osc = list(
    key     = "driven_osc",
    id      = "I.50.26",
    formula = "x1 * (cos(omega0*t) + alpha*cos(omega*t))",
    n_vars  = 5L,
    domains = list(x1 = c(1, 3), omega0 = c(1, 3), omega = c(1, 3), alpha = c(1, 3), t = c(1, 3)),
    fn      = function(x) x[1] * (cos(x[2] * x[5]) + x[4] * cos(x[3] * x[5])),
    stage   = "dev",
    ops     = c("mul", "add", "cos")
  ),

  heat_conduct = list(
    key     = "heat_conduct",
    id      = "II.2.42",
    formula = "kappa * (T2 - T1) / d",
    n_vars  = 4L,
    domains = list(kappa = c(1, 5), T2 = c(1, 5), T1 = c(1, 5), d = c(1, 5)),
    fn      = function(x) x[1] * (x[2] - x[3]) / x[4],
    stage   = "dev",
    ops     = c("mul", "sub", "div")
  ),

  boltzmann_dist = list(
    key     = "boltzmann_dist",
    id      = "II.11.3",
    formula = "n0 * exp(-m*g*x / (kB*T))",
    n_vars  = 6L,
    domains = list(n0 = c(1, 5), m = c(1, 5), g = c(1, 5), x = c(1, 5), kB = c(1, 5), T = c(1, 5)),
    fn      = function(x) x[1] * exp(-x[2] * x[3] * x[4] / (x[5] * x[6])),
    stage   = "dev",
    # 6-variable; expected to be hard — included to characterise, not to gate on
    ops     = c("mul", "div", "exp", "neg")
  ),

  clausius_moss = list(
    key     = "clausius_moss",
    id      = "II.11.27",
    formula = "n0*alpha / (1 - n0*alpha/3)",
    n_vars  = 2L,
    # n0*alpha/3 must be < 1: with both in [0.5,1.5] the product is <= 2.25 and /3 <= 0.75 < 1
    domains = list(n0 = c(0.5, 1.5), alpha = c(0.5, 1.5)),
    fn      = function(x) x[1] * x[2] / (1 - x[1] * x[2] / 3),
    stage   = "dev",
    ops     = c("mul", "div", "sub")
  ),

  clausius_moss2 = list(
    key     = "clausius_moss2",
    id      = "II.11.28",
    formula = "1 + n0*alpha / (1 - n0*alpha/3)",
    n_vars  = 2L,
    domains = list(n0 = c(0.5, 1.5), alpha = c(0.5, 1.5)),
    fn      = function(x) 1 + x[1] * x[2] / (1 - x[1] * x[2] / 3),
    stage   = "dev",
    ops     = c("add", "mul", "div", "sub")
  ),

  bohr_magneton = list(
    key     = "bohr_magneton",
    id      = "II.34.29a",
    formula = "q*hbar / (2*m)",
    n_vars  = 3L,
    domains = list(q = c(1, 5), hbar = c(1, 5), m = c(1, 5)),
    fn      = function(x) x[1] * x[2] / (2 * x[3]),
    stage   = "dev",
    ops     = c("mul", "div")
  ),

  bose_einstein = list(
    key     = "bose_einstein",
    id      = "III.4.33",
    formula = "hbar*omega / (exp(hbar*omega/(kB*T)) - 1)",
    n_vars  = 4L,
    domains = list(hbar = c(1, 3), omega = c(1, 3), kB = c(1, 3), T = c(1, 3)),
    fn      = function(x) x[1] * x[2] / (exp(x[1] * x[2] / (x[3] * x[4])) - 1),
    stage   = "dev",
    ops     = c("mul", "div", "exp", "sub")
  ),

  lens_eq = list(
    key     = "lens_eq",
    id      = "I.27.18",
    formula = "d1*d2 / (d2 + n*d1)",
    n_vars  = 3L,
    domains = list(d1 = c(1, 5), d2 = c(1, 5), n = c(1, 5)),
    fn      = function(x) x[1] * x[2] / (x[2] + x[3] * x[1]),
    stage   = "dev",
    ops     = c("mul", "div", "add")
  ),

  newtons_grav = list(
    key     = "newtons_grav",
    id      = "I.9.18s",
    formula = "G*m1*m2 / (dx^2 + dy^2)",
    n_vars  = 5L,
    # dx, dy in [1,3] so denominator >= 2, always positive
    domains = list(G = c(1, 5), m1 = c(1, 5), m2 = c(1, 5), dx = c(1, 3), dy = c(1, 3)),
    fn      = function(x) x[1] * x[2] * x[3] / (x[4]^2 + x[5]^2),
    stage   = "dev",
    ops     = c("mul", "div", "add", "square")
  )

)

# Ordered lists for deterministic iteration.
feynman_smoke_keys <- c(
  "gaussian", "rel_mass", "coulomb", "spring_pe", "lorentz_x",
  "rel_mom", "harmonic_ke", "interference", "bohr_radius", "larmor_rad"
)

feynman_dev_keys <- c(
  feynman_smoke_keys,
  "planck", "center_mass", "torque", "larmor_freq", "doppler_rel",
  "einstein_smol", "driven_osc", "heat_conduct", "boltzmann_dist",
  "clausius_moss", "clausius_moss2", "bohr_magneton", "bose_einstein",
  "lens_eq", "newtons_grav"
)

# Generate a dataset for a Feynman problem.
#
# @param key       Character: one of the keys in feynman_problems.
# @param n         Number of data points (default 1000; standard for Feynman).
# @param data_seed RNG seed. Use DATA_SEED (42L) for training, DATA_SEED+1 (43L)
#                  for the held-out test split. Passing either seed gives the same
#                  domain distribution; only the RNG state differs.
# @param split     Label attached to the returned list ("train" or "test").
#
# @return List: $X (n × p matrix), $y (length-n vector), $key, $id, $formula,
#               $n_vars, $stage, $split.
feynman_dataset <- function(key, n = 1000L, data_seed = 42L, split = "train") {
  prob <- feynman_problems[[key]]
  if (is.null(prob)) stop("Unknown Feynman problem key: ", key)

  set.seed(data_seed)
  p <- prob$n_vars
  dom <- prob$domains
  X <- matrix(NA_real_, nrow = n, ncol = p)
  for (j in seq_len(p)) {
    lo <- dom[[j]][1]
    hi <- dom[[j]][2]
    X[, j] <- runif(n, lo, hi)
  }

  y <- apply(X, 1, prob$fn)

  list(
    X       = X,
    y       = y,
    key     = prob$key,
    id      = prob$id,
    formula = prob$formula,
    n_vars  = prob$n_vars,
    stage   = prob$stage,
    split   = split
  )
}

`%||%` <- function(x, y) if (is.null(x)) y else x
