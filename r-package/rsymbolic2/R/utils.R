# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

# Pareto-front score, used by summary() and as.data.frame() for display.
#
# Reproduces the score the C++ core uses to pick the recommended model in
# select_best() (src/hall_of_fame.cpp): the loss-drop-per-complexity between a
# front member and its simpler predecessor,
#   score[i] = (log(loss[i-1]) - log(loss[i])) / (complexity[i] - complexity[i-1]),
# with losses floored at 1e-300 so an exact (zero-loss) fit yields a large finite
# score instead of -Inf. The first member has no predecessor, so its score is NA;
# a non-positive complexity step also yields NA. Computing this in R keeps the C++
# core unchanged and gives summary()/as.data.frame() the same score the engine
# used to choose `recommended`.
#
# Assumes the Pareto front is in ascending complexity order, as built by the C++
# bridge (rsymbolic2_r.cpp).
pareto_score <- function(complexity, loss) {
    n <- length(loss)
    score <- rep(NA_real_, n)
    if (n <= 1L) return(score)

    loss_floor <- 1e-300
    cur  <- pmax(loss[-1L],          loss_floor)
    prev <- pmax(loss[-n],           loss_floor)
    dc   <- complexity[-1L] - complexity[-n]

    s <- (log(prev) - log(cur)) / dc
    s[dc <= 0L] <- NA_real_
    score[-1L] <- s
    score
}
