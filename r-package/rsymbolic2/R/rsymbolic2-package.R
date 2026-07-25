# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
# Defaults matched to PySR / SymbolicRegression.jl (Apache-2.0); see NOTICE.

#' @useDynLib rsymbolic2, .registration = TRUE
"_PACKAGE"

# Data-frame column names referenced by non-standard evaluation inside
# ggplot2::aes() in the plot.rsymbolic2() helpers -- the Pareto front's own
# columns (pareto_plot), the frame fit_plot assembles, and the node/edge table
# tree_plot builds. Declared here so R CMD check does not flag them as undefined
# global variables.
utils::globalVariables(c(
    "complexity", "loss", "expression",
    "feature", "observed", "predicted",
    "x", "y", "xend", "yend", "drawn", "kind"
))
