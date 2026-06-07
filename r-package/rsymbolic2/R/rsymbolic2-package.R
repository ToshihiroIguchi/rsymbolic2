#' @useDynLib rsymbolic2, .registration = TRUE
#' @importFrom Rcpp evalCpp
"_PACKAGE"

# Column names of the Pareto-front data frame, referenced by non-standard
# evaluation inside ggplot2::aes() in plot.rsymbolic2(). Declared here so
# R CMD check does not flag them as undefined global variables.
utils::globalVariables(c("complexity", "loss", "expression"))
