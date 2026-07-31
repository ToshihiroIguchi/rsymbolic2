# Differential test: the R transcription of the engine's guarded operators in
# diag_structural_audit.R vs. the C++ they are transcribed from (dual.hpp).
#
# Why this exists. diag_structural_audit.R decides whether a "recovered" expression is
# real structure or a threshold grinder by evaluating it OUTSIDE the training box, which
# is exactly where the guarded operators fire. It therefore has to reimplement safe_sqrt
# and safe_pow in R, and a reimplementation drifts: it kept returning 0 out of domain for
# some time after the engine moved to NaN (commit 8077edb, docs/69), which silently lets a
# candidate the engine would have rejected earn a verdict. This script is what makes the
# drift loud. Run it after any change to the operator semantics in dual.hpp.
#
# Method is docs/69's, one language over: emit the C++ answers over an edge-case grid
# (including +-Inf, +-0 and NaN, where the "obvious" simplification of the branch table is
# wrong) and compare case by case, treating NaN == NaN as agreement.
#
# Usage:
#   Rscript benchmarks/diag_audit_operator_parity.R
#   Rscript benchmarks/diag_audit_operator_parity.R cxx=/usr/bin/g++
# Needs a C++17 compiler: Rtools' g++ on Windows, the system g++ on Ubuntu. Exit status is
# 0 on agreement, 1 on any divergence.

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(name, default) {
  hit <- regmatches(args, regexpr(paste0("^", name, "=.*"), args))
  hit <- hit[nzchar(hit)]
  if (length(hit) == 0L) return(default)
  sub(paste0("^", name, "="), "", hit[[1L]])
}
CXX <- get_arg("cxx", Sys.which("g++"))
if (!nzchar(CXX)) stop("no C++ compiler found; pass cxx=<path to g++>")

script_dir <- tryCatch(dirname(sys.frame(1)$ofile), error = function(e) "benchmarks")
repo       <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
src_dir    <- file.path(repo, "r-package", "rsymbolic2", "src")
audit_r    <- file.path(script_dir, "diag_structural_audit.R")

# --- The authority: the engine's own functions, over an edge-case grid ------------------
emitter <- '
#include <cstdio>
#include <limits>
#include <vector>
#include "rsymbolic/expression/dual.hpp"
int main() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {-inf, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0,
                                    0.5, 1.0, 1.5, 2.0, 3.0, inf, nan};
    const std::vector<double> ys = {-inf, -3.0, -2.5, -2.0, -1.0, -0.5, -0.0, 0.0,
                                    0.5, 1.0, 1.5, 2.0, 2.0000001, 3.0, inf, nan};
    std::printf("op,x,y,out\\n");
    for (double x : xs) std::printf("sqrt,%.17g,0,%.17g\\n", x, rsymbolic::sqrt(x));
    for (double x : xs)
        for (double y : ys)
            std::printf("pow,%.17g,%.17g,%.17g\\n", x, y, rsymbolic::pow(x, y));
    return 0;
}
'
tmp <- tempfile("op_parity_"); dir.create(tmp)
cpp <- file.path(tmp, "emit_grid.cpp")
exe <- file.path(tmp, if (.Platform$OS.type == "windows") "emit_grid.exe" else "emit_grid")
writeLines(emitter, cpp)

# platform_libm.cpp carries the UCRT dispatch table libm.hpp declares (commit 42c235e);
# dual.hpp does not link without it.
status <- system2(CXX, c("-std=c++17", "-O0", "-I", shQuote(src_dir), shQuote(cpp),
                         shQuote(file.path(src_dir, "platform_libm.cpp")),
                         "-o", shQuote(exe)))
if (status != 0L) stop("compiling the grid emitter failed")
grid_csv <- file.path(tmp, "engine_grid.csv")
if (system2(exe, stdout = grid_csv) != 0L) stop("running the grid emitter failed")

# --- The transcription under test ------------------------------------------------------
# Only the two operator definitions are pulled out; sourcing the audit script would run
# the whole audit.
env <- new.env()
for (e in parse(audit_r)) {
  if (is.call(e) && identical(as.character(e[[1L]]), "<-") && is.name(e[[2L]]) &&
      as.character(e[[2L]]) %in% c("engine_sqrt", "engine_pow")) eval(e, env)
}
stopifnot(is.function(env$engine_sqrt), is.function(env$engine_pow))

g <- read.csv(grid_csv, colClasses = "character")
g$x <- as.numeric(g$x); g$y <- as.numeric(g$y); g$out <- as.numeric(g$out)

is_sqrt <- g$op == "sqrt"
got <- numeric(nrow(g))
got[is_sqrt]  <- env$engine_sqrt(g$x[is_sqrt])
got[!is_sqrt] <- env$engine_pow(g$x[!is_sqrt], g$y[!is_sqrt])

# NaN is a legitimate answer here, so it has to compare equal to itself.
agree <- (is.nan(got) & is.nan(g$out)) | (!is.nan(got) & !is.nan(g$out) & got == g$out)

cat(sprintf("cases %d (sqrt %d, pow %d)   agree %d   DIFFER %d\n",
            nrow(g), sum(is_sqrt), sum(!is_sqrt), sum(agree), sum(!agree)))
if (any(!agree)) {
  d <- g[!agree, ]; d$r <- got[!agree]
  print(d[, c("op", "x", "y", "out", "r")], row.names = FALSE)
  cat("FAIL: diag_structural_audit.R no longer replicates the engine.\n")
  quit(status = 1L)
}
cat("PASS: diag_structural_audit.R replicates the engine on every case.\n")
