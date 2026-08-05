# CRAN's repository policy caps a check at two simultaneous threads, and the search
# defaults to every core (via omp_get_max_threads(), which reads OMP_NUM_THREADS). Set
# the cap here, before the package is loaded and any parallel region is entered, so it
# applies to every test without each one having to pass n_threads. The island model is
# bit-deterministic across thread counts, so this changes only how fast the tests run.
Sys.setenv(OMP_NUM_THREADS = "2")

library(testthat)
library(rsymbolic2)
test_check("rsymbolic2")
