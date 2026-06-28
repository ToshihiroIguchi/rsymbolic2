# Standalone C++ build harness (tests + benchmark executable).
#
# The shared algorithmic core lives in r-package/rsymbolic2/src/ — that
# directory is the single source of truth for all non-R code. This file
# references it directly; it does not duplicate or copy any sources.
#
# Benchmark-only code (problems.hpp, runner.hpp, runner.cpp) lives here
# in standalone/benchmarks/ and is not distributed in the R package.

set(CORE_SRC "${CMAKE_SOURCE_DIR}/r-package/rsymbolic2/src")
set(BENCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")

# rsymbolic2_core: shared algorithmic core — identical to what the R package
# compiles via Makevars. No third-party C++ dependency: the constant optimizer is
# the self-contained self-LM (pure C++/STL).
add_library(rsymbolic2_core
    ${CORE_SRC}/random_restart_optimizer.cpp
    ${CORE_SRC}/self_lm_optimizer.cpp
    ${CORE_SRC}/optimizer_factory.cpp
    ${CORE_SRC}/random_tree.cpp
    ${CORE_SRC}/mutation.cpp
    ${CORE_SRC}/crossover.cpp
    ${CORE_SRC}/hall_of_fame.cpp
    ${CORE_SRC}/simplify.cpp
    ${CORE_SRC}/evolutionary_search.cpp
)
target_include_directories(rsymbolic2_core PUBLIC ${CORE_SRC})
target_compile_features(rsymbolic2_core PUBLIC cxx_std_17)

find_package(Threads REQUIRED)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rsymbolic2_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# rsymbolic2_bench: benchmark harness (runner + problem suite).
# Not part of the R package; depends on core for the search headers.
add_library(rsymbolic2_bench STATIC ${BENCH_DIR}/runner.cpp)
target_include_directories(rsymbolic2_bench PUBLIC ${BENCH_DIR})
target_link_libraries(rsymbolic2_bench PUBLIC rsymbolic2_core)

if(RSYMBOLIC2_BUILD_TESTS)
    function(add_core_test name src)
        add_executable(test_${name} ${src})
        target_link_libraries(test_${name} PRIVATE rsymbolic2_core)
        add_test(NAME ${name} COMMAND test_${name})
    endfunction()

    add_core_test(random_restart_optimizer tests/test_random_restart_optimizer.cpp)
    add_core_test(self_lm_optimizer        tests/test_self_lm_optimizer.cpp)
    add_core_test(dual                     tests/test_dual.cpp)
    add_core_test(multi_dual               tests/test_multi_dual.cpp)
    add_core_test(soa_eval                 tests/test_soa_eval.cpp)
    add_core_test(tree_eval                tests/test_tree_eval.cpp)
    add_core_test(constant_fitting         tests/test_constant_fitting.cpp)
    add_core_test(random_tree              tests/test_random_tree.cpp)
    add_core_test(hall_of_fame             tests/test_hall_of_fame.cpp)
    add_core_test(evolutionary_search      tests/test_evolutionary_search.cpp)
    add_core_test(simplify                 tests/test_simplify.cpp)
    add_core_test(structural_mutation      tests/test_structural_mutation.cpp)
    add_core_test(mutation_operators       tests/test_mutation_operators.cpp)
    add_core_test(crossover                tests/test_crossover.cpp)
    add_core_test(multivariate             tests/test_multivariate.cpp)
    # Phase 1 interruptible-evaluation tests (docs/22 §5.4).
    add_core_test(optimizer_stop           tests/test_optimizer_stop.cpp)

    # benchmark_quick uses the benchmark harness, not just core.
    add_executable(test_benchmark_quick tests/test_benchmark_quick.cpp)
    target_link_libraries(test_benchmark_quick PRIVATE rsymbolic2_bench)
    add_test(NAME benchmark_quick COMMAND test_benchmark_quick)
    set_tests_properties(benchmark_quick PROPERTIES TIMEOUT 30)

    # Island-model: n_populations=1 regression + n=4 recovery + thread-count determinism.
    add_executable(test_island_model tests/test_island_model.cpp)
    target_link_libraries(test_island_model PRIVATE rsymbolic2_bench)
    add_test(NAME island_model COMMAND test_island_model)
    set_tests_properties(island_model PROPERTIES TIMEOUT 120)

    # warmup_maxsize_by: get_cur_maxsize formula + off-path regression + warmup determinism.
    add_executable(test_warmup_maxsize tests/test_warmup_maxsize.cpp)
    target_link_libraries(test_warmup_maxsize PRIVATE rsymbolic2_bench)
    add_test(NAME warmup_maxsize COMMAND test_warmup_maxsize)
    set_tests_properties(warmup_maxsize PROPERTIES TIMEOUT 60)
endif()

option(RSYMBOLIC2_BUILD_BENCHMARKS "Build the standalone benchmark executable" ON)
if(RSYMBOLIC2_BUILD_BENCHMARKS)
    add_executable(benchmark_main benchmarks/benchmark_main.cpp)
    target_link_libraries(benchmark_main PRIVATE rsymbolic2_bench)

    add_executable(bench_parallel benchmarks/bench_parallel.cpp)
    target_link_libraries(bench_parallel PRIVATE rsymbolic2_core)

    add_executable(bench_evolve benchmarks/bench_evolve.cpp)
    target_link_libraries(bench_evolve PRIVATE rsymbolic2_core Threads::Threads)

    # Hot-path phase profiler (docs/30). The per-phase breakdown only prints when the
    # core is compiled with RSYMBOLIC2_PROFILE; configure with
    # -DCMAKE_CXX_FLAGS=-DRSYMBOLIC2_PROFILE so both the core and this driver see it.
    add_executable(bench_profile benchmarks/bench_profile.cpp)
    target_link_libraries(bench_profile PRIVATE rsymbolic2_core)

    # SoA point-batched residual evaluator evidence micro-benchmark (docs/30 "SoA
    # point-batched evaluator"). Header-only; links core only for its include path.
    # For the -O2-vs-O3 / -march=native / -fopt-info-vec comparison, compile it directly
    # with the documented g++ command in the file header, not just this default target.
    add_executable(bench_soa_eval benchmarks/bench_soa_eval.cpp)
    target_link_libraries(bench_soa_eval PRIVATE rsymbolic2_core)

    # BFGS vs self-LM and Float32 vs Float64 constant-optimizer evidence harness
    # (docs/36). Header-only templated LM/BFGS driven by the production evaluate<T>;
    # links core only for its include path. As with bench_soa_eval, for the
    # -O2-vs-O3 comparison compile it directly with the g++ command in the file header.
    add_executable(bench_opt_precision benchmarks/bench_opt_precision.cpp)
    target_link_libraries(bench_opt_precision PRIVATE rsymbolic2_core)
endif()
