# Core C++ library. The walking skeleton currently contains the constant-optimization
# subsystem only; further subsystems (expression, evolution, search) will be added as
# source files to this target.
add_library(rsymbolic2_core
    src/optimization/random_restart_optimizer.cpp
    src/optimization/eigen_lm_optimizer.cpp
    src/optimization/optimizer_factory.cpp
    src/evolution/random_tree.cpp
    src/evolution/mutation.cpp
    src/evolution/hall_of_fame.cpp
    src/search/evolutionary_search.cpp
)

target_include_directories(rsymbolic2_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(rsymbolic2_core PUBLIC cxx_std_17)

# Eigen is used only inside eigen_lm_optimizer.cpp, so it is a PRIVATE dependency and
# does not leak into the public interface or onto consumers of rsymbolic2_core.
target_link_libraries(rsymbolic2_core PRIVATE Eigen3::Eigen)

if(RSYMBOLIC2_BUILD_TESTS)
    add_executable(test_random_restart_optimizer
        tests/test_random_restart_optimizer.cpp)
    target_link_libraries(test_random_restart_optimizer
        PRIVATE rsymbolic2_core)
    add_test(NAME random_restart_optimizer
        COMMAND test_random_restart_optimizer)

    add_executable(test_eigen_lm_optimizer
        tests/test_eigen_lm_optimizer.cpp)
    target_link_libraries(test_eigen_lm_optimizer
        PRIVATE rsymbolic2_core)
    add_test(NAME eigen_lm_optimizer
        COMMAND test_eigen_lm_optimizer)

    add_executable(test_dual tests/test_dual.cpp)
    target_link_libraries(test_dual PRIVATE rsymbolic2_core)
    add_test(NAME dual COMMAND test_dual)

    add_executable(test_tree_eval tests/test_tree_eval.cpp)
    target_link_libraries(test_tree_eval PRIVATE rsymbolic2_core)
    add_test(NAME tree_eval COMMAND test_tree_eval)

    add_executable(test_constant_fitting tests/test_constant_fitting.cpp)
    target_link_libraries(test_constant_fitting PRIVATE rsymbolic2_core)
    add_test(NAME constant_fitting COMMAND test_constant_fitting)

    add_executable(test_random_tree tests/test_random_tree.cpp)
    target_link_libraries(test_random_tree PRIVATE rsymbolic2_core)
    add_test(NAME random_tree COMMAND test_random_tree)

    add_executable(test_hall_of_fame tests/test_hall_of_fame.cpp)
    target_link_libraries(test_hall_of_fame PRIVATE rsymbolic2_core)
    add_test(NAME hall_of_fame COMMAND test_hall_of_fame)

    add_executable(test_evolutionary_search tests/test_evolutionary_search.cpp)
    target_link_libraries(test_evolutionary_search PRIVATE rsymbolic2_core)
    add_test(NAME evolutionary_search COMMAND test_evolutionary_search)
endif()
