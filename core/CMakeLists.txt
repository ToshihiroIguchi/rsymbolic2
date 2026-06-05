# Core C++ library. The walking skeleton currently contains the constant-optimization
# subsystem only; further subsystems (expression, evolution, search) will be added as
# source files to this target.
add_library(rsymbolic2_core
    src/optimization/random_restart_optimizer.cpp
    src/optimization/optimizer_factory.cpp
)

target_include_directories(rsymbolic2_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(rsymbolic2_core PUBLIC cxx_std_17)

if(RSYMBOLIC2_BUILD_TESTS)
    add_executable(test_random_restart_optimizer
        tests/test_random_restart_optimizer.cpp)
    target_link_libraries(test_random_restart_optimizer
        PRIVATE rsymbolic2_core)
    add_test(NAME random_restart_optimizer
        COMMAND test_random_restart_optimizer)
endif()
