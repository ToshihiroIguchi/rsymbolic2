cmake_minimum_required(VERSION 3.20)

project(rsymbolic2
    VERSION 0.1.0
    DESCRIPTION "Native symbolic regression library (C++ core)"
    LANGUAGES CXX)

# Default to C++17 for the whole project. Portability outranks performance, so we keep
# to a standard, extension-free dialect that GCC (Rtools/MinGW), Clang, and MSVC all
# support.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RSYMBOLIC2_BUILD_TESTS "Build the test suite" ON)

enable_testing()

add_subdirectory(core)
