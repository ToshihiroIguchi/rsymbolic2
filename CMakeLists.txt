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

# The library has no third-party C++ dependency: the default (and only) constant
# optimizer is a self-contained Levenberg-Marquardt written in pure C++/STL. The
# former Eigen-based backend was removed (docs/40), so there is no Eigen to find or
# fetch here.

enable_testing()

add_subdirectory(standalone)
