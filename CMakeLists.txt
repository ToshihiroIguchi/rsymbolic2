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

# --- Eigen dependency -------------------------------------------------------------
# Eigen is header-only. Prefer a system/installed Eigen (e.g. provided via RcppEigen
# in the eventual R build); otherwise fetch the headers. We only use the headers
# (including the unsupported NonLinearOptimization module), so we do not build Eigen.
find_package(Eigen3 3.3 NO_MODULE QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package; fetching Eigen 3.4.0 headers")
    include(FetchContent)
    # CMP0169 (>=3.30) deprecates calling FetchContent_Populate directly. We keep using
    # it intentionally: we want only the headers and must NOT run Eigen's own CMake.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    add_library(eigen_headers INTERFACE)
    target_include_directories(eigen_headers SYSTEM INTERFACE ${eigen_SOURCE_DIR})
    add_library(Eigen3::Eigen ALIAS eigen_headers)
endif()

enable_testing()

add_subdirectory(core)
