# CMake build for the rsymbolic2 Python extension (driven by scikit-build-core).
#
# Like the standalone C++ harness, this references the shared algorithmic core in
# r-package/rsymbolic2/src/ directly — that directory is the single source of truth
# for all non-binding code. We do NOT duplicate any core sources here; only the
# pybind11 bridge (src/rsymbolic2_py.cpp) is Python-specific.

cmake_minimum_required(VERSION 3.20)
project(rsymbolic2_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Path to the shared C++ core (identical to what the R package compiles via Makevars
# and what standalone/CMakeLists.txt links). Resolved relative to this file so an
# in-place build (pip install -e . / --no-build-isolation) from a cloned repo works.
set(CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../r-package/rsymbolic2/src")
if(NOT EXISTS "${CORE_SRC}/evolutionary_search.cpp")
    message(FATAL_ERROR
        "Shared C++ core not found at ${CORE_SRC}.\n"
        "Build the Python package from a full checkout of the rsymbolic2 repository "
        "with build isolation disabled, e.g.:\n"
        "    pip install pybind11 scikit-build-core numpy\n"
        "    pip install --no-build-isolation ./python")
endif()

# --- pybind11 ------------------------------------------------------------------------
# scikit-build-core puts the build-time pybind11 (a build dependency declared in
# pyproject.toml) on CMAKE_PREFIX_PATH, so find_package locates it without a system
# install.
find_package(pybind11 CONFIG REQUIRED)

# --- Extension module ----------------------------------------------------------------
# The core .cpp set is identical to standalone/CMakeLists.txt's rsymbolic2_core. There
# is no third-party C++ dependency: the constant optimizer is the self-contained
# self-LM (pure C++/STL), so there is no Eigen to find, fetch, or link.
pybind11_add_module(_core
    src/rsymbolic2_py.cpp
    ${CORE_SRC}/random_restart_optimizer.cpp
    ${CORE_SRC}/self_lm_optimizer.cpp
    ${CORE_SRC}/optimizer_factory.cpp
    ${CORE_SRC}/random_tree.cpp
    ${CORE_SRC}/mutation.cpp
    ${CORE_SRC}/crossover.cpp
    ${CORE_SRC}/hall_of_fame.cpp
    ${CORE_SRC}/simplify.cpp
    ${CORE_SRC}/display_simplify.cpp
    ${CORE_SRC}/egraph.cpp
    ${CORE_SRC}/evolutionary_search.cpp
)
target_include_directories(_core PRIVATE ${CORE_SRC})

# MSVC only: the core sources are UTF-8 without a BOM (non-ASCII appears in comments),
# which MSVC otherwise reads in the system code page and reports as C4819 on every file
# on a non-UTF-8 locale (e.g. CP932). /utf-8 fixes the source charset and silences it.
# GCC/Clang already assume UTF-8, so nothing changes for the R, Ubuntu or WASM builds.
if(MSVC)
    target_compile_options(_core PRIVATE /utf-8)
endif()

# OpenMP island-model parallelism. Optional: the core has a correct serial fallback
# when OpenMP is unavailable (e.g. some macOS toolchains), matching the R build.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the compiled module into the rsymbolic2 package directory.
install(TARGETS _core DESTINATION rsymbolic2)
