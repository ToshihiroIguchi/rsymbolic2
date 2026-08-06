# CMake build for the rsymbolic2 Python extension (driven by scikit-build-core).
#
# Like the standalone C++ harness, this references the shared algorithmic core in
# r-package/rsymbolic2/src/ directly — that directory is the single source of truth
# for all non-binding code. Nothing is duplicated in the repository; only the
# pybind11 bridge (src/rsymbolic2_py.cpp) is Python-specific.
#
# An sdist is the one place where that cannot hold, and the reason is structural, not
# fixable by configuration. A CRAN package must be self-contained, so the core has to
# live inside r-package/; a PEP 517 sdist is built from THIS directory and cannot
# reach above it (`sdist.include = ["../r-package/..."]` was tried and is silently
# ignored). So the core has to be physically present in two distribution trees, and
# the only real choice is WHEN the copy happens: committed into git, or produced at
# sdist time. This file produces it at sdist time — see the staging block below —
# which keeps the repository free of a duplicated engine that every core change would
# have to touch twice (docs/85).

cmake_minimum_required(VERSION 3.20)
project(rsymbolic2_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Where the core is depends on what we are being built from. A repository checkout has
# it in r-package/, with the .cpp list in cmake/CoreSources.cmake next to it; an
# unpacked sdist has neither, and carries the staged copy of both instead (the staging
# block below writes them together, so the list and the files it names always travel as
# a pair). The checkout is preferred whenever both exist, so a developer always
# compiles the real source of truth and never a stale staged copy.
set(CORE_CHECKOUT "${CMAKE_CURRENT_SOURCE_DIR}/../r-package/rsymbolic2/src")
set(CORE_STAGED "${CMAKE_CURRENT_SOURCE_DIR}/_core_src")
if(EXISTS "${CORE_CHECKOUT}/evolutionary_search.cpp")
    set(CORE_SRC "${CORE_CHECKOUT}")
    set(CORE_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CoreSources.cmake")
elseif(EXISTS "${CORE_STAGED}/evolutionary_search.cpp")
    set(CORE_SRC "${CORE_STAGED}")
    set(CORE_LIST "${CORE_STAGED}/CoreSources.cmake")
else()
    message(FATAL_ERROR
        "Shared C++ core not found.\n"
        "Looked in ${CORE_CHECKOUT} (repository checkout) and ${CORE_STAGED} (sdist).\n"
        "If this is a git checkout, build from the full repository:\n"
        "    pip install pybind11 scikit-build-core numpy\n"
        "    pip install --no-build-isolation ./python")
endif()
if(NOT EXISTS "${CORE_LIST}")
    message(FATAL_ERROR
        "Shared C++ core found in ${CORE_SRC}, but its source list is missing:\n"
        "    ${CORE_LIST}\n"
        "A checkout needs cmake/CoreSources.cmake from the repository root; an sdist "
        "carries a copy of it beside the staged sources.")
endif()

# RSYMBOLIC2_CORE_CPP (the .cpp file names) and rsymbolic2_core_sources() come from
# here — the same list the standalone and WebAssembly builds compile. The staging copy
# below and the extension target further down both consume it, so the set that ships in
# an sdist and the set that compiles cannot drift. The R-only translation units
# (cpp11.cpp, rsymbolic2_r.cpp) are deliberately not part of it.
include("${CORE_LIST}")
rsymbolic2_core_sources("${CORE_SRC}" CORE_SOURCES)

# --- Staging the core into the sdist ---------------------------------------------------
# scikit-build-core runs this configure step before it collects the sdist's files
# (sdist.cmake = true in pyproject.toml), so a directory written here is picked up by
# sdist.include. Gated on SKBUILD_STATE so an ordinary `pip install ./python` does not
# write into the source tree: only the sdist build does.
#
# The copy is explicit rather than a whole-directory glob. A glob would sweep in the
# R-only sources, the Makevars, and any .o/.so left in src/ by a local R build — an
# earlier attempt did exactly that and produced an sdist carrying object files.
if(SKBUILD_STATE STREQUAL "sdist" AND CORE_SRC STREQUAL CORE_CHECKOUT)
    message(STATUS "rsymbolic2: staging the shared C++ core into ${CORE_STAGED} for the sdist")
    file(REMOVE_RECURSE "${CORE_STAGED}")
    file(MAKE_DIRECTORY "${CORE_STAGED}")
    foreach(cpp IN LISTS RSYMBOLIC2_CORE_CPP)
        file(COPY "${CORE_CHECKOUT}/${cpp}" DESTINATION "${CORE_STAGED}")
    endforeach()
    # The list travels with the sources it names: an unpacked sdist has no
    # cmake/CoreSources.cmake above it, and sdist.include = ["_core_src/**"] picks this
    # copy up along with the .cpp files.
    file(COPY "${CORE_LIST}" DESTINATION "${CORE_STAGED}")
    # The header tree is the rest of the engine; take it whole, headers only.
    file(COPY "${CORE_CHECKOUT}/rsymbolic" DESTINATION "${CORE_STAGED}"
         FILES_MATCHING PATTERN "*.hpp")
endif()

# --- pybind11 ------------------------------------------------------------------------
# scikit-build-core puts the build-time pybind11 (a build dependency declared in
# pyproject.toml) on CMAKE_PREFIX_PATH, so find_package locates it without a system
# install.
find_package(pybind11 CONFIG REQUIRED)

# --- Extension module ----------------------------------------------------------------
# There is no third-party C++ dependency: the constant optimizer is the self-contained
# self-LM (pure C++/STL), so there is no Eigen to find, fetch, or link. CORE_SOURCES was
# expanded (and checked against CORE_SRC) above.
pybind11_add_module(_core src/rsymbolic2_py.cpp ${CORE_SOURCES})
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
