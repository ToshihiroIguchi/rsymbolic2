# The shared algorithmic core's translation units, named once for every CMake build.
#
# The core itself lives in r-package/rsymbolic2/src/ — that directory is the single
# source of truth for all non-binding code, and no build copies it (the one exception,
# the Python sdist, is explained in python/CMakeLists.txt). This file is the single
# source of truth for the *list*: standalone/, python/ and web/wasm/ each used to spell
# out the same twelve file names by hand, so adding a core .cpp meant four edits.
#
# The R package is deliberately not a consumer, and that is the trap this file guards
# against: Makevars compiles every src/*.cpp, so a new core file needs no edit there and
# an R build succeeds while the other three are still missing it. The first build a
# developer tries is usually R. rsymbolic2_core_sources() below therefore checks the
# list against what is actually on disk and stops the configure step when they differ,
# turning a link error (or worse, a silently unused translation unit) into a message
# naming the file and this list (docs/86).
include_guard(GLOBAL)

# Order is preserved deliberately: it is the order the sources are handed to the
# compiler and linker, so keeping it fixed keeps object and link order identical to
# what every build produced before the list was centralised. The consistency check
# below compares sets, not sequences, so this order stays a human choice.
set(RSYMBOLIC2_CORE_CPP
    platform_libm.cpp
    random_restart_optimizer.cpp
    self_lm_optimizer.cpp
    optimizer_factory.cpp
    random_tree.cpp
    mutation.cpp
    crossover.cpp
    hall_of_fame.cpp
    simplify.cpp
    display_simplify.cpp
    egraph.cpp
    evolutionary_search.cpp
)

# R-binding-only translation units. They sit in the same directory because a CRAN
# package must be self-contained, but only the R build may compile them: cpp11.cpp is
# generated registration code and rsymbolic2_r.cpp includes cpp11.hpp, neither of which
# exists outside an R installation. They are excluded from the check below rather than
# from the list above, because the check is what has to know about them.
set(RSYMBOLIC2_R_ONLY_CPP cpp11.cpp rsymbolic2_r.cpp)

# rsymbolic2_core_sources(<dir> <outvar>)
#
# Verifies that RSYMBOLIC2_CORE_CPP still describes the .cpp files in <dir>, then sets
# <outvar> in the caller's scope to the list expanded to full paths under <dir>.
#
# Verification and expansion are one call on purpose: there is no way to obtain the
# source list while skipping the check, so the check cannot be forgotten by a build
# that is added later.
#
# CONFIGURE_DEPENDS is load bearing. It makes the build system re-run the glob before
# each build and re-configure when the result changed, so a core file added after this
# tree was configured fails at the next `cmake --build` instead of at link time. The
# usual objection to globbing does not apply here: the glob only feeds the check, never
# the source list, which stays explicit above. Object files and shared libraries left
# in src/ by a local R build do not match *.cpp, so they cause no re-configure churn.
function(rsymbolic2_core_sources dir outvar)
    file(GLOB _found CONFIGURE_DEPENDS "${dir}/*.cpp")
    set(_names "")
    foreach(_f IN LISTS _found)
        get_filename_component(_n "${_f}" NAME)
        list(APPEND _names "${_n}")
    endforeach()
    list(REMOVE_ITEM _names ${RSYMBOLIC2_R_ONLY_CPP})

    set(_expected ${RSYMBOLIC2_CORE_CPP})
    list(SORT _names)
    list(SORT _expected)
    if(NOT _names STREQUAL _expected)
        set(_unlisted ${_names})
        list(REMOVE_ITEM _unlisted ${_expected})
        set(_absent ${_expected})
        if(_names)
            list(REMOVE_ITEM _absent ${_names})
        endif()
        message(FATAL_ERROR
            "The core source list no longer matches ${dir}.\n"
            "  in src/ but not listed: ${_unlisted}\n"
            "  listed but not in src/: ${_absent}\n"
            "Fix RSYMBOLIC2_CORE_CPP in cmake/CoreSources.cmake (and "
            "RSYMBOLIC2_R_ONLY_CPP if the new file is an R-binding source).\n"
            "The R package needs no such edit — Makevars compiles every src/*.cpp — "
            "which is why an R build can succeed while this one is incomplete "
            "(docs/86).")
    endif()

    set(_sources "")
    foreach(_cpp IN LISTS RSYMBOLIC2_CORE_CPP)
        list(APPEND _sources "${dir}/${_cpp}")
    endforeach()
    set(${outvar} "${_sources}" PARENT_SCOPE)
endfunction()
