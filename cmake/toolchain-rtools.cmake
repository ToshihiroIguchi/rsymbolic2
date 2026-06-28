# CMake toolchain file for Rtools45 (x86_64-w64-mingw32.static.posix, GCC/UCRT).
# Usage:
#   cmake -B build-mingw -G Ninja \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-rtools.cmake \
#         -DCMAKE_BUILD_TYPE=Release
#
# Override the root if Rtools is installed elsewhere:
#   cmake ... -DRTOOLS_ROOT=C:/rtools45

set(RTOOLS_ROOT "C:/rtools45" CACHE PATH "Rtools installation root")

set(_tc_bin "${RTOOLS_ROOT}/x86_64-w64-mingw32.static.posix/bin")

set(CMAKE_C_COMPILER   "${_tc_bin}/gcc.exe"  CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${_tc_bin}/g++.exe"  CACHE FILEPATH "C++ compiler")
set(CMAKE_AR           "${_tc_bin}/ar.exe"   CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB       "${_tc_bin}/ranlib.exe" CACHE FILEPATH "Ranlib")

# Target: Windows/MinGW with UCRT (same ABI that R 4.2+ uses on Windows).
set(CMAKE_SYSTEM_NAME Windows)

unset(_tc_bin)
