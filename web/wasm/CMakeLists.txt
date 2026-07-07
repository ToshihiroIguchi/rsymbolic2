# WebAssembly (Emscripten) build of the rsymbolic2 core + the embind bridge.
#
# Configure with `emcmake cmake` so CMAKE_CXX_COMPILER is em++. This target compiles
# the SAME 9 core .cpp files the R package and Python extension compile (the shared
# source of truth in r-package/rsymbolic2/src/) plus the embind bridge — no source is
# copied. It is single-threaded on purpose: no OpenMP, no pthreads, so the output runs
# on plain static hosting (GitHub Pages) with no COOP/COEP cross-origin-isolation
# headers. The island model is bit-deterministic across thread counts, so results are
# identical to the OpenMP R/Python builds for a fixed seed (docs/37).
#
# Build (from repo root):
#   emcmake cmake -S web/wasm -B web/wasm/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build web/wasm/build
# Output rsymbolic2.js + rsymbolic2.wasm land in web/app/vendor/ (see OUTPUT_DIR below).

cmake_minimum_required(VERSION 3.20)
project(rsymbolic2_wasm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# The shared algorithmic core (identical list to standalone/CMakeLists.txt and
# python/CMakeLists.txt). CORE_SRC is the single source of truth; nothing is duplicated.
set(CORE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../r-package/rsymbolic2/src")

add_executable(rsymbolic2
    ${CMAKE_CURRENT_SOURCE_DIR}/rsymbolic2_wasm.cpp
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
target_include_directories(rsymbolic2 PRIVATE ${CORE_SRC})
# NOTE: RTTI must stay ON — embind uses typeid internally for type registration, so
# -fno-rtti would break the bindings. Exceptions are required (the unit parser / optimizer
# factory throw std::invalid_argument, caught at the bridge boundary).
target_compile_options(rsymbolic2 PRIVATE -O3 -fexceptions)

# Emscripten link flags. -fexceptions is required (the core throws std::invalid_argument
# from the unit parser / optimizer factory, caught at the bridge boundary). ES6 module +
# MODULARIZE lets worker.js `import` it; ALLOW_MEMORY_GROWTH copes with larger datasets.
target_link_options(rsymbolic2 PRIVATE
    -fexceptions
    -lembind
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=createRsymbolic2Module
    # Fixed heap, NO memory growth. With ALLOW_MEMORY_GROWTH the WASM memory becomes a
    # *resizable* ArrayBuffer once it grows, and the browser's TextDecoder.decode() rejects
    # views onto a resizable buffer ("must not be resizable"), which breaks reading result
    # strings (to_string / to_latex) after the heap grows. A fixed 128 MB heap keeps the
    # buffer non-resizable so string marshalling always works; 128 MB is far more than the
    # search needs (small trees + a modest dataset), and a dataset too large for it fails
    # clearly at allocation rather than mid-run. For very large data, use the R/Python build.
    -sINITIAL_MEMORY=134217728
    -sENVIRONMENT=web,worker
)

# Emit rsymbolic2.js (+ rsymbolic2.wasm) straight into the static site's vendor dir.
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../app/vendor")
set_target_properties(rsymbolic2 PROPERTIES
    OUTPUT_NAME "rsymbolic2"
    SUFFIX ".js"
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
)

# --- Node.js test variant --------------------------------------------------------
# A second build of the same sources targeted at Node.js (CommonJS, ENVIRONMENT=node)
# so the parity/smoke test (web/wasm/test/parity_test.cjs) can require() it directly —
# the web,worker ES6 artifact above does not load under Node. Same engine, same seed =>
# identical results; this target exists only to run that check in CI/dev, not to ship.
set(CORE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/rsymbolic2_wasm.cpp
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
add_executable(rsymbolic2_node ${CORE_SRCS})
target_include_directories(rsymbolic2_node PRIVATE ${CORE_SRC})
target_compile_options(rsymbolic2_node PRIVATE -O3 -fexceptions)
target_link_options(rsymbolic2_node PRIVATE
    -fexceptions
    -lembind
    -sMODULARIZE=1
    -sEXPORT_NAME=createRsymbolic2Module
    # Match the web target's fixed heap so the Node parity test exercises the same memory
    # configuration (and Node's TextDecoder is lenient about resizable buffers anyway).
    -sINITIAL_MEMORY=134217728
    -sENVIRONMENT=node
)
set_target_properties(rsymbolic2_node PROPERTIES
    OUTPUT_NAME "rsymbolic2_node"
    SUFFIX ".cjs"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/test"
)
