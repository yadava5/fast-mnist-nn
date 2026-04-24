cmake_minimum_required(VERSION 3.20)

project(fast_mnist_nn VERSION 0.1.0 LANGUAGES CXX)

option(FAST_MNIST_ENABLE_OPENMP "Enable OpenMP support" ON)
option(FAST_MNIST_ENABLE_NATIVE "Enable -march=native" OFF)
option(FAST_MNIST_ENABLE_BENCHMARKS "Enable Google Benchmark" OFF)
option(FAST_MNIST_ENABLE_DOXYGEN "Enable Doxygen docs target" OFF)

# When configured with emcmake (toolchain = Emscripten.cmake), skip
# the native executables, OpenMP search, and test harness. The WASM
# target lives at the bottom of this file and is the only thing
# emitted in that configuration.
if(EMSCRIPTEN)
  set(FAST_MNIST_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
  set(FAST_MNIST_ENABLE_SERVER OFF CACHE BOOL "" FORCE)
  set(FAST_MNIST_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
  # Emscripten has no std::filesystem threads model; disable the
  # test / benchmark / CLI targets and just build the library plus
  # the Embind shim.
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
endif()

add_library(fast_mnist
  src/Matrix.cpp
  src/NeuralNet.cpp
)

target_include_directories(fast_mnist
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(fast_mnist PUBLIC cxx_std_17)

if(MSVC)
  target_compile_options(fast_mnist PRIVATE /W4)
else()
  target_compile_options(fast_mnist PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(FAST_MNIST_ENABLE_NATIVE)
  if(MSVC)
    message(WARNING "FAST_MNIST_ENABLE_NATIVE is not supported on MSVC")
  else()
    target_compile_options(fast_mnist PRIVATE -march=native)
  endif()
endif()

if(FAST_MNIST_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(fast_mnist PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(fast_mnist PUBLIC FAST_MNIST_USE_OPENMP=1)
  elseif(APPLE)
    find_library(OPENMP_LIB omp
      PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
    )
    if(OPENMP_LIB)
      target_compile_options(fast_mnist PUBLIC -Xpreprocessor -fopenmp)
      target_include_directories(fast_mnist PUBLIC
        /opt/homebrew/opt/libomp/include
        /usr/local/opt/libomp/include
      )
      target_link_libraries(fast_mnist PUBLIC ${OPENMP_LIB})
      target_compile_definitions(fast_mnist PUBLIC FAST_MNIST_USE_OPENMP=1)
    else()
      message(STATUS "OpenMP not found; building without OpenMP")
    endif()
  else()
    message(STATUS "OpenMP not found; building without OpenMP")
  endif()
endif()

if(NOT EMSCRIPTEN)
  add_executable(fast_mnist_cli apps/fast_mnist_cli.cpp)
  target_link_libraries(fast_mnist_cli PRIVATE fast_mnist)

  # Model trainer
  add_executable(fast_mnist_trainer apps/train_model.cpp)
  target_link_libraries(fast_mnist_trainer PRIVATE fast_mnist)

  # Binary weight exporter: converts ASCII model.weights -> compact
  # little-endian .bin for loading in the WASM browser path.
  add_executable(export_weights apps/export_weights.cpp)
  target_link_libraries(export_weights PRIVATE fast_mnist)
endif()

# Web API Server
option(FAST_MNIST_ENABLE_SERVER "Enable HTTP API server" ON)
if(FAST_MNIST_ENABLE_SERVER AND NOT EMSCRIPTEN)
  add_executable(fast_mnist_server apps/server.cpp)
  target_include_directories(fast_mnist_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
  target_link_libraries(fast_mnist_server PRIVATE fast_mnist)
  
  # Find threads library (required by httplib)
  find_package(Threads REQUIRED)
  target_link_libraries(fast_mnist_server PRIVATE Threads::Threads)
  
  # macOS needs additional frameworks for networking
  if(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    find_library(SECURITY_LIBRARY Security)
    if(COREFOUNDATION_LIBRARY AND SECURITY_LIBRARY)
      target_link_libraries(fast_mnist_server PRIVATE ${COREFOUNDATION_LIBRARY} ${SECURITY_LIBRARY})
    endif()
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2
  )
  FetchContent_MakeAvailable(Catch2)

  add_executable(fast_mnist_tests
    tests/test_matrix.cpp
    tests/test_neural_net.cpp
  )
  target_link_libraries(fast_mnist_tests
    PRIVATE
      fast_mnist
      Catch2::Catch2WithMain
  )

  include(Catch)
  catch_discover_tests(fast_mnist_tests)
endif()

if(FAST_MNIST_ENABLE_BENCHMARKS)
  include(FetchContent)
  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  add_executable(fast_mnist_benchmarks
    benchmarks/bench_matrix.cpp
  )
  target_link_libraries(fast_mnist_benchmarks
    PRIVATE fast_mnist benchmark::benchmark
  )
endif()

if(FAST_MNIST_ENABLE_DOXYGEN)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    add_custom_target(docs
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Generating Doxygen documentation"
      VERBATIM
    )
  else()
    message(STATUS "Doxygen not found; docs target disabled")
  endif()
endif()

# -------------------------------------------------------------------
# Browser / WebAssembly target
#
# Produces fast_mnist.{js,wasm} when CMake is configured through the
# Emscripten toolchain (emcmake). The JS file is an ES module wrapper
# that lazily loads the wasm; both end up in the web/public/wasm/
# directory via tools/build_wasm.sh. RTTI stays on because Embind's
# class_<>::function() uses dynamic_cast under the hood; turning it
# off triggers link errors even for a trivial wrapper.
# -------------------------------------------------------------------
if(EMSCRIPTEN)
  add_executable(fast_mnist_wasm
    src/Matrix.cpp
    src/NeuralNet.cpp
    wasm/wasm_bindings.cpp
  )
  target_include_directories(fast_mnist_wasm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_compile_features(fast_mnist_wasm PRIVATE cxx_std_17)
  target_compile_options(fast_mnist_wasm PRIVATE
    -O3
    -msimd128
    # Exceptions + RTTI stay on: Matrix throws std::bad_alloc and
    # NeuralNet::loadBinary throws std::runtime_error on malformed
    # payloads, both of which are valid ways to signal failure back
    # to JS. Embind also relies on RTTI for dynamic_cast.
  )
  # Embind hooks via --bind; module wrapping + memory growth so the
  # browser can ask for more heap if a future model outgrows 16 MB.
  target_link_options(fast_mnist_wasm PRIVATE
    --bind
    -O3
    -sMODULARIZE=1
    -sEXPORT_NAME=createFastMnist
    -sENVIRONMENT=web
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=16MB
    -sSTACK_SIZE=1MB
    -sEXPORT_ES6=1
    -sUSE_ES6_IMPORT_META=1
  )
  set_target_properties(fast_mnist_wasm PROPERTIES
    OUTPUT_NAME "fast_mnist"
    SUFFIX ".js"
  )
endif()
