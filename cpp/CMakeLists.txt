cmake_minimum_required(VERSION 3.22)

# EDGEVISION_CPU_ONLY=ON builds the GPU-free part of the runtime (edgevision_cpu: tracker,
# metrics, class names, CLI parsing, JSON report) and its tests. It needs nothing but a C++17
# compiler, so the CI runs it on a plain Ubuntu runner; the default OFF builds the full
# TensorRT/NVDEC runtime on top of it.
option(EDGEVISION_CPU_ONLY "Build only edgevision_cpu and its tests (no CUDA, TensorRT, OpenCV, libav)" OFF)

if(EDGEVISION_CPU_ONLY)
  project(edgevision_trt LANGUAGES CXX)
else()
  project(edgevision_trt LANGUAGES CXX CUDA)
endif()

# Target GPU: T4 = 75 (default). Jetson Orin = 87, Ampere A10G = 86, L4 = 89.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# --- GPU-free library: built, warned strictly and tested everywhere ------------------------
set(EDGEVISION_STRICT_WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion)

add_library(edgevision_cpu STATIC
  src/tracker.cpp
  src/metrics.cpp
  src/names.cpp
  src/cli.cpp
  src/report.cpp)
target_include_directories(edgevision_cpu PUBLIC include)
target_compile_options(edgevision_cpu PRIVATE ${EDGEVISION_STRICT_WARNINGS})

foreach(suite tracker cli report)
  add_executable(test_${suite} tests/test_${suite}.cpp)
  target_link_libraries(test_${suite} PRIVATE edgevision_cpu)
  target_compile_options(test_${suite} PRIVATE ${EDGEVISION_STRICT_WARNINGS})
  add_test(NAME ${suite} COMMAND test_${suite})
endforeach()

if(EDGEVISION_CPU_ONLY)
  message(STATUS "EDGEVISION_CPU_ONLY: building edgevision_cpu and its tests only")
  return()
endif()

# --- full runtime: CUDA, TensorRT, OpenCV, libav (NVDEC) ---------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/local/tensorrt/include /usr/include)
find_library(NVINFER_LIBRARY nvinfer
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/tensorrt/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIBRARY)
  message(FATAL_ERROR "TensorRT not found (NvInfer.h / libnvinfer). Build inside docker/Dockerfile.tensorrt.")
endif()
message(STATUS "TensorRT headers: ${TENSORRT_INCLUDE_DIR}, lib: ${NVINFER_LIBRARY}")

add_library(edgevision_core STATIC
  src/letterbox.cu
  src/trt_engine.cpp
  src/detector.cpp
  src/batch_pipeline.cpp
  src/video_decoder.cpp
  src/stream_runner.cpp)
target_include_directories(edgevision_core PUBLIC include ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(edgevision_core PUBLIC
  edgevision_cpu ${NVINFER_LIBRARY} CUDA::cudart ${OpenCV_LIBS} PkgConfig::LIBAV)
target_compile_options(edgevision_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

find_package(Threads REQUIRED)
add_executable(edgevision_trt src/main.cpp)
target_link_libraries(edgevision_trt PRIVATE edgevision_core Threads::Threads)

add_executable(test_letterbox tests/test_letterbox.cpp)
target_link_libraries(test_letterbox PRIVATE edgevision_core)
add_test(NAME letterbox COMMAND test_letterbox)
add_executable(test_nv12 tests/test_nv12.cpp)
target_link_libraries(test_nv12 PRIVATE edgevision_core)
add_test(NAME nv12 COMMAND test_nv12)
