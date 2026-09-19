cmake_minimum_required(VERSION 3.22)
project(edgevision_trt LANGUAGES CXX CUDA)

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

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)

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
  src/metrics.cpp
  src/names.cpp)
target_include_directories(edgevision_core PUBLIC include ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(edgevision_core PUBLIC
  ${NVINFER_LIBRARY} CUDA::cudart ${OpenCV_LIBS})
target_compile_options(edgevision_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

add_executable(edgevision_trt src/main.cpp)
target_link_libraries(edgevision_trt PRIVATE edgevision_core)

enable_testing()
add_executable(test_letterbox tests/test_letterbox.cpp)
target_link_libraries(test_letterbox PRIVATE edgevision_core)
add_test(NAME letterbox COMMAND test_letterbox)
