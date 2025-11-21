cmake_minimum_required(VERSION 3.25)

# Set GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include(FetchContent)
project(standalone_attention CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch cuDNN frontend
FetchContent_Declare(
    cudnn-fe
    GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
    GIT_TAG v1.12.0
    EXCLUDE_FROM_ALL
)

# Find CUDA toolkit components
find_package(CUDAToolkit COMPONENTS cuBLAS REQUIRED)

FetchContent_MakeAvailable(cudnn-fe)

# Find cuDNN library
find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS $ENV{CUDNN_INCLUDE_PATH}
          /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include
          ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY cudnn
    HINTS $ENV{CUDNN_LIBRARY_PATH}
          /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
          ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib64 lib)

if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN library not found. Please set CUDNN_LIBRARY_PATH environment variable.")
endif()

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "cuDNN Include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "cuDNN Library: ${CUDNN_LIBRARY}")

# Create library with both attention implementations
add_library(attention_lib SHARED
    cudnn_att.cpp
    attention_cuda.cu
)

# Link libraries
target_link_libraries(attention_lib PRIVATE 
    cudnn_frontend 
    ${CUDNN_LIBRARY}
    CUDA::cudart
)

# Include directories
target_include_directories(attention_lib PRIVATE ${CUDNN_INCLUDE_DIR})
target_include_directories(attention_lib PUBLIC .)

# Compile options
target_compile_options(attention_lib PUBLIC 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# Example test executable (optional)
add_executable(test_attention EXCLUDE_FROM_ALL
    test_main.cpp
)

target_link_libraries(test_attention PRIVATE attention_lib)

# Installation
install(TARGETS attention_lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)