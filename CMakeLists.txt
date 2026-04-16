cmake_minimum_required(VERSION 3.18)  # 3.18+ required for first-class CUDA
project(HybridRetrieval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Build options
# ============================================================================
option(ENABLE_CUDA    "Build CUDA GPU acceleration path"         OFF)
option(ENABLE_AVX2    "Enable AVX2 SIMD on x86 (requires -mavx2)" OFF)
option(ENABLE_AVX512  "Enable AVX-512 SIMD on x86 servers"        OFF)

# ============================================================================
# OpenMP
# ============================================================================
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP REQUIRED)

# ============================================================================
# CUDA (optional)
# ============================================================================
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # A100 = sm_80. Keep 70/75/80/90 for portability to V100/T4/A100/H100.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 75 80 90)
    endif()

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

    message(STATUS "CUDA enabled. Arch: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# ============================================================================
# Third-party: hnswlib (header-only)
# ============================================================================
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE ${CMAKE_SOURCE_DIR}/third_party/hnswlib)

# ============================================================================
# Core retrieval library
# ============================================================================
add_library(hybrid_retrieval STATIC
    src/tokenizer.cpp
    src/sparse_index.cpp
    src/dense_index.cpp
    src/fusion.cpp
    src/pipeline.cpp
    src/corpus_generator.cpp
)
target_include_directories(hybrid_retrieval PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(hybrid_retrieval PUBLIC OpenMP::OpenMP_CXX hnswlib)

if(ENABLE_AVX512)
    target_compile_options(hybrid_retrieval PRIVATE -mavx512f -mavx512dq -mavx512bw)
    target_compile_definitions(hybrid_retrieval PUBLIC HAS_AVX512=1)
elseif(ENABLE_AVX2)
    target_compile_options(hybrid_retrieval PRIVATE -mavx2 -mfma)
    target_compile_definitions(hybrid_retrieval PUBLIC HAS_AVX2=1)
endif()

# ============================================================================
# Agent memory library
# ============================================================================
add_library(agent_memory STATIC
    src/agent_corpus_generator.cpp
    src/memory_fusion.cpp
    src/memory_store.cpp
    src/temporal_index.cpp
    src/hierarchical_memory.cpp
)
target_include_directories(agent_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(agent_memory PUBLIC hybrid_retrieval)

# ============================================================================
# GPU library (optional — only when ENABLE_CUDA=ON)
# ============================================================================
if(ENABLE_CUDA)
    add_library(hybrid_retrieval_gpu STATIC
        src/sparse_index_gpu.cu
    )
    target_include_directories(hybrid_retrieval_gpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(hybrid_retrieval_gpu
        PUBLIC  hybrid_retrieval
        PRIVATE CUDA::cudart)
    set_target_properties(hybrid_retrieval_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON)
endif()

# ============================================================================
# Executables
# ============================================================================
add_executable(benchmark src/main.cpp)
target_link_libraries(benchmark PRIVATE hybrid_retrieval)

add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE agent_memory)

if(ENABLE_CUDA)
    add_executable(benchmark_gpu src/benchmark_gpu.cpp)
    target_link_libraries(benchmark_gpu PRIVATE hybrid_retrieval_gpu CUDA::cudart)
endif()

# ============================================================================
# Release build
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ============================================================================
# Summary
# ============================================================================
message(STATUS "=== HybridRetrieval build configuration ===")
message(STATUS "  CMAKE_BUILD_TYPE : ${CMAKE_BUILD_TYPE}")
message(STATUS "  ENABLE_CUDA      : ${ENABLE_CUDA}")
message(STATUS "  ENABLE_AVX2      : ${ENABLE_AVX2}")
message(STATUS "  ENABLE_AVX512    : ${ENABLE_AVX512}")
message(STATUS "===========================================")
