cmake_minimum_required(VERSION 3.16)
project(HybridRetrieval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- OpenMP (handle Apple Clang with Homebrew libomp) ---
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

# --- hnswlib (header-only) ---
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE ${CMAKE_SOURCE_DIR}/third_party/hnswlib)

# --- Core retrieval library ---
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

# --- Agent memory library ---
add_library(agent_memory STATIC
    src/agent_corpus_generator.cpp
    src/memory_fusion.cpp
    src/memory_store.cpp
    src/temporal_index.cpp
    src/hierarchical_memory.cpp
)
target_include_directories(agent_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(agent_memory PUBLIC hybrid_retrieval)

# --- Generic benchmark executable ---
add_executable(benchmark src/main.cpp)
target_link_libraries(benchmark PRIVATE hybrid_retrieval)

# --- Agent memory demo executable ---
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE agent_memory)

# --- Release build by default ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
