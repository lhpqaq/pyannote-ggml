cmake_minimum_required(VERSION 3.14)
project(embedding-ggml CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EMBEDDING_COREML "Enable CoreML acceleration" OFF)

if(APPLE)
    set(GGML_METAL_DEFAULT ON)
else()
    set(GGML_METAL_DEFAULT OFF)
endif()
option(GGML_METAL "Enable Metal backend for GPU acceleration" ${GGML_METAL_DEFAULT})

# Add GGML as subdirectory (guarded for multi-project builds)
if(NOT TARGET ggml)
    add_subdirectory(../../ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
endif()

# Add kaldi-native-fbank (disable tests/python builds, guarded for multi-project builds)
if(NOT TARGET kaldi-native-fbank-core)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory(kaldi-native-fbank)
endif()

# --- Core library (model/inference code, reusable by other projects) ---
add_library(embedding-core STATIC
    src/model.cpp
    src/fbank.cpp
)

target_include_directories(embedding-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/kaldi-native-fbank
    ../../ggml/include
)

target_link_libraries(embedding-core PUBLIC ggml kaldi-native-fbank-core)

if(APPLE)
    target_link_libraries(embedding-core PUBLIC "-framework Accelerate")
    target_compile_definitions(embedding-core PUBLIC ACCELERATE_NEW_LAPACK)
endif()

if(GGML_METAL)
    if(APPLE)
        target_compile_definitions(embedding-core PUBLIC GGML_USE_METAL)
        message(STATUS "Metal backend enabled for embedding-ggml")
    else()
        message(WARNING "GGML_METAL requested for embedding-ggml on non-Apple platform; ignored")
    endif()
else()
    message(STATUS "Metal backend disabled for embedding-ggml")
endif()

# --- Executable ---
add_executable(embedding-ggml src/main.cpp)
target_link_libraries(embedding-ggml PRIVATE embedding-core)

set_target_properties(embedding-ggml PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --- Optional CoreML bridge (separate library) ---
if(EMBEDDING_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)

    add_library(embedding-coreml
        src/coreml/coreml_bridge.h
        src/coreml/coreml_bridge.mm
    )
    target_include_directories(embedding-coreml PUBLIC src/coreml)
    target_link_libraries(embedding-coreml PRIVATE
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
    set_target_properties(embedding-coreml PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    target_link_libraries(embedding-ggml PRIVATE embedding-coreml)
    target_compile_definitions(embedding-ggml PRIVATE EMBEDDING_USE_COREML)
    message(STATUS "CoreML acceleration enabled")
endif()

message(STATUS "embedding-ggml project configured successfully")
