cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(levmar_mini CXX)

set(PROJECT_VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SOURCE_FILES
    levmar.cpp
    levmar_utils.cpp
    main.cpp
    )

set(HEADER_FILES
    levmar.h
    levmar_utils.h
)

# Generate executable with all source and header files
add_executable(levmar_mini
    ${SOURCE_FILES}
    ${HEADER_FILES}
)

target_include_directories(levmar_mini
    PUBLIC
    ${PROJECT_SOURCE_DIR}
)

target_compile_options(levmar_mini PRIVATE
    # For GCC or Clang
    $<$<CXX_COMPILER_ID:GNU>: -Wall -Wextra>
    $<$<CXX_COMPILER_ID:GNU>: -Wall -Wextra>
    
    # For MSVC
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT levmar_mini)
