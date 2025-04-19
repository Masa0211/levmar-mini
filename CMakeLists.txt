cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(levmar_mini CXX C)  # Note: Added C language support

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

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT levmar_mini)
