cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(levmar_mini CXX C)  # Note: Added C language support

set(PROJECT_VERSION 1.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find all source and header files in the current directory
# file(GLOB SOURCE_FILES CONFIGURE_DEPENDS 
#     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
#     "${CMAKE_CURRENT_SOURCE_DIR}/*.c"
# )
# file(GLOB HEADER_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

set(SOURCE_FILES
    axb.cpp
    lm.cpp
    misc.cpp
    main.cpp
    )

set(HEADER_FILES
    levmar.h
    misc.h
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
