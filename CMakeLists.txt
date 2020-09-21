cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
cmake_policy(SET CMP0010 NEW)
cmake_policy(SET CMP0025 NEW)

if(NOT CMAKE_VERSION VERSION_LESS 3.15.0)
  cmake_policy(SET CMP0092 NEW)
endif()

project(TinyTorch CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "BUILD_SHARED_LIBS" ON)

include_directories(BEFORE ${PROJECT_SOURCE_DIR})
include_directories(BEFORE ${PROJECT_BINARY_DIR})

add_subdirectory(c10)