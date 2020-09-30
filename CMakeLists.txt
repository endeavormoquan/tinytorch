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

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if
# modules are built with the main Caffe2 library then one does not need to do
# find caffe2 in the cmake script. One can usually guard it in some way like
#    if(NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
#      find_package(Caffe2 REQUIRED)
#    endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)


#option(HAVE_SOVERSION "Whether to add SOVERSION to the shared objects" OFF)
option(BUILD_SHARED_LIBS "BUILD_SHARED_LIBS" ON)
option(BUILD_TEST "BUILD TEST" ON)

# ---[ Utils
# TODO: merge the following 3 files into cmake/public/utils.cmake.
include(cmake/Utils.cmake)
include(cmake/public/utils.cmake)

# ---[ Version numbers for generated libraries
file(READ version.txt TORCH_DEFAULT_VERSION)
# Strip trailing newline
string(REGEX REPLACE "\n$" "" TORCH_DEFAULT_VERSION "${TORCH_DEFAULT_VERSION}")
if("${TORCH_DEFAULT_VERSION} " STREQUAL " ")
  message(WARNING "Could not get version from base 'version.txt'")
  # If we can't get the version from the version file we should probably
  # set it to something non-sensical like 0.0.0
  set(TORCH_DEFAULT_VERSION, "0.0.0")
endif()
set(TORCH_BUILD_VERSION "${TORCH_DEFAULT_VERSION}" CACHE STRING "Torch build version")

caffe2_parse_version_str(TORCH ${TORCH_BUILD_VERSION})
caffe2_parse_version_str(CAFFE2 ${TORCH_BUILD_VERSION})
set(TORCH_SOVERSION "${TORCH_VERSION_MAJOR}.${TORCH_VERSION_MINOR}")
message("TORCH_SOVERSION: " ${TORCH_SOVERSION})
message("TORCH_VERSION: " ${TORCH_VERSION})

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/torch)

include_directories(BEFORE ${PROJECT_SOURCE_DIR})
include_directories(BEFORE ${PROJECT_BINARY_DIR})

add_subdirectory(c10)

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()



# CMake config for external projects.
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/TorchConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
  @ONLY)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/TorchConfig.cmake.in
  ${PROJECT_BINARY_DIR}/TorchConfig.cmake
  @ONLY)

install(FILES
  ${PROJECT_BINARY_DIR}/TorchConfigVersion.cmake
  ${PROJECT_BINARY_DIR}/TorchConfig.cmake
  DESTINATION share/cmake/Torch)
