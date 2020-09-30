cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(c10 CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(C10_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}) # used in cmake_macros.h.in
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/macros/cmake_macros.h.in
    ${CMAKE_BINARY_DIR}/c10/macros/cmake_macros.h)

file(GLOB C10_SRCS
        *.cpp
        core/*.cpp
        macros/*.cpp
        util/*.cpp
        )
file(GLOB_RECURSE C10_HEADERS *.h)
add_library(c10 ${C10_SRCS} ${C10_HEADERS})

find_package(Threads)
target_link_libraries(c10 ${CMAKE_THREAD_LIBS_INIT})

# If building shared library, set dllimport/dllexport proper.
target_compile_options(c10 PRIVATE "-DC10_BUILD_MAIN_LIB")
# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10 PRIVATE "-fvisibility=hidden")
endif()

find_package(Backtrace)
if(Backtrace_FOUND)
  target_include_directories(c10 PRIVATE ${Backtrace_INCLUDE_DIRS})
  target_link_libraries(c10 PRIVATE ${Backtrace_LIBRARIES})
  target_compile_definitions(c10 PRIVATE SUPPORTS_BACKTRACE=1)
else()
  target_compile_definitions(c10 PRIVATE SUPPORTS_BACKTRACE=0)
endif()

target_include_directories(
    c10 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>)

# ---[ Installation
install(TARGETS c10 LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_BINARY_DIR}/c10/macros/cmake_macros.h
        DESTINATION include/c10/macros)
