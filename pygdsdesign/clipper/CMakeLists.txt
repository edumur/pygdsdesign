cmake_minimum_required(VERSION 3.0)

project(clipper)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release type")
set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix")

# The header name clipper.hpp is too generic, so install in a subdirectory
set(CMAKE_INSTALL_INCDIR "${CMAKE_INSTALL_PREFIX}/include/polyclipping")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}")
set(CMAKE_INSTALL_PKGCONFIGDIR "${CMAKE_INSTALL_PREFIX}/share/pkgconfig")
set(PCFILE "${CMAKE_CURRENT_BINARY_DIR}/polyclipping.pc")

# Build shared libraries (.dll/.so) instead of static ones (.lib/.a)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries instead of static ones")

add_library(clipper clipper.cpp)

configure_file(polyclipping.pc.cmakein "${PCFILE}" @ONLY)

install(FILES clipper.hpp DESTINATION "${CMAKE_INSTALL_INCDIR}")
install(TARGETS clipper LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(FILES "${PCFILE}" DESTINATION "${CMAKE_INSTALL_PKGCONFIGDIR}")

set_target_properties(clipper PROPERTIES VERSION 22.0.0 SOVERSION 22)
target_link_libraries(clipper PUBLIC Python3::Python)
