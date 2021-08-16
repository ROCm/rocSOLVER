# ########################################################################
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.13)

# This has to be initialized before the project() command appears
# Set the default build type to Release
if(NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")
endif()

if(NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC})
  set(CMAKE_Fortran_COMPILER  "gfortran")
endif()

# ROCM_BUILD_ID is added to the package name by rocm-cmake. Unsetting it prevents that.
unset(ENV{ROCM_BUILD_ID})

# CMake v3.21.0 can identify ROCMClang, but there are some issues with feature detection.
# For now, disable ROCMClang detection to make it work the same as CMake v3.20 and earlier.
set(__skip_rocmclang ON)

project(rocsolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ROCSOLVER_EMBED_FMT "Hide libfmt symbols" OFF)
option(OPTIMAL "Build specialized kernels for small matrix sizes" ON)

# Add our CMake helper files to the lookup path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

# ########################################################################
# Main
# ########################################################################

# Get rocm-cmake
include(get-rocm-cmake)

# Include the rocm-cmake components we use
include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)

include(os-detection)
get_os_id(OS_ID)
message(STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set(VERSION_STRING "3.16.0")
rocm_setup_version(VERSION ${VERSION_STRING})

# Workaround until llvm and hip CMake modules fix symlink logic in their config files
list(APPEND CMAKE_PREFIX_PATH
  ${ROCM_PATH}
  ${ROCM_PATH}/llvm
  ${ROCM_PATH}/hip
  /opt/rocm
  /opt/rocm/llvm
  /opt/rocm/hip
)

option(BUILD_VERBOSE "Output additional build information" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(DEFAULT_ARMOR_LEVEL 1)
else()
  set(DEFAULT_ARMOR_LEVEL 0)
endif()
set(ARMOR_LEVEL "${DEFAULT_ARMOR_LEVEL}" CACHE STRING "Enables increasingly expensive runtime correctness checks")
include(armor-config)

# This option only works for make, nmake and ninja, but no reason it shouldn't be on all the time
# It creates a compile_commands.json file for use with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# BUILD_SHARED_LIBS is a cmake built-in
# Make it an explicit option such that it shows in cmake-gui
option(BUILD_SHARED_LIBS "Build rocSOLVER as a shared library" ON)

# Include helper functions and wrapper functions
include(util)

option(BUILD_LIBRARY "Build rocSOLVER library" ON)
option_opposite(BUILD_LIBRARY SKIP_LIBRARY)
option(BUILD_CLIENTS_TESTS "Build rocSOLVER unit tests" OFF)
option(BUILD_CLIENTS_BENCHMARKS "Build rocSOLVER benchmarks" OFF)
option(BUILD_CLIENTS_SAMPLES "Build rocSOLVER samples" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)
option(BUILD_CODE_COVERAGE "Build rocSOLVER with code coverage enabled" OFF)
option(WERROR "Treat warnings as errors" OFF)

message(STATUS "Tests: ${BUILD_CLIENTS_TESTS}")
message(STATUS "Benchmarks: ${BUILD_CLIENTS_BENCHMARKS}")
message(STATUS "Samples: ${BUILD_CLIENTS_SAMPLES}")

# Force library install path to lib (CentOS 7 defaults to lib64)
set(CMAKE_INSTALL_LIBDIR "lib" CACHE INTERNAL "Installation directory for libraries" FORCE)

# Query for compiler support of GPU archs
rocm_check_target_ids(OPTIONAL_AMDGPU_TARGETS
  TARGETS
    gfx90a:xnack-
    gfx90a:xnack+
)

# Set this before finding hip so that hip::device has the required arch flags
# added as usage requirements on its interface
set(AMDGPU_TARGETS "gfx900;gfx906:xnack-;gfx908:xnack-;gfx1030;${OPTIONAL_AMDGPU_TARGETS}"
  CACHE STRING "List of specific machine types for library to target")

# Find HIP dependencies
find_package(hip REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm)

find_package(rocblas REQUIRED CONFIG PATHS ${ROCM_PATH})
get_imported_target_location(location roc::rocblas)
message(STATUS "Found rocBLAS: ${location}")

add_subdirectory(common)

if(BUILD_LIBRARY)
  add_subdirectory(library)
endif()

if(BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_SAMPLES)
  add_subdirectory(clients)
endif()

# Code Coverage Build Commands:
#   make coverage_cleanup (clean coverage related files)
#   make coverage GTEST_FILTER=<>
#   make coverage_analysis GTEST_FILTER=<> (analyze tests)
#   make coverage_output (generate html documentation)
if(BUILD_CODE_COVERAGE)
  # Run coverage analysis
  add_custom_target(coverage_analysis
    COMMAND echo Coverage GTEST_FILTER=\${GTEST_FILTER}
    COMMAND ./clients/staging/rocsolver-test --gtest_filter=\"\${GTEST_FILTER}\"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  add_dependencies(coverage_analysis rocsolver)

  # Generate gcov-tool script
  # This little script is generated because the option '--gcov-tool <program name>' of lcov cannot take arguments.
  add_custom_target(coverage_output
    DEPENDS coverage_analysis
    COMMAND mkdir -p lcoverage
    COMMAND echo "\\#!/bin/bash" > llvm-gcov.sh
    COMMAND echo "\\# THIS FILE HAS BEEN GENERATED" >> llvm-gcov.sh
    COMMAND printf "exec /opt/rocm/llvm/bin/llvm-cov gcov $$\\@" >> llvm-gcov.sh
    COMMAND chmod +x llvm-gcov.sh
  )

  # Generate code coverage report
  add_custom_command(TARGET coverage_output
    COMMAND lcov --directory . --base-directory . --gcov-tool ${CMAKE_BINARY_DIR}/llvm-gcov.sh --capture -o lcoverage/raw_main_coverage.info
    COMMAND lcov --remove lcoverage/raw_main_coverage.info "'/opt/*'" "'/usr/*'" -o lcoverage/main_coverage.info
    COMMAND genhtml --ignore-errors source lcoverage/main_coverage.info --output-directory lcoverage
  )
  add_custom_target(coverage DEPENDS coverage_output)

  # Delete gcov data files
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endif()
