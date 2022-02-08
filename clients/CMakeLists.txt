# ########################################################################
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
# ########################################################################

project(rocsolver-clients LANGUAGES C CXX)
if(UNIX)
  enable_language(Fortran)
endif()

# Specify where to put the client binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")

# The rocsolver target will exist if the library is being built along with the clients,
# but if this is a clients-only build, we'll have to search for it.
if(NOT TARGET rocsolver)
  find_package(rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsolver /opt/rocm/rocsolver)
  get_imported_target_location(location roc::rocsolver)
  message(STATUS "Found rocSOLVER: ${location}")
endif()

if(BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  # Linking lapack library requires fortran flags
  find_package(cblas REQUIRED CONFIG)
  target_include_directories(cblas SYSTEM INTERFACE ${CBLAS_INCLUDE_DIRS})

  add_library(clients-common INTERFACE)
  target_include_directories(clients-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(clients-common INTERFACE fmt::fmt)
  set(common_source_files
    common/lapack_host_reference.cpp
    rocblascommon/utility.cpp
  )
  prepend_path("${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths)
  target_sources(clients-common INTERFACE ${common_source_paths})

  if(BUILD_CLIENTS_BENCHMARKS)
    add_subdirectory(benchmarks)
  endif()

  if(BUILD_CLIENTS_TESTS)
    add_subdirectory(gtest)
  endif()
endif()

if(BUILD_CLIENTS_SAMPLES)
  add_subdirectory(samples)
endif()

if(BUILD_CLIENTS_EXTRA_TESTS)
  add_subdirectory(extras)
endif()
