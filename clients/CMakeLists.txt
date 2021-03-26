# ########################################################################
# Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required( VERSION 3.8 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to Release
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
  )
endif()

project( rocsolver-clients LANGUAGES C CXX Fortran )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( NOT TARGET rocsolver )
  find_package( rocsolver REQUIRED CONFIG PATHS ${ROCM_PATH}/rocsolver /opt/rocm/rocsolver )
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} /opt/rocm )

#Don't search for hip package again if it's already found
if(NOT hip_FOUND)
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} )
endif()

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS )
  add_library( clients-common INTERFACE )
  target_include_directories( clients-common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  set( common_source_files
    common/lapack_host_reference.cpp
    rocblascommon/utility.cpp
  )
  prepend_path( "${CMAKE_CURRENT_SOURCE_DIR}/" common_source_files common_source_paths )
  target_sources( clients-common INTERFACE ${common_source_paths} )
  target_compile_definitions( clients-common
    INTERFACE
      ROCSOLVER_COMMON_NAMESPACE=rocsolver_clients
  )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )
endif( )

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )
