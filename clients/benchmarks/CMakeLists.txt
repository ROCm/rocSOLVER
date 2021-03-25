# ########################################################################
# Copyright (c) 2016-2021 Advanced Micro Devices, Inc.
# ########################################################################

# Linking lapack library requires fortran flags
enable_language( Fortran )
find_package( cblas REQUIRED CONFIG )

add_executable( rocsolver-bench client.cpp )

add_armor_flags( rocsolver-bench "${ARMOR_LEVEL}" )

# External header includes
target_include_directories( rocsolver-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIRS}>
    )

target_link_libraries( rocsolver-bench PRIVATE
  cblas
  lapack
  Threads::Threads
  roc::rocsolver
)

if( CUDA_FOUND )
  target_include_directories( rocsolver-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>
    )
  target_compile_definitions( rocsolver-bench PRIVATE __HIP_PLATFORM_NVCC__ )
  target_link_libraries( rocsolver-bench PRIVATE ${CUDA_LIBRARIES} )
else( )
  target_link_libraries( rocsolver-bench PRIVATE hip::device )
endif( )

# Turn on f16c intrinsics
target_compile_options( rocsolver-bench PRIVATE -mf16c )
target_compile_definitions( rocsolver-bench PRIVATE ROCM_USE_FLOAT16 )

set_target_properties( rocsolver-bench PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

target_link_libraries( rocsolver-bench PRIVATE rocsolver-common clients-common )
