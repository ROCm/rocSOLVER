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
    ${CBLAS_INCLUDE_DIRS}
    )

target_link_libraries( rocsolver-bench PRIVATE
  cblas
  lapack
  Threads::Threads
  hip::device
  rocsolver-common
  clients-common
  roc::rocsolver
)

# Turn on f16c intrinsics
target_compile_options( rocsolver-bench PRIVATE -mf16c )
target_compile_definitions( rocsolver-bench PRIVATE ROCM_USE_FLOAT16 )
