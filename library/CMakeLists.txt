# ########################################################################
# Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
# ########################################################################

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.1)

# Create version header from templated .in file using CMake info
configure_file(include/rocsolver/rocsolver-version.h.in "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h")

set(rocsolver_headers_public
  include/rocsolver/rocsolver.h
  include/rocsolver/rocsolver-extra-types.h
  include/rocsolver/rocsolver-aliases.h
  include/rocsolver/rocsolver-functions.h
  ${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h
  ${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-export.h
)

# For IDE integration
source_group("Header Files\\Public" FILES ${rocsolver_headers_public})

add_subdirectory(src)
