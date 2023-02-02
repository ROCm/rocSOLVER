# ########################################################################
# Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
# ########################################################################

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.1)

# Create version header from templated .in file using CMake info
configure_file(include/rocsolver/internal/rocsolver-version.h.in "${PROJECT_BINARY_DIR}/include/rocsolver/internal/rocsolver-version.h")

set(rocsolver_headers_public
  include/rocsolver/rocsolver.h
  include/rocsolver/internal/rocsolver-extra-types.h
  include/rocsolver/internal/rocsolver-aliases.h
  include/rocsolver/internal/rocsolver-functions.h
  ${PROJECT_BINARY_DIR}/include/rocsolver/internal/rocsolver-version.h
  ${PROJECT_BINARY_DIR}/include/rocsolver/internal/rocsolver-export.h
)

# For IDE integration
source_group("Header Files\\Public" FILES ${rocsolver_headers_public})

add_subdirectory(src)
