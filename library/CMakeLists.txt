# ########################################################################
# Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
# ########################################################################

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.1)

# Create version header from templated .in file using CMake info
configure_file(include/rocsolver/rocsolver-version.h.in "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h")

add_subdirectory(src)
