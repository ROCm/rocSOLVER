# ########################################################################
#  Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without modification, 
#  are permitted provided that the following conditions are met:
#  1)Redistributions of source code must retain the above copyright notice, 
#  this list of conditions and the following disclaimer.
#  2)Redistributions in binary form must reproduce the above copyright notice, 
#  this list of conditions and the following disclaimer in the documentation 
#  and/or other materials provided with the distribution
# ########################################################################

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.1)

# Create version header from templated .in file using CMake info
configure_file(include/rocsolver/rocsolver-version.h.in "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h")

add_subdirectory(src)
