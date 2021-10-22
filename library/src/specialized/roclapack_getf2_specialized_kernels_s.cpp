/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETF2_PANEL(float, float*);
INSTANTIATE_GETF2_PANEL(float, float* const*);

INSTANTIATE_GETF2_SCALE_UPDATE(float, float*);
INSTANTIATE_GETF2_SCALE_UPDATE(float, float* const*);
