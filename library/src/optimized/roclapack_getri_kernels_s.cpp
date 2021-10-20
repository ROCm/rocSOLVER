/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri_kernels.hpp"

#ifdef OPTIMAL

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETRI_SMALL(float, float*);
INSTANTIATE_GETRI_SMALL(float, float* const*);

#endif
