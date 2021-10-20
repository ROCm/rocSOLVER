/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2_kernels.hpp"

#ifdef OPTIMAL

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETF2_SMALL(float, float* const*);

#endif
