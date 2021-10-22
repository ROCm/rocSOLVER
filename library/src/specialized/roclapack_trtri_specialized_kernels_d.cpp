/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_trtri_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_TRTI2_SMALL(double, double*);
INSTANTIATE_TRTI2_SMALL(double, double* const*);
