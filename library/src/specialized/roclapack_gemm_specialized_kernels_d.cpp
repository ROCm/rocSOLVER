/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_gemm_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GEMM(0, 0, double, double*);
INSTANTIATE_GEMM(0, 1, double, double*);
INSTANTIATE_GEMM(1, 0, double, double* const*);
