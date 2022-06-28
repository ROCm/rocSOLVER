/* ************************************************************************
 * Copyright (c) 2019-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_trsm_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_TRSM_MEM(0, 0, double);
INSTANTIATE_TRSM_LOWER(0, 0, double, double*);
INSTANTIATE_TRSM_UPPER(0, 0, double, double*);

INSTANTIATE_TRSM_MEM(0, 1, double);
INSTANTIATE_TRSM_LOWER(0, 1, double, double*);
INSTANTIATE_TRSM_UPPER(0, 1, double, double*);

INSTANTIATE_TRSM_MEM(1, 0, double);
INSTANTIATE_TRSM_LOWER(1, 0, double, double* const*);
INSTANTIATE_TRSM_UPPER(1, 0, double, double* const*);
