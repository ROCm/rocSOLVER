/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getri_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETRI_SMALL(rocblas_double_complex, rocblas_double_complex*);
INSTANTIATE_GETRI_SMALL(rocblas_double_complex, rocblas_double_complex* const*);
