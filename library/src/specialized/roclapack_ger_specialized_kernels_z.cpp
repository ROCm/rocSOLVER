/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_ger_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GER(false, rocblas_double_complex, rocblas_double_complex*);
INSTANTIATE_GER(true, rocblas_double_complex, rocblas_double_complex*);
INSTANTIATE_GER(false, rocblas_double_complex, rocblas_double_complex* const*);
INSTANTIATE_GER(true, rocblas_double_complex, rocblas_double_complex* const*);
