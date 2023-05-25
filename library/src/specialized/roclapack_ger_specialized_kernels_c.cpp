/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_ger_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GER(false, rocblas_float_complex, rocblas_float_complex*);
INSTANTIATE_GER(true, rocblas_float_complex, rocblas_float_complex*);
INSTANTIATE_GER(false, rocblas_float_complex, rocblas_float_complex* const*);
INSTANTIATE_GER(true, rocblas_float_complex, rocblas_float_complex* const*);
