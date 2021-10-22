/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETF2_PANEL(rocblas_float_complex, rocblas_float_complex*);
INSTANTIATE_GETF2_PANEL(rocblas_float_complex, rocblas_float_complex* const*);

INSTANTIATE_GETF2_SCALE_UPDATE(rocblas_float_complex, rocblas_float_complex*);
INSTANTIATE_GETF2_SCALE_UPDATE(rocblas_float_complex, rocblas_float_complex* const*);
