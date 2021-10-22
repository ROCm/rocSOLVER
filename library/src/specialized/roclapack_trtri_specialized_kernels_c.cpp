/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_trtri_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_TRTI2_SMALL(rocblas_float_complex, rocblas_float_complex*);
INSTANTIATE_TRTI2_SMALL(rocblas_float_complex, rocblas_float_complex* const*);
