/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_trtri_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_TRTI2_SMALL(rocblas_double_complex, rocblas_double_complex*);
INSTANTIATE_TRTI2_SMALL(rocblas_double_complex, rocblas_double_complex* const*);
