/* ************************************************************************
 * Copyright (c) 2019-2021 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getf2_kernels.hpp"

#ifdef OPTIMAL

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GETF2_SMALL(rocblas_double_complex, rocblas_double_complex*);

#endif
