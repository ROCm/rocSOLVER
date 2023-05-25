/* ************************************************************************
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_ger_specialized_kernels.hpp"

/*************************************************************
    Instantiate template methods using macros
*************************************************************/

INSTANTIATE_GER(false, double, double*);
INSTANTIATE_GER(false, double, double* const*);
