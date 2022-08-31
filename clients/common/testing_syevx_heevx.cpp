/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_syevx_heevx.hpp>

#define TESTING_SYEVX_HEEVX(...) template void testing_syevx_heevx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEVX_HEEVX, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
