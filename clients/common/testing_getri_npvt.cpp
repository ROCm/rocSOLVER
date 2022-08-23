/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_getri_npvt.hpp>

#define TESTING_GETRI_NPVT(...) template void testing_getri_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETRI_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
