/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_bttrf_npvt.hpp>

#define TESTING_BTTRF_NPVT(...) template void testing_bttrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_BTTRF_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
