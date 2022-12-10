/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_geblttrf_npvt.hpp>

#define TESTING_GEBLTTRF_NPVT(...) template void testing_geblttrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEBLTTRF_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
