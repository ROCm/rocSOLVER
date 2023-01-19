/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_geblttrs_npvt.hpp>

#define TESTING_GEBLTTRS_NPVT(...) template void testing_geblttrs_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEBLTTRS_NPVT, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
