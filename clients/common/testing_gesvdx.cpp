/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gesvdx.hpp>

#define TESTING_GESVDX(...) template void testing_gesvdx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GESVDX, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
