/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gesvd.hpp>

#define TESTING_GESVD(...) template void testing_gesvd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GESVD, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
