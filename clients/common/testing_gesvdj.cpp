/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gesvdj.hpp>

#define TESTING_GESVDJ(...) template void testing_gesvdj<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GESVDJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
