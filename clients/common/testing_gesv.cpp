/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gesv.hpp>

#define TESTING_GESV(...) template void testing_gesv<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GESV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
