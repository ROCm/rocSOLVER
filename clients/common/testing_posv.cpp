/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_posv.hpp>

#define TESTING_POSV(...) template void testing_posv<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_POSV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
