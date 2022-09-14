/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_getrs.hpp>

#define TESTING_GETRS(...) template void testing_getrs<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETRS, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
