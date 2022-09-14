/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_getri.hpp>

#define TESTING_GETRI(...) template void testing_getri<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETRI, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
