/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_potrs.hpp>

#define TESTING_POTRS(...) template void testing_potrs<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_POTRS, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
