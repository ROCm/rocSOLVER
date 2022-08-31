/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_potri.hpp>

#define TESTING_POTRI(...) template void testing_potri<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_POTRI, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
