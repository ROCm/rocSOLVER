/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_syev_heev.hpp>

#define TESTING_SYEV_HEEV(...) template void testing_syev_heev<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEV_HEEV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
