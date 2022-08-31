/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gels.hpp>

#define TESTING_GELS(...) template void testing_gels<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GELS, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
