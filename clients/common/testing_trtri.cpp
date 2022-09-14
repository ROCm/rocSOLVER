/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_trtri.hpp>

#define TESTING_TRTRI(...) template void testing_trtri<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_TRTRI, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
