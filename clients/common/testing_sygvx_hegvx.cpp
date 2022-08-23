/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sygvx_hegvx.hpp>

#define TESTING_SYGVX_HEGVX(...) template void testing_sygvx_hegvx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGVX_HEGVX, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
