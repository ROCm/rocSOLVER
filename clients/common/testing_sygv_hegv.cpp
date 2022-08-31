/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sygv_hegv.hpp>

#define TESTING_SYGV_HEGV(...) template void testing_sygv_hegv<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGV_HEGV, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
