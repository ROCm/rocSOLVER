/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sygvd_hegvd.hpp>

#define TESTING_SYGVD_HEGVD(...) template void testing_sygvd_hegvd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGVD_HEGVD, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
