/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sygvdj_hegvdj.hpp>

#define TESTING_SYGVDJ_HEGVDJ(...) template void testing_sygvdj_hegvdj<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGVDJ_HEGVDJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
