/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sygvj_hegvj.hpp>

#define TESTING_SYGVJ_HEGVJ(...) template void testing_sygvj_hegvj<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGVJ_HEGVJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
