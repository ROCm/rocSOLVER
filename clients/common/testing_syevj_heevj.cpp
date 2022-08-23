/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_syevj_heevj.hpp>

#define TESTING_SYEVJ_HEEVJ(...) template void testing_syevj_heevj<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEVJ_HEEVJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
