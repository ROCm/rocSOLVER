/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_syevdj_heevdj.hpp>

#define TESTING_SYEVDJ_HEEVDJ(...) template void testing_syevdj_heevdj<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEVDJ_HEEVDJ, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
