/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_syevd_heevd.hpp>

#define TESTING_SYEVD_HEEVD(...) template void testing_syevd_heevd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYEVD_HEEVD, FOREACH_MATRIX_DATA_LAYOUT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
