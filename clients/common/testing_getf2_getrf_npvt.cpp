/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_getf2_getrf_npvt.hpp>

#define TESTING_GETF2_GETRF_NPVT(...) \
    template void testing_getf2_getrf_npvt<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETF2_GETRF_NPVT,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
