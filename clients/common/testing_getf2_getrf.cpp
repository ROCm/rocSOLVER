/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_getf2_getrf.hpp>

#define TESTING_GETF2_GETRF(...) template void testing_getf2_getrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GETF2_GETRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
