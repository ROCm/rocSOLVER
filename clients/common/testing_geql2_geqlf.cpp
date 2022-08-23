/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_geql2_geqlf.hpp>

#define TESTING_GEQL2_GEQLF(...) template void testing_geql2_geqlf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEQL2_GEQLF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
