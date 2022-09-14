/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gebd2_gebrd.hpp>

#define TESTING_GEBD2_GEBRD(...) template void testing_gebd2_gebrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEBD2_GEBRD,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
