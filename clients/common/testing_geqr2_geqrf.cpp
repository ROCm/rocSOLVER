/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_geqr2_geqrf.hpp>

#define TESTING_GEQR2_GEQRF(...) template void testing_geqr2_geqrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GEQR2_GEQRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
