/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gelq2_gelqf.hpp>

#define TESTING_GELQ2_GELQF(...) template void testing_gelq2_gelqf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GELQ2_GELQF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
