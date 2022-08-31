/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_gerq2_gerqf.hpp>

#define TESTING_GERQ2_GERQF(...) template void testing_gerq2_gerqf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_GERQ2_GERQF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
