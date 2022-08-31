/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sygsx_hegsx.hpp>

#define TESTING_SYGSX_HEGSX(...) template void testing_sygsx_hegsx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYGSX_HEGSX,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
