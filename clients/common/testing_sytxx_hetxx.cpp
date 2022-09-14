/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sytxx_hetxx.hpp>

#define TESTING_SYTXX_HETXX(...) template void testing_sytxx_hetxx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYTXX_HETXX,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
