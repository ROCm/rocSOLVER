/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sytf2_sytrf.hpp>

#define TESTING_SYTF2_SYTRF(...) template void testing_sytf2_sytrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_SYTF2_SYTRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
