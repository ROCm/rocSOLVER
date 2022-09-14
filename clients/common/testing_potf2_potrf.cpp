/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_potf2_potrf.hpp>

#define TESTING_POTF2_POTRF(...) template void testing_potf2_potrf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_POTF2_POTRF,
            FOREACH_MATRIX_DATA_LAYOUT,
            FOREACH_BLOCKED_VARIANT,
            FOREACH_SCALAR_TYPE,
            APPLY_STAMP)
