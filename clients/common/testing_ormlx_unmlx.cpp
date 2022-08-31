/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_ormlx_unmlx.hpp>

#define TESTING_ORMLX_UNMLX(...) template void testing_ormlx_unmlx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORMLX_UNMLX, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
