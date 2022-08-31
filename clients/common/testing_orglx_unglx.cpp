/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_orglx_unglx.hpp>

#define TESTING_ORGLX_UNGLX(...) template void testing_orglx_unglx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORGLX_UNGLX, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
