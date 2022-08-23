/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_orgxl_ungxl.hpp>

#define TESTING_ORGXL_UNGXL(...) template void testing_orgxl_ungxl<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORGXL_UNGXL, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
