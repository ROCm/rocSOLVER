/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_orgxr_ungxr.hpp>

#define TESTING_ORGXR_UNGXR(...) template void testing_orgxr_ungxr<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORGXR_UNGXR, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
