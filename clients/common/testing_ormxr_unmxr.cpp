/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_ormxr_unmxr.hpp>

#define TESTING_ORMXR_UNMXR(...) template void testing_ormxr_unmxr<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORMXR_UNMXR, FOREACH_SCALAR_TYPE, FOREACH_BLOCKED_VARIANT, APPLY_STAMP)
