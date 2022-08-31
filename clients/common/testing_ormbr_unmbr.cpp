/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_ormbr_unmbr.hpp>

#define TESTING_ORMBR_UNMBR(...) template void testing_ormbr_unmbr<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_ORMBR_UNMBR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
