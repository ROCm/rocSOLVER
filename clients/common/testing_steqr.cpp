/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_steqr.hpp>

#define TESTING_STEQR(...) template void testing_steqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_STEQR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
