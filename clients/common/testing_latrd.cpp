/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_latrd.hpp>

#define TESTING_LATRD(...) template void testing_latrd<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LATRD, FOREACH_SCALAR_TYPE, APPLY_STAMP)
