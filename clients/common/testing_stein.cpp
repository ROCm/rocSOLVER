/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_stein.hpp>

#define TESTING_STEIN(...) template void testing_stein<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_STEIN, FOREACH_SCALAR_TYPE, APPLY_STAMP)
