/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_bdsvdx.hpp>

#define TESTING_BDSVDX(...) template void testing_bdsvdx<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_BDSVDX, FOREACH_REAL_TYPE, APPLY_STAMP)
