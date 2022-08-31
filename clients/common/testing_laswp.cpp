/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_laswp.hpp>

#define TESTING_LASWP(...) template void testing_laswp<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LASWP, FOREACH_SCALAR_TYPE, APPLY_STAMP)
