/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_stebz.hpp>

#define TESTING_STEBZ(...) template void testing_stebz<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_STEBZ, FOREACH_REAL_TYPE, APPLY_STAMP)
