/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_larfg.hpp>

#define TESTING_LARFG(...) template void testing_larfg<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LARFG, FOREACH_SCALAR_TYPE, APPLY_STAMP)
