/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_larfb.hpp>

#define TESTING_LARFB(...) template void testing_larfb<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LARFB, FOREACH_SCALAR_TYPE, APPLY_STAMP)
