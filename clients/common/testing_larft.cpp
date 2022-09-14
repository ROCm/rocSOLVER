/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_larft.hpp>

#define TESTING_LARFT(...) template void testing_larft<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LARFT, FOREACH_SCALAR_TYPE, APPLY_STAMP)
