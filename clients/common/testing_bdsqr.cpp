/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_bdsqr.hpp>

#define TESTING_BDSQR(...) template void testing_bdsqr<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_BDSQR, FOREACH_SCALAR_TYPE, APPLY_STAMP)
