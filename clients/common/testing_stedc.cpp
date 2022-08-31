/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_stedc.hpp>

#define TESTING_STEDC(...) template void testing_stedc<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_STEDC, FOREACH_SCALAR_TYPE, APPLY_STAMP)
