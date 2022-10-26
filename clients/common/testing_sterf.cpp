/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_sterf.hpp>

#define TESTING_STERF(...) template void testing_sterf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_STERF, FOREACH_REAL_TYPE, APPLY_STAMP)
