/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_lasyf.hpp>

#define TESTING_LASYF(...) template void testing_lasyf<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_LASYF, FOREACH_SCALAR_TYPE, APPLY_STAMP)
