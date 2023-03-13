/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <testing_csrrf_analysis.hpp>

#define TESTING_CSRRF_ANALYSIS(...) template void testing_csrrf_analysis<__VA_ARGS__>(Arguments&);

INSTANTIATE(TESTING_CSRRF_ANALYSIS, FOREACH_SCALAR_TYPE, APPLY_STAMP)
